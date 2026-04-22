import paths
import json
import types
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'DiffSynth-Studio'))
from cvpr_2026_workshop_wm_track.pipelines import GigaBrain0Pipeline
from cvpr_2026_workshop_wm_track.model_config import model_config, DATA_DIR
import torch
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from cvpr_2026_workshop_wm_track.utils import resize_with_pad, split_data
from cvpr_2026_workshop_wm_track.image_utils import concat_images_grid
from cvpr_2026_workshop_wm_track.sockets import RobotInferenceClient
from PIL import Image
import numpy as np
import argparse
import multiprocessing 
import os
import glob
from multiprocessing import Process
import pickle
from einops import rearrange
from typing import Any
from tqdm import tqdm
import imageio
from decord import VideoReader

def get_policy(
    ckpt_dir: str,
    tokenizer_model_path: str,
    fast_tokenizer_path: str,
    embodiment_id: int,
    norm_stats_path: str,
    delta_mask: list[bool],
    original_action_dim: int,
    depth_img_prefix_name: str | None = None,
) -> GigaBrain0Pipeline:
    """Build and initialize a GigaBrain0 policy for inference.

    Args:
        ckpt_dir: Path to the model checkpoint directory.
        tokenizer_model_path: Path to the tokenizer model.
        fast_tokenizer_path: Path to the fast tokenizer model.
        embodiment_id: Embodiment identifier of the robot/task.
        norm_stats_path: Path to the JSON file containing normalization stats.
        delta_mask: Boolean mask indicating which action dimensions are delta-controlled.
        original_action_dim: Expected original action vector dimension.
        depth_img_prefix_name: Optional prefix for depth image keys when depth is enabled.

    Returns:
        Initialized GigaBrain0Pipeline with CUDA device and compiled graph. Also binds
        a convenience `inference` method to the returned instance.
    """
    with open(norm_stats_path, 'r') as f:
        norm_stats_data = json.load(f)['norm_stats']

    pipe = GigaBrain0Pipeline(
        model_path=ckpt_dir,
        tokenizer_model_path=tokenizer_model_path,
        fast_tokenizer_path=fast_tokenizer_path,
        embodiment_id=embodiment_id,
        state_norm_stats=norm_stats_data['observation.state'],
        action_norm_stats=norm_stats_data['action'],
        delta_mask=delta_mask,
        original_action_dim=original_action_dim,
        depth_img_prefix_name=depth_img_prefix_name,
    )
    pipe.to('cuda')
    # pipe.compile()

    def inference(self, data: dict[str, Any]) -> torch.Tensor:
        """Run policy inference to get the predicted action.

        Args:
            data: Input dictionary containing observation images, optional depth images,
                a task string under key 'task', and a state tensor under
                'observation.state'.

        Returns:
            Predicted action tensor produced by the policy.
        """
        images = {
            'observation.images.cam_high': data['observation.images.cam_high'],
            'observation.images.cam_left_wrist': data['observation.images.cam_left_wrist'],
            'observation.images.cam_right_wrist': data['observation.images.cam_right_wrist'],
        }
        if pipe.enable_depth_img and 'observation.depth_images.cam_high' in data:
            images['observation.depth_images.cam_high'] = data['observation.depth_images.cam_high']
        if pipe.enable_depth_img and 'observation.depth_images.cam_left_wrist' in data:
            images['observation.depth_images.cam_left_wrist'] = data['observation.depth_images.cam_left_wrist']
        if pipe.enable_depth_img and 'observation.depth_images.cam_right_wrist' in data:
            images['observation.depth_images.cam_right_wrist'] = data['observation.depth_images.cam_right_wrist']

        task = data['task']
        state = data['observation.state']

        pred_action = pipe(images, task, state)

        return pred_action

    pipe.inference = types.MethodType(inference, pipe)

    return pipe

def make_infer_data(camera_high, camera_left, camera_right, task_name, qpos):
    assert qpos.shape == (14,)
    camera_high_chw = rearrange(camera_high, 'h w c -> c h w')
    camera_left_chw = rearrange(camera_left, 'h w c -> c h w')
    camera_right_chw = rearrange(camera_right, 'h w c -> c h w')
    observation = {
        'observation.state': torch.from_numpy(qpos).to(torch.float32),
        'observation.images.cam_high': torch.from_numpy(camera_high_chw),
        'observation.images.cam_left_wrist': torch.from_numpy(camera_left_chw),
        'observation.images.cam_right_wrist': torch.from_numpy(camera_right_chw),
        'task': task_name,
    }
    return observation

class InferenceEngine:
    def __init__(self, transformer_model_path, device, dtype=torch.bfloat16, num_views=3, mode='offline', seed=1,
                 wm_frame_per_time=24, num_inference_steps=50, cfg_scale=5.0,
                 render_subsample_step=1, negative_prompt=None, debug_dir=None,
                 no_control_video=False):
        assert mode in ['offline', 'online'], f"mode must be offline or online, but got {mode}"
        torch.cuda.set_device(device)
        device = "cuda"
        print(f"Loading model from {transformer_model_path}")

        base_model_dir = "/path/to/wan_models/Wan2.1-14B-Control"
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=dtype,
            device=device,
            model_configs=[
                ModelConfig(path=[transformer_model_path]),
                ModelConfig(path=f"{base_model_dir}/models_t5_umt5-xxl-enc-bf16.pth"),
                ModelConfig(path=f"{base_model_dir}/Wan2.1_VAE.pth"),
                ModelConfig(path=f"{base_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ],
            tokenizer_config=ModelConfig(path=f"{base_model_dir}/google/umt5-xxl"),
        )
        self.mode = mode

        self.dst_size = (224, 224)
        self.wm_frame_per_time = wm_frame_per_time
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = cfg_scale
        self.num_views = num_views
        self.seed = seed
        self.render_subsample_step = render_subsample_step
        self.negative_prompt = negative_prompt or "vivid colors, overexposed, static, blurry details, subtitles, stylized, artwork, painting, still frame, overall grayish, worst quality, low quality, JPEG compression artifacts, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn face, deformed, disfigured, malformed limbs, fused fingers, frozen frame, cluttered background, three legs, crowded background, walking backwards"
        self.debug_dir = debug_dir
        self.no_control_video = no_control_video
        if no_control_video:
            print("  → [NO CONTROL VIDEO] WM will generate purely from prompt + reference image")

        
    def activate_policy(self, policy_ckpt_dir, norm_stats_path):
        tokenizer_model_path = model_config['paligemma']
        fast_tokenizer_path = model_config['fast-tokenizer']
        self.policy = get_policy(
            ckpt_dir=policy_ckpt_dir,
            norm_stats_path=norm_stats_path,
            tokenizer_model_path=tokenizer_model_path,
            fast_tokenizer_path=fast_tokenizer_path,
            embodiment_id=0,
            delta_mask=[True, True, True, True, True, True, False, True, True, True, True, True, True, False],
            original_action_dim=14,
            depth_img_prefix_name=None,
        )

    def activate_simulator_client(self, sim_ip, sim_port):
        self.sim_api = RobotInferenceClient(host=sim_ip, port=sim_port)
    
    def wm_inference_per_time(self, replay, ref_image, task=None):
        control_video = None if self.no_control_video else replay
        with torch.no_grad():
            output_frames = self.pipe(
                prompt=task,
                negative_prompt=self.negative_prompt,
                reference_image=ref_image,
                control_video=control_video,
                height=self.dst_size[1],
                width=self.dst_size[0] * self.num_views,
                num_frames=self.wm_frame_per_time + 1,
                cfg_scale=self.guidance_scale,
                num_inference_steps=self.num_inference_steps,
                seed=self.seed,
                output_type='quantized',
            )
        torch.cuda.empty_cache()
        # output_frames: list of PIL.Image (output_type='quantized' already returns PIL Images)
        return output_frames
    
    def resize_image(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        image = Image.fromarray(image).convert("RGB").resize((self.dst_size[0], self.dst_size[1]), resample=Image.Resampling.BICUBIC)
        # image = Image.fromarray(resize_with_pad(image, self.dst_size[0], self.dst_size[1]))
        return image

    def resize_images(self, images):
        images = [self.resize_image(image) for image in images]
        return images
    
    def wm_inference(self, ref_images, action_images, task=None):
        img_front = ref_images['front']
        img_left = ref_images['left']
        img_right = ref_images['right']
        img_front = self.resize_image(img_front)
        img_left = self.resize_image(img_left)
        img_right = self.resize_image(img_right)
        ref_image = concat_images_grid(
            [img_front, img_left, img_right], cols=3, pad=0)

        front_replay_images = action_images['front_replay']
        left_replay_images = action_images['left_replay']
        right_replay_images = action_images['right_replay']
        front_replay_images = self.resize_images(front_replay_images)
        left_replay_images = self.resize_images(left_replay_images)
        right_replay_images = self.resize_images(right_replay_images)
        action_images = []
        for i in range(len(front_replay_images)):
            replay_image = concat_images_grid(
                [front_replay_images[i], left_replay_images[i], right_replay_images[i]], cols=3, pad=0
            )
            action_images.append(replay_image)

        action_chunk = len(action_images)
        wm_inference_time = (action_chunk - 1) // self.wm_frame_per_time
        if action_chunk % self.wm_frame_per_time != 0:
            Warning(f"action_chunk {action_chunk} is not divisible by wm_frame_per_time {self.wm_frame_per_time}")
        print(f"wm_inference_time {wm_inference_time}")
        all_output_images = []
        replay_condition_images = []
        for step in tqdm(range(wm_inference_time)):
            start = step * self.wm_frame_per_time
            end = (step + 1) * self.wm_frame_per_time + 1
            action_images_chunk = action_images[start:end]
            output_images = self.wm_inference_per_time(action_images_chunk, ref_image, task)
            if step == wm_inference_time - 1:
                output_images = output_images
                replay_condition_images.extend(action_images_chunk)
            else:
                output_images = output_images[:-1]
                replay_condition_images.extend(action_images_chunk[:-1])
            all_output_images.extend(output_images)
            ref_image = output_images[-1]
            assert len(all_output_images) == len(replay_condition_images)
        condition_images_dict = {
            'replay': replay_condition_images,
        }
        return all_output_images, condition_images_dict
    
    def get_action(self, img_front, img_left, img_right, state, task_name):
        if isinstance(img_front, Image.Image):
            img_front = np.array(img_front)
            img_left = np.array(img_left)
            img_right = np.array(img_right)
        img_front = resize_with_pad(img_front, 224, 224).astype(np.float32) / 255.0
        img_left = resize_with_pad(img_left, 224, 224).astype(np.float32) / 255.0
        img_right = resize_with_pad(img_right, 224, 224).astype(np.float32) / 255.0
        obs = make_infer_data(
            img_front,
            img_left,
            img_right,
            task_name,
            state,
        )
        action = self.policy.inference(obs)
        return action

    def render_qpos(self, action):
        render_frames = self.sim_api.inference({"action": action})
        front_replay_images = VideoReader(render_frames['sim_front_rgb'])
        left_replay_images = VideoReader(render_frames['sim_left_rgb'])
        right_replay_images = VideoReader(render_frames['sim_right_rgb'])
        front_replay_images = [Image.fromarray(front_replay_images[i].asnumpy()) for i in
                               range(len(front_replay_images))]
        left_replay_images = [Image.fromarray(left_replay_images[i].asnumpy()) for i in range(len(left_replay_images))]
        right_replay_images = [Image.fromarray(right_replay_images[i].asnumpy()) for i in
                               range(len(right_replay_images))]
        # Subsample rendered frames and truncate to wm_frame_per_time + 1
        subsample_step = self.render_subsample_step
        max_frames = self.wm_frame_per_time + 1
        front_replay_images = front_replay_images[::subsample_step][:max_frames]
        left_replay_images = left_replay_images[::subsample_step][:max_frames]
        right_replay_images = right_replay_images[::subsample_step][:max_frames]
        print(f"  → Rendered frames subsampled: {len(front_replay_images)} frames (every {subsample_step} frames, max {max_frames})")
        return {
            'front_replay': front_replay_images,
            'left_replay': left_replay_images,
            'right_replay': right_replay_images,
        }
    
    def crop_three_view_images(self, ref_image):
        img_front = ref_image.crop((0, 0, self.dst_size[0], self.dst_size[1]))
        img_left = ref_image.crop((self.dst_size[0], 0, self.dst_size[0] * 2, self.dst_size[1]))
        img_right = ref_image.crop((self.dst_size[0] * 2, 0, self.dst_size[0] * 3, self.dst_size[1]))
        return img_front, img_left, img_right

    def interaction(self, ref_images, state, task,
                    max_interactions=15, episode_name=None,
        ):
        """Run interaction loop with per-round prompt support.

        Args:
            task: Either a single prompt string (used for all rounds) or a list
                of prompt strings (one per round). If the list is shorter than
                max_interactions, the last prompt is reused for remaining rounds.
            episode_name: Optional episode identifier used for debug output subdirectory.
        """
        img_front = ref_images['front']
        img_left = ref_images['left']
        img_right = ref_images['right']
        img_front = self.resize_image(img_front)
        img_left = self.resize_image(img_left)
        img_right = self.resize_image(img_right)
        # Auto-compute pos_lookahead_step based on wm_frame_per_time and render_subsample_step
        pos_lookahead_step = (self.wm_frame_per_time + 1) * self.render_subsample_step - 1
        print(f"  → Auto-computed pos_lookahead_step={pos_lookahead_step} "
              f"(wm_frame_per_time={self.wm_frame_per_time}, render_subsample_step={self.render_subsample_step})")

        # Prepare debug output directory for this episode
        episode_debug_dir = None
        if self.debug_dir is not None:
            episode_debug_dir = os.path.join(self.debug_dir, episode_name or "unknown")
            os.makedirs(episode_debug_dir, exist_ok=True)
            print(f"  → Debug output dir: {os.path.abspath(episode_debug_dir)}")

        # Normalize task to a list of prompts, one per interaction round
        if isinstance(task, str):
            prompt_list = [task] * max_interactions
        else:
            prompt_list = list(task)
            if len(prompt_list) < max_interactions:
                prompt_list.extend([prompt_list[-1]] * (max_interactions - len(prompt_list)))

        all_output_images = []
        replay_condition_images = []
        for step in tqdm(range(max_interactions)):
            current_prompt = prompt_list[step]
            print(f"\n{'='*60}")
            print(f"Interaction step {step}/{max_interactions}")
            print(f"Prompt: {current_prompt}")
            print(f"{'='*60}")

            # Save VLA input images (the three views fed to the policy)
            if episode_debug_dir is not None:
                step_dir = os.path.join(episode_debug_dir, f"step_{step:03d}")
                os.makedirs(step_dir, exist_ok=True)
                img_front.save(os.path.join(step_dir, "vla_input_front.png"))
                img_left.save(os.path.join(step_dir, "vla_input_left.png"))
                img_right.save(os.path.join(step_dir, "vla_input_right.png"))

            actions = self.get_action(img_front, img_left, img_right, state, current_prompt)
            actions = actions[:pos_lookahead_step]
            future_state = np.concatenate([state[None, :], actions], axis=0)
            action_images = self.render_qpos(future_state)
            output_images, condition_images_dict = self.wm_inference(ref_images, action_images, task=current_prompt)

            # Save WM output chunk and replay condition as videos
            if episode_debug_dir is not None:
                wm_chunk_path = os.path.join(step_dir, "wm_output_chunk.mp4")
                replay_chunk_path = os.path.join(step_dir, "replay_condition.mp4")
                imageio.mimsave(wm_chunk_path, [np.array(img) for img in output_images], fps=24)
                imageio.mimsave(replay_chunk_path, [np.array(img) for img in condition_images_dict['replay']], fps=24)

            all_output_images.extend(output_images)
            replay_condition_images.extend(condition_images_dict['replay'])
            state = future_state[-1]
            ref_image = output_images[-1]
            img_front, img_left, img_right = self.crop_three_view_images(ref_image)
            ref_images = {
                'front': img_front,
                'left': img_left,
                'right': img_right,
            }
            assert len(all_output_images) == len(replay_condition_images)
        
        condition_images_dict = {
            'replay': replay_condition_images,
        }
        return all_output_images, condition_images_dict


def inference(args, device, world_size, rank):
    mode = args.mode
    if mode == 'offline':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'video_quality')
    elif mode == 'online':
        eval_data_dir = os.path.join(args.data_dir, args.task, 'evaluator')
    else:
        raise ValueError(f"mode {mode} is not supported.")
    episode_list = sorted(os.listdir(eval_data_dir))
    if args.episode_group is not None:
        episodes_per_group = args.episodes_per_group
        group_start = args.episode_group * episodes_per_group
        group_end = group_start + episodes_per_group
        episode_list = episode_list[group_start:group_end]
    data_list = split_data(episode_list, world_size, rank)
    # Load task config from task_config_file if provided
    task_config = None
    if args.task_config_file is not None:
        with open(args.task_config_file, 'r') as tcf:
            all_task_configs = json.load(tcf)
        if args.task in all_task_configs:
            task_config = all_task_configs[args.task]
            print(f"Loaded task config for '{args.task}' from {args.task_config_file}")
        else:
            print(f"Warning: task '{args.task}' not found in {args.task_config_file}, falling back to defaults")

    # Resolve negative_prompt: CLI > task_config_file > prompt_plan_file > default
    resolved_negative_prompt = args.negative_prompt
    if resolved_negative_prompt is None and task_config is not None and 'negative_prompt' in task_config:
        resolved_negative_prompt = task_config['negative_prompt']
    if resolved_negative_prompt is None and args.prompt_plan_file is not None:
        with open(args.prompt_plan_file, 'r') as npf:
            plan_data = json.load(npf)
            if isinstance(plan_data, dict) and 'negative_prompt' in plan_data:
                resolved_negative_prompt = plan_data['negative_prompt']

    # Resolve max_interactions: CLI (if explicitly set) > task_config_file > CLI default
    resolved_max_interactions = args.max_interactions
    if task_config is not None and 'max_interactions' in task_config:
        resolved_max_interactions = task_config['max_interactions']

    inference_engine = InferenceEngine(
        args.transformer_model_path, device=device, mode=mode, seed=args.seed,
        wm_frame_per_time=args.wm_frame_per_time,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        render_subsample_step=args.render_subsample_step,
        negative_prompt=resolved_negative_prompt,
        debug_dir=args.debug_dir,
        no_control_video=args.no_control_video,
    )
    output_dir = os.path.join(args.output_dir, 'video_quality_eval' if mode == 'offline' else 'evaluator_test', args.task)
    os.makedirs(output_dir, exist_ok=True)
    if mode == 'online':
        inference_engine.activate_policy(args.policy_ckpt_dir, args.policy_norm_stats_path)
        inference_engine.activate_simulator_client(args.simulator_ip, args.simulator_port)
    for episode_name in data_list:
        episode_dir = os.path.join(eval_data_dir, episode_name)
        print("Episode name: {}".format(episode_name), episode_dir)

        if os.path.exists(os.path.join(output_dir, '{}.mp4'.format(episode_name))):
            continue

        if not os.path.isdir(episode_dir):
            continue

        if mode == 'offline':
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')
            front_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_high.mp4'))
            left_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_left_wrist.mp4'))
            right_replay_images = VideoReader(os.path.join(episode_dir, 'simulator_cam_right_wrist.mp4'))
            front_replay_images = [Image.fromarray(front_replay_images[i].asnumpy()) for i in
                                range(len(front_replay_images))]
            left_replay_images = [Image.fromarray(left_replay_images[i].asnumpy()) for i in range(len(left_replay_images))]
            right_replay_images = [Image.fromarray(right_replay_images[i].asnumpy()) for i in
                                range(len(right_replay_images))]
            action_images = {
                'front_replay': front_replay_images,
                'left_replay': left_replay_images,
                'right_replay': right_replay_images,
            }
            ref_images = {
                'front': cam_high,
                'left': cam_left_wrist,
                'right': cam_right_wrist,
            }
            all_output_images, condition_images_dict = inference_engine.wm_inference(ref_images, action_images)

        elif mode == 'online':
            cam_high = Image.open(os.path.join(episode_dir, 'cam_high.png')).convert('RGB')
            cam_left_wrist = Image.open(os.path.join(episode_dir, 'cam_left_wrist.png')).convert('RGB')
            cam_right_wrist = Image.open(os.path.join(episode_dir, 'cam_right_wrist.png')).convert('RGB')
            ref_images = {
                'front': cam_high,
                'left': cam_left_wrist,
                'right': cam_right_wrist,
            }
            initial_state = pickle.load(open(os.path.join(episode_dir, 'initial_state.pkl'), 'rb'))
            meta = json.load(open(os.path.join(episode_dir, 'meta.json')))
            # Support per-round prompts: priority: task_config_file > prompt_plan_file > prompt_plan CLI > meta.json prompt_plan > meta.json prompt
            if task_config is not None and 'prompt_plan' in task_config:
                prompt_plan = task_config['prompt_plan']
                print(f"prompt_plan (from task_config '{args.task}'): {[p[:60] + '...' for p in prompt_plan]}")
            elif args.prompt_plan_file is not None:
                with open(args.prompt_plan_file, 'r') as pf:
                    plan_data = json.load(pf)
                if isinstance(plan_data, dict):
                    prompt_plan = plan_data['prompt_plan']
                else:
                    prompt_plan = plan_data
                print(f"prompt_plan (from file {args.prompt_plan_file}): {[p[:60] + '...' for p in prompt_plan]}")
            elif args.prompt_plan is not None:
                prompt_plan = json.loads(args.prompt_plan)
                print(f"prompt_plan (from CLI): {[p[:60] + '...' for p in prompt_plan]}")
            elif 'prompt_plan' in meta:
                prompt_plan = meta['prompt_plan']
                print(f"prompt_plan (from meta.json): {[p[:60] + '...' for p in prompt_plan]}")
            else:
                prompt_plan = meta['prompt']
                print(f"prompt: {prompt_plan}")
            all_output_images, condition_images_dict = inference_engine.interaction(ref_images, initial_state, prompt_plan, resolved_max_interactions, episode_name=episode_name)
        
        replay_condition_images = condition_images_dict['replay']
        vis_images = []
        save_length = min(len(all_output_images), len(replay_condition_images))
        for k in range(save_length):
            vis_image = [all_output_images[k], replay_condition_images[k]]
            vis_image = concat_images_grid(vis_image, cols=1, pad=2)
            vis_images.append(vis_image)
        save_path = os.path.join(output_dir, '{}.mp4'.format(episode_name))
        concat_save_path = os.path.join(output_dir, 'concat_{}.mp4'.format(episode_name))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        imageio.mimsave(save_path, all_output_images, fps=24)
        imageio.mimsave(concat_save_path, vis_images, fps=24)
                             

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--transformer_model_path', type=str, default='wan2.2-5b-diffusers')
    parser.add_argument('--device_list', type=str, default='0,1,2,3')
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--mode', type=str, default='offline')
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--task", type=str, default='task4')
    parser.add_argument("--output_dir", type=str, default='outputs/baseline_wm')

    # online inference parameter
    parser.add_argument("--simulator_ip", type=str, default='127.0.0.1')
    parser.add_argument("--simulator_port", type=str, default='9151')
    parser.add_argument("--policy_ckpt_dir", type=str, default=None)
    parser.add_argument("--policy_norm_stats_path", type=str, default=None)
    parser.add_argument("--max_interactions", type=int, default=15)
    parser.add_argument("--episode_group", type=int, default=None,
                        help="Episode group index (0-based). Each group contains episodes_per_group episodes. "
                             "If not set, run all episodes.")
    parser.add_argument("--episodes_per_group", type=int, default=5,
                        help="Number of episodes per group. Default 5. "
                             "group i -> episode[i*N : i*N+N] where N=episodes_per_group.")

    # World model inference parameters
    parser.add_argument("--wm_frame_per_time", type=int, default=24,
                        help="Number of frames per world model inference chunk")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                        help="Number of denoising steps")
    parser.add_argument("--cfg_scale", type=float, default=5.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--render_subsample_step", type=int, default=1,
                        help="Subsample rendered sim frames: take every Nth frame (1 = no subsampling)")

    # Per-round prompt plan
    parser.add_argument("--prompt_plan", type=str, default=None,
                        help='JSON list of per-round prompts, e.g. \'["open the lid", "pour the chips"]\'.')
    parser.add_argument("--prompt_plan_file", type=str, default=None,
                        help="Path to a JSON file containing per-round prompts (list or dict with "
                             "'prompt_plan' and optional 'negative_prompt'). Takes priority over --prompt_plan.")
    parser.add_argument("--task_config_file", type=str, default=None,
                        help="Path to a JSON file containing per-task configs (keyed by task name). "
                             "Each task entry can have 'prompt_plan', 'negative_prompt', and 'max_interactions'. "
                             "Takes highest priority over prompt_plan_file and CLI prompt_plan.")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt for world model inference. Takes priority over "
                             "negative_prompt in prompt_plan_file. If not set, uses default.")
    parser.add_argument("--debug_dir", type=str, default=None,
                        help="Directory to save per-step debug outputs (VLA input images, "
                             "WM output chunks, replay conditions). If not set, debug saving is disabled.")
    parser.add_argument("--no_control_video", action="store_true", default=False,
                        help="Disable control_video input to WM. The model will generate purely "
                             "from prompt + reference image (I2V mode).")
    args = parser.parse_args()

    if args.policy_ckpt_dir is None:
        args.policy_ckpt_dir = model_config[f'cvpr-2026-worldmodel-track-model-{args.task}']

    if args.policy_norm_stats_path is None:
        args.policy_norm_stats_path = os.path.join(model_config[f'cvpr-2026-worldmodel-track-model-{args.task}'], 'norm_stat_gigabrain.json')

    # inference(args, "cuda:0", 1, 0)
    # exit()

    devices = args.device_list.split(',')
    multiprocessing.set_start_method('spawn')
    process_list = []
    gpu_ids = devices
    world_size = len(gpu_ids)
    for i in range(world_size):
        device = f'cuda:{gpu_ids[i]}'
        rank = i
        process = Process(target=inference, args=(args, device, world_size, rank))
        process.start()
        process_list.append(process)
    for process in process_list:
        process.join()
    
    print("Inference done")





        
