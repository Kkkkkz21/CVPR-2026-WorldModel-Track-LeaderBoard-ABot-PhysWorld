from ..model_config import model_config, DATA_DIR
import os

# 基础参数
dst_size = (224, 224)
num_frames = 8
rollout = 4
total_frames = num_frames * rollout + 1
fps = 16
project_dir = "/path/to/experiments/baseline_wm/task1_test/"

# 【关键修改 1】自动检测逻辑
# 如果是在多机环境下，通常由启动命令设置 WORLD_SIZE 和 RANK
# 我们不需要在这里指定 gpu_ids，DeepSpeed 会自动接管所有可见设备
current_gpu_count = int(os.environ.get("LOCAL_WORLD_SIZE", -1)) 
# 如果本地没检测到，默认留空让 DeepSpeed 自行探测所有可见 CUDA 设备
gpu_ids_setting = None 

# =============================================================
config = dict(
    project_dir=project_dir,
    runners=["cvpr_2026_workshop_wm_track.trainer.BaselineWMTrainer"],
    launch=dict(
        # 【修改】注释掉或删除硬编码的 gpu_ids
        # gpu_ids=[0, 1, 2, 3], 
        gpu_ids=gpu_ids_setting, # 设置为 None 或省略，表示使用所有可见设备
        
        distributed_type='DEEPSPEED',
        deepspeed_config=dict(
            deepspeed_config_file='accelerate_configs/zero2.json',
        ),
        until_completion=True,
        # 【可选】如果是某些特定框架，可能需要显式开启多机标志，
        # 但通常 DeepSpeed 只要检测到 MASTER_ADDR 就会自动进入多机模式
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=[
                f"{DATA_DIR}/task1/train"
            ],
            batch_size_per_gpu=1, # 注意：总批次 = batch_size_per_gpu * 总卡数
            num_workers=2,
            filter=dict(
                mode='overall_func',
                func='cvpr_2026_workshop_wm_track.configs.baseline_wm_task1.filter_data',
                dst_size=dst_size,
                min_num_frames=num_frames,
                min_area=dst_size[0] * dst_size[1],
                min_size=4,
            ),
            transform=dict(
                type='WMTransforms',
                dst_size=dst_size,
                num_frames=total_frames,
                sub_frames=num_frames,
                image_cfg=dict(
                    mask_generator=dict(
                        max_ref_frames=1,
                        start=1,
                        factor=4,
                    ),
                ),
                is_train=True,
                fps=fps,
                max_stride=4,
                num_views=3
            ),
            sampler=dict(
                type='BucketSampler',
            ),
            collator=dict(
                is_equal=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=model_config['wan2.2-5b-diffusers'],
        flow_shift=5.0,
        expand_timesteps=True,
        view_dir=project_dir,
        sub_frames=num_frames,
        rollout=rollout,
    ),
    optimizers=dict(
        type='CAME8Bit',
        lr=2 ** (-14.5),
        weight_decay=1e-2,
    ),
    schedulers=dict(
        type='ConstantScheduler',
    ),
    train=dict(
        resume=True,
        max_epochs=100,
        # 【重要提示】多机多卡时，梯度累积步数可能需要调整以保持等效全局 Batch Size
        # 原单机 4 卡：Global BS = 1 * 4 * 4 = 16
        # 若多机共 32 卡：Global BS = 1 * 32 * 4 = 128 (变大了 8 倍)
        # 建议：如果希望保持全局 Batch Size 不变，需动态调整 gradient_accumulation_steps
        # 这里先保持固定值，你可以在启动脚本中通过环境变量覆盖它
        gradient_accumulation_steps=4, 
        
        mixed_precision='bf16',
        checkpoint_interval=10,
        checkpoint_total_limit=-1,
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=1,
        with_ema=True,
        activation_checkpointing=False,
        activation_class_names=["WanTransformerBlock"],
    ),
    test=dict(),
)

def filter_data(
    all_data_list,
    dst_size=(1280, 704),
    min_num_frames=121,
    multiple=16,
    min_area=-1,
    min_size=1,
):
    from giga_datasets import image_utils

    video_info_dict = dict()
    for n, data_list in enumerate(all_data_list):
        for m, data_dict in enumerate(data_list):
            video_info = dst_size
            if video_info not in video_info_dict:
                video_info_dict[video_info] = []
            video_info_dict[video_info].append((n, m))

    new_all_data_list = [[] for _ in range(len(all_data_list))]
    bucket_index = 0
    for video_info, data_indexes in video_info_dict.items():
        if len(data_indexes) >= min_size:
            for n, m in data_indexes:
                data_dict = all_data_list[n][m]
                data_dict['bucket_index'] = bucket_index
                data_dict['video_info'] = video_info
                new_all_data_list[n].append(data_dict)
            bucket_index += 1
    return new_all_data_list
