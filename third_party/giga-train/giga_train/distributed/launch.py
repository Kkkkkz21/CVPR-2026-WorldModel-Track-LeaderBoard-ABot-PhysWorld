import os
import socket
import subprocess
import time
from typing import Any
import sys
import torch
from accelerate import DistributedType
from accelerate.commands.config.config_args import ClusterConfig
from accelerate.utils import ComputeEnvironment

from ..configs import load_config
from ..utils import wait_for_gpu_memory
from .run_task import run_tasks


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


class Launcher:
    """Thin wrapper around Accelerate cluster config and launch command.

    Supports single-node and multi-node launches, optionally with DeepSpeed/FSDP.
    """

    def __init__(
        self,
        gpu_ids: list[int] | list[list[int]] | None = None, # 允许为 None
        num_machines: int | None = None,
        distributed_type: DistributedType | str | None = None,
        main_process_ip: str | list[str] = '127.0.0.1',
        main_process_port: int | None = None,
        num_cpu_threads_per_process: int = 2,
        nccl_socket_ifname: str | None = None,
        save_config_path: str | None = None,
        save_hostfile_path: str | None = None,
        env: dict[str, str] | None = None,
        executable: str | None = None,
        until_completion: bool = False,
        **kwargs: Any,
    ) -> None:
        
        # --- [核心逻辑修复] 处理 gpu_ids 为 None 的情况 ---
        if gpu_ids is None:
            visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if visible_devices:
                try:
                    # 将 "0,1,2,3" 或 "0, 1, 2, 3" 解析为 [0, 1, 2, 3]
                    # 注意：这里解析出来的是逻辑 ID (0 到 N-1)，这正是 accelerate 需要的
                    gpu_ids = [int(x.strip()) for x in visible_devices.split(",") if x.strip()]
                    print(f"[Launcher] Auto-detected GPU IDs from CUDA_VISIBLE_DEVICES='{visible_devices}': {gpu_ids}")
                except ValueError:
                    pass
            
            if gpu_ids is None and torch.cuda.is_available():
                count = torch.cuda.device_count()
                if count > 0:
                    gpu_ids = list(range(count))
                    print(f"[Launcher] Auto-detected {count} physical GPUs, using IDs: {gpu_ids}")
            
            if gpu_ids is None:
                # 极端兜底
                gpu_ids = [0]
                print("[Launcher] Warning: No GPUs detected, defaulting to [0]")
        
        # 此时 gpu_ids 保证是一个有效的列表
        # --------------------------------------------------

        if num_machines is None:
            if isinstance(main_process_ip, str):
                num_machines = 1
            elif isinstance(main_process_ip, list):
                num_machines = len(main_process_ip)
            else:
                assert False, "main_process_ip must be str or list[str]"

        if main_process_port is None:
            main_process_port = _find_free_port()
            
        cur_time = time.strftime('%Y-%m-%d-%H%M%S', time.localtime(time.time()))
        os.makedirs('_tmp', exist_ok=True)
        if save_config_path is None:
            save_config_path = f'_tmp/{cur_time}_config.json'
        if save_hostfile_path is None:
            save_hostfile_path = f'_tmp/{cur_time}_hostfile'
        if env is None:
            env = os.environ.copy() # 使用 copy 避免污染全局
        
        # ... (保留 executable 查找逻辑不变) ...
        if executable is None:
            process = subprocess.run(['which', 'accelerate'], env=env, capture_output=True, text=True)
            if process.returncode == 0 and process.stdout.strip():
                executable = process.stdout.strip()
            else:
                possible_paths = [
                    "/root/.local/bin/accelerate",
                    "/opt/conda/envs/python3.10.13/bin/accelerate",
                    "/opt/conda/bin/accelerate",
                ]
                found = False
                for path in possible_paths:
                    if os.path.exists(path):
                        executable = path
                        found = True
                        break
                if not found:
                    conda_bin = os.path.dirname(sys.executable)
                    candidate = os.path.join(conda_bin, "accelerate")
                    if os.path.exists(candidate):
                        executable = candidate
                    else:
                        raise ValueError(f"accelerate not found")

        # --- 单节点逻辑 ---
        if num_machines == 1:
            if distributed_type is None:
                distributed_type = DistributedType.MULTI_GPU
            
            distributed_type = DistributedType(distributed_type)
            
            deepspeed_config = None
            fsdp_config = None

            if distributed_type == DistributedType.DEEPSPEED:
                deepspeed_config = kwargs.pop('deepspeed_config', {})
                deepspeed_config_file = deepspeed_config.get('deepspeed_config_file', None)
                if deepspeed_config_file is not None:
                    if not os.path.exists(deepspeed_config_file):
                        cur_dir = os.path.dirname(os.path.abspath(__file__))
                        deepspeed_config_file = os.path.join(cur_dir, deepspeed_config_file)
                    assert os.path.exists(deepspeed_config_file), f"Deepspeed config not found: {deepspeed_config_file}"
                    deepspeed_config['deepspeed_config_file'] = deepspeed_config_file
            
            elif distributed_type == DistributedType.FSDP:
                fsdp_config = kwargs.pop('fsdp_config', {})

            num_processes = len(gpu_ids)
            # 将列表转换为逗号分隔的字符串，例如 "0,1,2,3"
            # 这里的 gpu_ids 已经是逻辑 ID (0..N-1)，完全正确
            gpu_ids_str = ','.join([str(i) for i in gpu_ids])
            
            cluster_config = ClusterConfig(
                compute_environment=ComputeEnvironment.LOCAL_MACHINE,
                distributed_type=distributed_type,
                mixed_precision=None,
                use_cpu=False,
                debug=False,
                num_processes=num_processes,
                gpu_ids=gpu_ids_str, 
                main_process_ip=main_process_ip,
                main_process_port=main_process_port,
                deepspeed_config=deepspeed_config,
                fsdp_config=fsdp_config,
                **kwargs,
            )
        
        # --- 多节点逻辑 (保持原样，但确保 gpu_ids 处理一致) ---
        else:
            assert nccl_socket_ifname is not None, "nccl_socket_ifname is required for multi-node"
            env['NCCL_SOCKET_IFNAME'] = nccl_socket_ifname
            env['NCCL_NET'] = 'IB'
            env['NCCL_IB_DISABLE'] = '0'
            env['NCCL_IB_PCI_RELAXED_ORDERING'] = '1'
            
            if distributed_type is None:
                distributed_type = DistributedType.DEEPSPEED
            distributed_type = DistributedType(distributed_type)
            assert distributed_type == DistributedType.DEEPSPEED

            # 处理多机下的 gpu_ids 格式
            if isinstance(gpu_ids[0], list):
                assert len(gpu_ids) == num_machines
            else:
                # 假设每台机器卡的分布是一样的
                gpu_ids = [gpu_ids for _ in range(num_machines)]
            
            num_processes = sum(len(_) for _ in gpu_ids)
            assert isinstance(main_process_ip, list) and len(main_process_ip) == num_machines
            
            # 生成 hostfile
            with open(save_hostfile_path, 'w') as fn:
                for ip in main_process_ip:
                    # 这里假设每台机器卡数相同，取第一个长度作为 slots
                    fn.write(f'{ip} slots={len(gpu_ids[0])}\n')
            
            includes = []
            for ip, gpu_ids_i in zip(main_process_ip, gpu_ids):
                gpu_ids_i_str = ','.join([str(i) for i in gpu_ids_i])
                includes.append(f'{ip}:{gpu_ids_i_str}')
            includes = '@'.join(includes)
            
            deepspeed_config = kwargs.pop('deepspeed_config', {})
            deepspeed_config.update(
                dict(
                    deepspeed_multinode_launcher='pdsh',
                    deepspeed_hostfile=save_hostfile_path,
                    deepspeed_inclusion_filter=includes,
                )
            )
            cluster_config = ClusterConfig(
                compute_environment=ComputeEnvironment.LOCAL_MACHINE,
                distributed_type=distributed_type,
                mixed_precision=None,
                use_cpu=False,
                debug=False,
                num_processes=num_processes,
                num_machines=num_machines,
                main_process_port=main_process_port,
                deepspeed_config=deepspeed_config,
                **kwargs,
            )

        cluster_config.to_json_file(save_config_path)
        self.cluster_config = cluster_config
        self.config_file = save_config_path
        self.hostfile_path = save_hostfile_path
        self.num_cpu_threads_per_process = num_cpu_threads_per_process
        self.env = env
        self.executable = executable
        self.until_completion = until_completion

    def launch(self, script: str) -> None:
        command = f'{self.executable} launch'
        command += f' --config_file {self.config_file}'
        command += f' --num_cpu_threads_per_process {self.num_cpu_threads_per_process}'
        command += f' {script}'
        print(f"[Launch Command] {command}", flush=True)
        
        command_list = command.split(' ')
        try:
            while True:
                process = subprocess.run(command_list, env=self.env)
                if process.returncode != 0 and self.until_completion:
                    print("[Launcher] Process failed, retrying in 10s...", flush=True)
                    time.sleep(10)
                else:
                    break
        finally:
            if os.path.exists(self.config_file):
                os.remove(self.config_file)
            if os.path.exists(self.hostfile_path):
                os.remove(self.hostfile_path)

def launch_from_config(config_path: str, runners: list[str] | str | None = None, gpu_memory: float | None = None, seconds: int = 10) -> None:
    from ..configs import load_config
    from .run_task import run_tasks
    
    config = load_config(config_path)
    
    # --- 重新统一获取逻辑 GPU 列表的逻辑 ---
    # 无论单卡还是多卡，我们都应该基于 CUDA_VISIBLE_DEVICES 来获取“逻辑索引”
    visible_devices_str = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    detected_logical_gpus = []
    
    if visible_devices_str:
        try:
            # 分割并转为整数，这些就是逻辑索引 (0, 1, 2...)
            detected_logical_gpus = [int(x.strip()) for x in visible_devices_str.split(",") if x.strip()]
        except ValueError:
            pass
            
    if not detected_logical_gpus and torch.cuda.is_available():
        # 如果没有环境变量，回退到物理探测 (此时逻辑 ID = 物理 ID)
        count = torch.cuda.device_count()
        detected_logical_gpus = list(range(count))
        
    if not detected_logical_gpus:
        detected_logical_gpus = [0] # 兜底

    num_machines = config.launch.get('num_machines', 1)
    
    # 【关键修复】使用检测到的逻辑 GPU 列表进行判断
    # 注意：len(detected_logical_gpus) 才是当前进程真正能用的卡数
    current_visible_count = len(detected_logical_gpus)

    # 如果是单节点 且 当前环境只可见 1 张卡 -> 走单卡模式
    if num_machines == 1 and current_visible_count == 1:
        # 【修复点】强制设置为逻辑卡 0，绝对安全
        # 不需要关心物理卡号是多少，在当前的进程视图里，它永远是 0
        torch.cuda.set_device(0) 
        print(f"[Single GPU Mode] Running on logical GPU 0 (Visible devices: {visible_devices_str})")
        run_tasks(config, runners)
        
    else:
        # 多卡或多机模式，启动 Launcher
        # 此时将 config.launch.gpu_ids 强制设为 None，让 Launcher 内部去读取环境变量
        # 这样可以避免配置文件里残留的物理 ID 干扰
        launch_args = dict(config.launch)
        launch_args['gpu_ids'] = None 
        
        print(f"[Multi-GPU Mode] Starting launcher. Visible GPUs: {detected_logical_gpus}")
        launcher = Launcher(**launch_args)
        
        # ... (后续启动逻辑保持不变) ...
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, 'run_task.py')
        if not os.path.exists(file_path):
             file_path = os.path.join(os.path.abspath(__file__).split('launch')[0], 'run_task.py')

        if runners is not None:
            if isinstance(runners, (list, tuple)):
                runners = ','.join(runners)
            launcher.launch('{} --config {} --runners {}'.format(file_path, config_path, runners))
        else:
            launcher.launch('{} --config {}'.format(file_path, config_path))
