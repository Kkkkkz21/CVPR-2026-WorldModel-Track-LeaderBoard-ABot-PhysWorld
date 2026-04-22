#!/bin/bash
# ============================================================
# 单 GPU 串行推理脚本
#
# 功能：在单个 GPU 上串行推理 episode
#
# 使用方式：
#   1. 先手动启动 simulator（一个端口即可）
#   2. bash run_single_gpu.sh
#   指定单个 group：  EPISODE_GROUPS=1 bash run_single_gpu.sh
#   指定多个 group：  EPISODE_GROUPS=0,1,2,3 bash run_single_gpu.sh
#   不指定则跑全部 episode
# ============================================================

set -e

# ==================== 用户配置 ====================

# 指定要跑的 task（如 task1, task2, ..., task8）
TASK="task1"

# 使用哪个 GPU（单个 GPU 编号）
GPU_ID="1"

# Simulator 配置（只需一个端口）
SIMULATOR_PORT="6001"
SIMULATOR_IP="127.0.0.1"

# 使用哪个推理脚本：inference.py 或 inference_abot.py
INFERENCE_SCRIPT="scripts/inference.py"

# -------------------- 模型路径 --------------------
TRANSFORMER_MODEL_PATH="/path/to/ckpts/giga_challenge.safetensors"

# -------------------- 推理配置 --------------------
SEED=1024
OUTPUT_DIR="outputs"
TASK_CONFIG_FILE="task_configs_v1.json"

# -------------------- World Model 推理参数 --------------------
WM_FRAME_PER_TIME=40
NUM_INFERENCE_STEPS=50
CFG_SCALE=5.0
RENDER_SUBSAMPLE_STEP=1

# -------------------- Policy 模型路径 --------------------
POLICY_MODEL_BASE="/path/to/wmtrack/model/models--open-gigaai--CVPR-2026-WorldModel-Track-Model-"
POLICY_CKPT_DIR=""    # 留空则自动拼接为 ${POLICY_MODEL_BASE}Task{N}
POLICY_NORM_STATS=""  # 留空则自动拼接为 ${POLICY_CKPT_DIR}/norm_stat_gigabrain.json

# Episode groups（可选，逗号分隔，不设置则跑全部 episode）
# 例如：EPISODE_GROUPS=0,1,2  表示依次跑 group 0、1、2
EPISODE_GROUPS=${EPISODE_GROUPS:-""}
EPISODES_PER_GROUP=${EPISODES_PER_GROUP:-5}

# -------------------- Python 路径 --------------------
PYTHON="/path/to/miniconda3/envs/giga_torch/bin/python"

# ==================== 以下无需修改 ====================

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${REPO_DIR}/logs/single_gpu_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

# Policy 模型路径：留空则根据 TASK 自动拼接
RESOLVED_POLICY_CKPT="${POLICY_CKPT_DIR}"
if [ -z "${RESOLVED_POLICY_CKPT}" ]; then
    TASK_SUFFIX=$(echo "${TASK}" | sed 's/task/Task/')
    RESOLVED_POLICY_CKPT="${POLICY_MODEL_BASE}${TASK_SUFFIX}"
fi
RESOLVED_NORM_STATS="${POLICY_NORM_STATS}"
if [ -z "${RESOLVED_NORM_STATS}" ]; then
    RESOLVED_NORM_STATS="${RESOLVED_POLICY_CKPT}/norm_stat_gigabrain.json"
fi

INF_LOG="${LOG_DIR}/inference_all_episodes_gpu${GPU_ID}.log"

echo "============================================================"
echo "Single GPU Inference"
echo "============================================================"
echo "Task:            ${TASK}"
echo "GPU:             ${GPU_ID}"
echo "Simulator:       ${SIMULATOR_IP}:${SIMULATOR_PORT}"
echo "Inference:       ${INFERENCE_SCRIPT}"
echo "Task Config:     ${TASK_CONFIG_FILE}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Log Dir:         ${LOG_DIR}"
if [ -n "${EPISODE_GROUPS}" ]; then
echo "Episode groups:  ${EPISODE_GROUPS} (${EPISODES_PER_GROUP} episodes/group)"
else
echo "Episode groups:  (all episodes)"
fi
echo "============================================================"

# 构建公共推理参数
build_inf_cmd() {
    local CMD="CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON} ${REPO_DIR}/${INFERENCE_SCRIPT}"
    CMD+=" --transformer_model_path ${TRANSFORMER_MODEL_PATH}"
    CMD+=" --device_list 0"
    CMD+=" --seed ${SEED}"
    CMD+=" --mode online"
    CMD+=" --task ${TASK}"
    CMD+=" --output_dir ${OUTPUT_DIR}"
    CMD+=" --wm_frame_per_time ${WM_FRAME_PER_TIME}"
    CMD+=" --num_inference_steps ${NUM_INFERENCE_STEPS}"
    CMD+=" --cfg_scale ${CFG_SCALE}"
    CMD+=" --render_subsample_step ${RENDER_SUBSAMPLE_STEP}"
    CMD+=" --simulator_ip ${SIMULATOR_IP}"
    CMD+=" --simulator_port ${SIMULATOR_PORT}"
    CMD+=" --task_config_file ${TASK_CONFIG_FILE}"
    CMD+=" --policy_ckpt_dir ${RESOLVED_POLICY_CKPT}"
    CMD+=" --policy_norm_stats_path ${RESOLVED_NORM_STATS}"

    # VACE 参数（仅 inference_abot.py 使用）
    if [[ "${INFERENCE_SCRIPT}" == *"inference_abot"* ]]; then
        CMD+=" --vace_checkpoint ${VACE_CHECKPOINT}"
        CMD+=" --vace_in_dim ${VACE_IN_DIM}"
        CMD+=" --vace_layers_step ${VACE_LAYERS_STEP}"
    fi
    echo "${CMD}"
}

run_inference() {
    local INF_CMD="$1"
    local LOG_FILE="$2"
    local LABEL="$3"

    echo ""
    echo "[Running] ${LABEL}"
    echo "  Command: ${INF_CMD}"
    echo ""

    eval ${INF_CMD} 2>&1 | tee "${LOG_FILE}"
    local EXIT_CODE=${PIPESTATUS[0]}

    if [ ${EXIT_CODE} -ne 0 ]; then
        echo "ERROR: ${LABEL} failed with exit code ${EXIT_CODE}"
        echo "Check log: ${LOG_FILE}"
        exit ${EXIT_CODE}
    fi
    echo ">>> ${LABEL} done."
}

if [ -z "${EPISODE_GROUPS}" ]; then
    # 未指定 group，跑全部 episode
    INF_CMD=$(build_inf_cmd)
    run_inference "${INF_CMD}" "${LOG_DIR}/inference_all_gpu${GPU_ID}.log" "All episodes on GPU ${GPU_ID}"
else
    # 指定了 group 列表，逐个跑
    IFS=',' read -ra GROUP_LIST <<< "${EPISODE_GROUPS}"
    for group_idx in "${GROUP_LIST[@]}"; do
        INF_CMD=$(build_inf_cmd)
        INF_CMD+=" --episode_group ${group_idx}"
        INF_CMD+=" --episodes_per_group ${EPISODES_PER_GROUP}"
        run_inference "${INF_CMD}" "${LOG_DIR}/inference_group${group_idx}_gpu${GPU_ID}.log" "Episode group ${group_idx} on GPU ${GPU_ID}"
    done
fi

echo ""
echo "============================================================"
echo "All inference tasks completed."
echo "Output: ${OUTPUT_DIR}/evaluator_test/${TASK}/"
echo "============================================================"
