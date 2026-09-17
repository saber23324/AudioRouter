#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 8 ]]; then
  echo "usage: $0 PHYSICAL_GPU SETTING CHECKPOINT OUTPUT_DIR TASK [NUM_QUERIES] [ATTENTION_TEMPERATURE] [FPS_SETTING]" >&2
  echo "SETTING: full | vision | audio" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
SETTING="$2"
CHECKPOINT="$3"
OUTPUT_DIR="$4"
TASK="$5"
NUM_QUERIES="${6:-64}"
ATTENTION_TEMPERATURE="${7:-0.03}"
FPS_SETTING="${8:-auto}"

case "${SETTING}" in
  full)
    BOTTLENECK_STAGE=0
    ;;
  vision)
    BOTTLENECK_STAGE=1
    ;;
  audio)
    BOTTLENECK_STAGE=3
    ;;
  *)
    echo "invalid SETTING=${SETTING}; expected full, vision, or audio" >&2
    exit 2
    ;;
esac

if [[ "${BOTTLENECK_STAGE}" != "0" && ! -f "${CHECKPOINT}" ]]; then
  echo "checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 2
fi

source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate audiorouter
cd /home/yxd/AudioRouter

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/yxd/AudioRouter
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export HF_HOME=/home/yxd/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BEATS_EMBEDDING_CACHE=/home/yxd/AudioRouter/results/cache/beats
export AudioRouter_MEASURE_MEMORY=1
export AudioRouter_PROFILE="${AudioRouter_PROFILE:-0}"
export AudioRouter_LOG_SAMPLING=1
export AudioRouter_PREFIX_KV_CACHE="${AudioRouter_PREFIX_KV_CACHE:-1}"
export AudioRouter_PREFIX_PREFILL_CHUNK_SIZE="${AudioRouter_PREFIX_PREFILL_CHUNK_SIZE:-32768}"
export AudioRouter_VIDEO_TENSOR_CACHE="${AudioRouter_VIDEO_TENSOR_CACHE:-1}"
export WRAPPER=AudioRouter
export VIDEOMME_SPLIT_FILE=./results/adbt_videomme_split.json
export VIDEOMME_EVAL_SPLIT=test
export AUDIO_BOTTLENECK_STAGE="${BOTTLENECK_STAGE}"
export AUDIO_BOTTLENECK_NUM_QUERIES="${NUM_QUERIES}"
export AUDIO_BOTTLENECK_HIDDEN_SIZE=256
export AUDIO_BOTTLENECK_NUM_HEADS=8
export AUDIO_BOTTLENECK_ARCHITECTURE=phase4
export AUDIO_BOTTLENECK_TEMPERATURE="${ATTENTION_TEMPERATURE}"
export AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE=1
export AUDIO_BOTTLENECK_VALUE_MODE=native
export AUDIO_BOTTLENECK_BACKEND=direct
export AUDIO_BOTTLENECK_LATENT_NORM=none
export AUDIO_BOTTLENECK_CHECKPOINT="${CHECKPOINT}"
export AUDIO_ABLATION=real
export BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
export BEATS_BATCH_SIZE=8
export LLAVA_VISION_ENCODER_BATCH_SIZE=4
export STREAMING_ENCODER_BATCH_SIZE=4
export CTR_K=7
export CTR_BETA=0.6
export CTR_SIMILARITY_THRESHOLD=0.9
export CTR_RETAIN_TOKENS=64
export OQM_GROUP_SIZE=64
export OQM_SLIDING_WINDOW_SIZE=4800
export OQM_RETRIEVAL_MAX_TOKENS=12544
export OQM_ENABLE_QUANTIZATION=0
export OQM_QUANTIZATION_BITS=4
export OQM_INIT_TOKEN_COUNT=14

echo "[EXPERIENCE CONFIG] gpu=${PHYSICAL_GPU} setting=${SETTING} checkpoint=${CHECKPOINT} output=${OUTPUT_DIR} task=${TASK} queries=${NUM_QUERIES} temperature=${ATTENTION_TEMPERATURE} fps=${FPS_SETTING} prefix_chunk=${AudioRouter_PREFIX_PREFILL_CHUNK_SIZE} git=$(git rev-parse HEAD)"

EXTRA_ARGS=()
if [[ -n "${EXPERIENCE_LIMIT:-}" ]]; then
  EXTRA_ARGS+=(--limit "${EXPERIENCE_LIMIT}")
  echo "[EXPERIENCE SMOKE] limit=${EXPERIENCE_LIMIT}; never report this as a benchmark result"
fi

python -m lmms_eval \
  --model llava_onevision \
  --model_args "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=${FPS_SETTING},device_map=auto,attn_implementation=flash_attention_2" \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}.log"
