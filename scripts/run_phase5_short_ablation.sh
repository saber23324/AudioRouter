#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 7 ]]; then
  echo "usage: $0 PHYSICAL_GPU AUDIO_MODE CHECKPOINT OUTPUT_DIR [ATTENTION_TEMPERATURE] [TASK] [BOTTLENECK_STAGE]" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
AUDIO_MODE="$2"
ADBT_CHECKPOINT="$3"
OUTPUT_DIR="$4"
ATTENTION_TEMPERATURE="${5:-0.05}"
TASK="${6:-videomme_short}"
BOTTLENECK_STAGE="${7:-3}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${CONDA_DEFAULT_ENV:-}" != "audiorouter" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
  elif [[ -f /home/yxd/miniconda3/etc/profile.d/conda.sh ]]; then
    CONDA_BASE=/home/yxd/miniconda3
  else
    echo "conda was not found; activate the audiorouter environment first" >&2
    exit 1
  fi
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate audiorouter
fi
cd "${REPO_ROOT}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/lmms-eval:${REPO_ROOT}/LLaVA-NeXT:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AudioRouter_MEASURE_MEMORY=1
export WRAPPER=AudioRouter
export VIDEOMME_SPLIT_FILE="${VIDEOMME_SPLIT_FILE:-./results/adbt_videomme_split.json}"
export VIDEOMME_EVAL_SPLIT=test
export AUDIO_BOTTLENECK_STAGE="${BOTTLENECK_STAGE}"
export AUDIO_BOTTLENECK_NUM_QUERIES=64
export AUDIO_BOTTLENECK_HIDDEN_SIZE=256
export AUDIO_BOTTLENECK_NUM_HEADS=8
export AUDIO_BOTTLENECK_ARCHITECTURE=phase4
export AUDIO_BOTTLENECK_TEMPERATURE="${ATTENTION_TEMPERATURE}"
if [[ $# -ge 5 ]]; then
  # A fifth argument is an explicit inference-only temperature sweep.  The
  # checkpoint guard remains enabled for all legacy four-argument calls.
  export AUDIO_BOTTLENECK_ALLOW_TEMPERATURE_OVERRIDE=1
fi
export AUDIO_BOTTLENECK_VALUE_MODE=native
export AUDIO_BOTTLENECK_BACKEND=direct
export AUDIO_BOTTLENECK_LATENT_NORM=none
export AUDIO_BOTTLENECK_CHECKPOINT="${ADBT_CHECKPOINT}"
export AUDIO_ABLATION="${AUDIO_MODE}"
export AUDIO_CROSSVIDEO_PATH="${AUDIO_CROSSVIDEO_PATH:-${HF_HOME}/videomme/data/-QuCz7kxBr8.mp4}"
export BEATS_CHECKPOINT="${BEATS_CHECKPOINT:-/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt}"
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

python -m lmms_eval \
  --model llava_onevision \
  --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2' \
  --tasks "${TASK}" \
  --batch_size 1 \
  --log_samples \
  --output_path "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}.log"
