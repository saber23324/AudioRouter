#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 || $# -gt 8 ]]; then
  echo "usage: $0 PHYSICAL_GPU BOTTLENECK_STAGE NUM_QUERIES OUTPUT_DIR EPOCHS LEARNING_RATE [RESUME_CHECKPOINT] [WANDB_RUN_ID]" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
BOTTLENECK_STAGE="$2"
NUM_QUERIES="$3"
OUTPUT_DIR="$4"
EPOCHS="$5"
LEARNING_RATE="$6"
RESUME_CHECKPOINT="${7:-}"
WANDB_RUN_ID="${8:-}"
RUN_NAME="$(basename "${OUTPUT_DIR}")"

source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate AudioRouter
cd /home/yxd/AudioRouter

export PYTHONNOUSERSITE=1
export PYTHONWARNINGS="ignore:None of the inputs have requires_grad=True"
export PYTHONPATH=/home/yxd/AudioRouter
export HF_HOME=/home/yxd/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BEATS_EMBEDDING_CACHE=/home/yxd/AudioRouter/results/cache/beats

EXTRA_ARGS=()
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--resume "${RESUME_CHECKPOINT}" --override-resume-learning-rate)
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  EXTRA_ARGS+=(--wandb-run-id "${WANDB_RUN_ID}")
fi

echo "[EXPERIENCE TRAIN] gpu=${PHYSICAL_GPU} stage=${BOTTLENECK_STAGE} queries=${NUM_QUERIES} output=${OUTPUT_DIR} epochs=${EPOCHS} lr=${LEARNING_RATE} resume=${RESUME_CHECKPOINT:-none} git=$(git rev-parse HEAD)"

python -m AudioRouter.train_adbt_videomme \
  --epochs "${EPOCHS}" \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir "${OUTPUT_DIR}" \
  --train-max-frames 32 \
  --num-queries "${NUM_QUERIES}" \
  --bottleneck-hidden 256 \
  --num-heads 8 \
  --bottleneck-stage "${BOTTLENECK_STAGE}" \
  --architecture phase4 \
  --attention-temperature 0.05 \
  --value-mode native \
  --latent-norm none \
  --training-objective phase4 \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay 0.01 \
  --gradient-accumulation-steps 1 \
  --max-grad-norm 1 \
  --log-every 10 \
  --rank-log-every 30 \
  --module-log-every 30 \
  --save-every-videos 20 \
  --eval-every-steps 100 \
  --eval-max-videos 4 \
  --eval-max-frames 32 \
  --wandb \
  --wandb-project AudioRouter-videomme \
  --wandb-entity amd_yes \
  --wandb-run-name "${RUN_NAME}" \
  --wandb-mode online \
  --wandb-log-every 1 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${OUTPUT_DIR}.log"
