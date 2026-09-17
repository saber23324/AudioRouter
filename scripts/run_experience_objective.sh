#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU {kd|av|kd_av|full} OUTPUT_DIR" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
VARIANT="$2"
OUTPUT_DIR="$3"
MAX_TRAIN_SAMPLES="${EXPERIENCE_MAX_TRAIN_SAMPLES:-0}"
WANDB_MODE="${EXPERIENCE_WANDB_MODE:-online}"

# Keep the coefficients of the validated Phase-5.1 run.  For the three
# literal objective-only rows requested by docs/experience.md, QA is disabled.
# ``full`` exactly reproduces the historical Phase-5.1 objective.
case "${VARIANT}" in
  kd)
    QA_WEIGHT=0
    ROUTE_WEIGHT=0.1
    FEATURE_WEIGHT=0
    AV_WEIGHT=0
    ;;
  av)
    QA_WEIGHT=0
    ROUTE_WEIGHT=0
    FEATURE_WEIGHT=0
    AV_WEIGHT=0.05
    ;;
  kd_av)
    QA_WEIGHT=0
    ROUTE_WEIGHT=0.1
    FEATURE_WEIGHT=0
    AV_WEIGHT=0.05
    ;;
  full)
    QA_WEIGHT=1
    ROUTE_WEIGHT=0.1
    FEATURE_WEIGHT=0.05
    AV_WEIGHT=0.05
    ;;
  *)
    echo "invalid variant: ${VARIANT}" >&2
    exit 2
    ;;
esac

source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate audiorouter
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

INIT_CHECKPOINT=results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt
RUN_NAME="$(basename "${OUTPUT_DIR}")"

echo "[EXPERIENCE OBJECTIVE] gpu=${PHYSICAL_GPU} variant=${VARIANT} qa=${QA_WEIGHT} route=${ROUTE_WEIGHT} feature=${FEATURE_WEIGHT} av=${AV_WEIGHT} max_train_samples=${MAX_TRAIN_SAMPLES} init=${INIT_CHECKPOINT} output=${OUTPUT_DIR} git=$(git rev-parse HEAD)"

python -m AudioRouter.train_adbt_videomme \
  --epochs 1 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir "${OUTPUT_DIR}" \
  --train-max-frames 32 \
  --max-train-samples "${MAX_TRAIN_SAMPLES}" \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase5_1 \
  --qa-loss-weight "${QA_WEIGHT}" \
  --route-loss-weight "${ROUTE_WEIGHT}" \
  --feature-loss-weight "${FEATURE_WEIGHT}" \
  --av-routing-loss-weight "${AV_WEIGHT}" \
  --teacher-importance-topk 64 \
  --av-routing-temperature 0.07 --av-min-temporal-offset 5 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint /home/yxd/.cache/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 \
  --eval-every-steps 100 --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter "${INIT_CHECKPOINT}" \
  --wandb --wandb-project AudioRouter-videomme --wandb-entity amd_yes \
  --wandb-run-name "${RUN_NAME}" --wandb-mode "${WANDB_MODE}" --wandb-log-every 1 \
  2>&1 | tee "${OUTPUT_DIR}.log"
