#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 PHYSICAL_GPU NUM_QUERIES [INFERENCE_TEMPERATURE]" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
NUM_QUERIES="$2"
INFERENCE_TEMPERATURE="${3:-0.03}"
ROOT="results/exp04_audio_q${NUM_QUERIES}"
EPOCH2_DIR="${ROOT}_train_epoch2"
EPOCH3_DIR="${ROOT}_train_epoch3_lr1e5"

cd /home/yxd/AudioRouter

# Match the selected q64 regimen exactly: two epochs at 2e-5, followed by one
# resumed epoch at 1e-5.  Each budget is trained independently from the same
# random seed used by the trainer; query banks are never resized at inference.
bash scripts/run_experience_train.sh \
  "${PHYSICAL_GPU}" 3 "${NUM_QUERIES}" "${EPOCH2_DIR}" 2 2e-5

bash scripts/run_experience_train.sh \
  "${PHYSICAL_GPU}" 3 "${NUM_QUERIES}" "${EPOCH3_DIR}" 3 1e-5 \
  "${EPOCH2_DIR}/adbt_epoch_2_video_720.pt"

CHECKPOINT="${EPOCH3_DIR}/adbt_epoch_3.pt"
sha256sum "${CHECKPOINT}" | tee "${ROOT}_checkpoint.sha256"

for SPLIT in short medium long; do
  bash scripts/run_experience_eval.sh \
    "${PHYSICAL_GPU}" audio "${CHECKPOINT}" \
    "${ROOT}_${SPLIT}" "videomme_${SPLIT}" \
    "${NUM_QUERIES}" "${INFERENCE_TEMPERATURE}" auto
done

source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate audiorouter
python scripts/summarize_videomme_results.py \
  --result short "${ROOT}_short" \
  --result medium "${ROOT}_medium" \
  --result long "${ROOT}_long" \
  --output "${ROOT}_summary.json"
