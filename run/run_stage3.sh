PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=4,5 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m AudioRouter.train_adbt_videomme \
  --epochs 5 \
  --split-file ./results/adbt_videomme_split.json \
  --output-dir ./results/adbt-checkpoints-shifted \
  --train-max-frames 32 \
  --vision-batch-size 16 \
  --beats-batch-size 8 \
  --beats-checkpoint \
  /nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt \
  --log-every 10 \
  --save-every-videos 100 \
  --wandb \
  --wandb-project AudioRouter-videomme \
  --wandb-run-name adbt-stage3-full \
  --eval-every-steps 100 \
  --eval-max-videos 4 \
  --eval-max-frames 32