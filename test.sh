source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate AudioRouter
cd /home/yxd/AudioRouter

CUDA_VISIBLE_DEVICES=6,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
accelerate launch \
    --multi_gpu \
    --num_processes 2 \
    --num_machines 1 \
    --gpu_ids 6,7 \
    --mixed_precision no \
    --main_process_port 0 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto' \
    --tasks videomme_short,videomme_medium,videomme_long \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/videomme-baseline

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
WRAPPER=AudioRouter CTR_K=7 CTR_BETA=0.6 \
CTR_SIMILARITY_THRESHOLD=0.9 CTR_RETAIN_TOKENS=50 \
OQM_GROUP_SIZE=50 OQM_SLIDING_WINDOW_SIZE=4800 \
OQM_RETRIEVAL_MAX_TOKENS=12544 OQM_ENABLE_QUANTIZATION=1 \
OQM_QUANTIZATION_BITS=4 OQM_INIT_TOKEN_COUNT=14 \
STREAMING_ENCODER_BATCH_SIZE=32 AudioRouter_USE_FULL_PROMPT=0 \
accelerate launch --num_processes=8 -m lmms_eval \
    --model llava_onevision \
    --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto' \
    --tasks videomme_short,videomme_medium,videomme_long \
    --batch_size 1 --log_samples \
    --output_path ./results/videomme-AudioRouter


CUDA_VISIBLE_DEVICES=0,7 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
python -m lmms_eval \
    --model llava_onevision \
    --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=auto,attn_implementation=flash_attention_2,max_frames_num=16' \
    --tasks videomme_short,videomme_medium,videomme_long \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/videomme-baseline

CUDA_VISIBLE_DEVICES=3,5 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m lmms_eval \
    --model llava_onevision \
    --model_args 'pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,fps=auto,device_map=balanced_low_0,attn_implementation=sdpa,max_frames_num=16' \
    --tasks videomme_short \
    --batch_size 1 \
    --log_samples \
    --output_path ./results/videomme-baseline