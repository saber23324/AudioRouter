# ADBT 最新权重、训练、验证与可视化指南

更新时间：2026-09-13（Asia/Shanghai）

本文是当前项目的统一入口，覆盖：

- VideoMME；
- MLVU Ego 和完整七任务 MCQA；
- StreamingBench Real-Time Visual Understanding（本地已下载子集）；
- 三个数据集的训练、正式验证和 Where-to-Look heatmap 命令。

## 1. 当前最佳权重

| 数据集 | 当前最佳 checkpoint  | 验证配置 | 当前结果 |
|---|---|---|---:|
| VideoMME | `results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt` |  real audio，推理 tau=0.03 | 60.1852%（325/540） |
| MLVU Ego | `results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt` | real audio，推理 tau=0.03 | 69.3548%（43/62） |
| MLVU Full MCQA | `results/mlvu_full_mcqa_q64_tau005_from_ego_epoch3_lr1e5/adbt_epoch_2.pt` |  real audio，推理 tau=0.015 | 67.8322%（291/429） |
| StreamingBench Real-Time | `/home/yxd/AudioRouter/results/streamingbench_seedall_q64_from_912_video300_hardmargin05_tau055_lr2e6_r3_913/adbt_epoch_3.pt` |  real audio，推理 tau=0.05 | 72.40%（362/500） |

/home/yxd/AudioRouter/results/streamingbench_seedall_q64_from_912_video300_hardmargin05_tau055_lr2e6_r3_913

### VideoMME 分长度结果

| Split | 正确/总数 | Accuracy |
|---|---:|---:|
| Short | 144/210 | 68.5714% |
| Medium | 84/147 | 57.1429% |
| Long | 97/183 | 53.0055% |
| **Overall** | **325/540** | **60.1852%** |

对应结果：

- `results/adbt-phase4.1-epoch3-tune-infer-tau003-short/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_184833_results.json`
- `results/adbt-phase4.1-epoch3-tune-infer-tau003-medium/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_185452_results.json`
- `results/adbt-phase4.1-epoch3-tune-infer-tau003-long/lmms-lab__llava-onevision-qwen2-7b-ov/20260909_191701_results.json`


### MLVU Ego 结果

```text
43 / 62 = 69.3548%
mean QA loss = 0.869074
sampled frames = 4,307
peak allocated = cuda:0 12,093.6 MiB / cuda:1 16,711.8 MiB
peak reserved = cuda:0 12,744 MiB / cuda:1 17,832 MiB
```

结果文件：
`results/mlvu_ego_adbt_q64_tau003_epoch3_eval.json`。

### MLVU Full MCQA 结果

2026-09-13 从上述 Ego checkpoint 初始化，在七个 MLVU-dev 选择题任务的无媒体泄漏
train split 上训练到 3 epoch。epoch 2 是最佳点，正式 validation 得到：

```text
291 / 429 = 67.8322%
mean QA loss = 0.882342
sampled frames = 71,509
peak allocated = cuda:0 19,147.2 MiB / cuda:1 25,347.9 MiB
peak reserved = cuda:0 20,384 MiB / cuda:1 26,428 MiB
```

结果文件：
`results/mlvu_full_mcqa_epoch2_tau0015_full_eval.json`。
`sub_scene` 和 `summary` 是自由生成任务，不进入 A--D accuracy。划分、分任务结果、
命令和音频异常说明见 `docs/MLVU_FULL_MCQA_20260912.md`；epoch/temperature sweep、
哈希和新最佳结果见 `docs/MLVU_FULL_MCQA_TUNING_20260913.md`。epoch 3 在相同
tau=0.015 下回落到 276/429，故不继续盲目训练到 epoch 10。

若允许按已知任务类型设置推理温度，同一个 epoch 2 checkpoint 对 Needle 使用
tau=0.01、其余任务使用 tau=0.015，可得到 tuned/in-sample 292/429（68.0653%）；
相对 epoch 1 六项提升、Needle 持平。该结果是在同一 validation 上选温度，不替换
上表的固定温度主结果。组合记录与审计见上述 20260913 文档。

### StreamingBench Real-Time 结果

```text
362 / 500 = 72.40%
sampled frames = 60,328
peak allocated = cuda:0 11,099.8 MiB / cuda:1 19,437.8 MiB
peak reserved = cuda:0 11,704 MiB / cuda:1 20,928 MiB
```
结果文件：
`/home/yxd/AudioRouter/results/streamingbench_seedall_q64_from_912_video300_hardmargin05_tau055_lr2e6_r3_913`


这里的 `cuda:0/cuda:1` 是设置 `CUDA_VISIBLE_DEVICES=6,7` 后的逻辑编号，分别映射
到物理 GPU 6/7。VideoMME 的这组三个历史 lmms-eval JSON 没有保存 peak-memory
字段；重新验证时脚本设置的 `AudioRouter_MEASURE_MEMORY=1` 会把显存统计写入日志，
不要为补出一个历史数字而推测或复用其他 run 的显存。


因此两者虽文件名不同，400/100 个 train/test 视频集合完全一致，不存在因这次
改名产生的数据泄漏。新实验统一推荐使用含义清楚的 seed1234 文件名。

## 2. 模型和训练配置

三个最佳权重使用相同的主体配置：

```text
LLaVA backbone: lmms-lab/llava-onevision-qwen2-7b-ov
BEATs: frozen temporal audio encoder
architecture: phase4 / Phase-4.1 ADBT
latent slots: 64
bottleneck hidden: 256
attention heads: 8
value mode: native
latent norm: none
training objective: pure causal QA loss (phase4)
training temperature: 0.05
training frame cap: 32
```

冻结参数：

- LLaVA language model；
- SigLIP vision tower；
- LLaVA visual projector；
- BEATs audio encoder。

训练参数：

- Audio Query Generator；
- ADBT query/key、slot-specific audio/time conditioning 和 query self-attention；
- 其他属于 `AudioConditionedBottleneck` 的可训练参数。

不变的核心约束：

```text
P = softmax(Q_audio K_visual^T / tau)
Z = P V_native
```

Audio 只控制视觉压缩权重，不直接进入 LLM；被加权的 Value 全部来自视觉 token。
当前最佳配置不启用 CTR/OQM 双重压缩，不启用 visual value encoder，不加入 ASR，
也不更新 LLaVA 权重。

## 3. 环境和数据

所有命令先执行：

```bash
source /home/yxd/miniconda3/etc/profile.d/conda.sh
conda activate audiorouter
cd /home/yxd/AudioRouter

export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/yxd/AudioRouter
export HF_HOME=/home/yxd/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export BEATS_EMBEDDING_CACHE=/home/yxd/AudioRouter/results/cache/beats
export BEATS_CHECKPOINT=/nvme_data/pkt/huggingface/modules/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt
export CUDA_VISIBLE_DEVICES=6,7
```

不要通过减少正式评测 FPS 或设置 frame cap 隐藏 OOM。训练可使用
`--train-max-frames 32`；正式验证必须使用 `--fps auto --max-frames 0`。

### 数据路径与划分

| 数据集 | 媒体/缓存 | Manifest | Split |
|---|---|---|---|
| VideoMME | `/home/yxd/.cache/huggingface/videomme/data` | Hugging Face 本地缓存 | `results/adbt_videomme_split.json` |
| MLVU Ego | `/home/yxd/pkt/huggingface/hub/datasets--MLVU--MVLU/snapshots/06ddc388aa34746b3abba77972b7a7dfd977f7a3` | `results/dataset_manifests/mlvu_ego.json` | `results/dataset_splits/mlvu_ego_seed1234_80_20.json` |
| MLVU Full MCQA | 同上 | `results/dataset_manifests/mlvu_full_mcqa.json` | `results/dataset_splits/mlvu_full_mcqa_seed1234_pathgroup_80_20.json` |
| StreamingBench Real-Time | `/home/yxd/pkt/streaming/dataset` | `results/dataset_manifests/streamingbench_realtime.json` | `results/dataset_splits/streamingbench_realtime_seed1234_80_20.json` |

划分规模：

| 数据集 | Train | Validation |
|---|---:|---:|
| VideoMME | 720 videos / 2,160 QA | 180 videos / 540 QA |
| MLVU Ego | 67 videos / 290 QA | 17 videos / 62 QA |
| MLVU Full MCQA | 992 task-video IDs / 1,745 QA | 250 task-video IDs / 429 QA |
| StreamingBench Real-Time | 400 videos / 1,999 valid QA | 100 videos / 500 QA |

StreamingBench 的 `sample_242_1` 因标注时间超过视频长度被 manifest 排除。
`sample_332` 的 H.264 源文件损坏且位于训练集，训练时必须显式增加：

```text
--skip-video sample_332
```

这使每个实际训练 epoch 使用 399 个可解码视频、1,994 道问题。验证集不包含
`sample_332`。

若需要重新生成两个本地数据集的 manifest/split：

```bash
python scripts/prepare_partial_mcqa_datasets.py --dataset all
```

## 4. VideoMME 训练命令

当前最佳 VideoMME 模型使用两段学习率：epoch 1--2 为 `2e-5`，epoch 3 为
`1e-5`。请使用新的输出目录，不要覆盖当前最佳权重。

### Epoch 1--2

```bash
export VMME_RUN12=results/repro-videomme-q64-tau005-epoch12

python -m AudioRouter.train_adbt_videomme \
  --epochs 2 \
  --split-file results/adbt_videomme_split.json \
  --output-dir "$VMME_RUN12" \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 2e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 30 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --wandb --wandb-project AudioRouter-videomme --wandb-entity amd_yes \
  --wandb-run-name repro-videomme-q64-tau005-epoch12 \
  --wandb-mode online --wandb-log-every 1
```

### Epoch 3，学习率降到 1e-5

```bash
export VMME_RUN3=results/repro-videomme-q64-tau005-epoch3-lr1e5

python -m AudioRouter.train_adbt_videomme \
  --epochs 3 \
  --split-file results/adbt_videomme_split.json \
  --output-dir "$VMME_RUN3" \
  --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 \
  --vision-batch-size 16 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 30 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --resume "$VMME_RUN12/adbt_epoch_2_video_720.pt" \
  --override-resume-learning-rate \
  --wandb --wandb-project AudioRouter-videomme --wandb-entity amd_yes \
  --wandb-run-name repro-videomme-q64-tau005-epoch3-lr1e5 \
  --wandb-mode online --wandb-log-every 1
```

历史最佳训练的 W&B run：epoch 1--2 为 `epc9j4mg`，epoch 3 为 `jvs2k6lx`。

## 5. MLVU Ego 训练命令

MLVU 从当前最佳 VideoMME adapter 初始化，再在 MLVU train split 上训练 3 个
epoch。使用新目录可避免覆盖现有最佳 checkpoint：

```bash
export MLVU_RUN=results/repro-mlvu-ego-q64-tau005-epoch3-lr1e5

python -m AudioRouter.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/mlvu_ego.json \
  --split-file results/dataset_splits/mlvu_ego_seed1234_80_20.json \
  --output-dir "$MLVU_RUN" \
  --epochs 3 --train-max-frames 32 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 20 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project AudioRouter-mcqa --wandb-entity amd_yes \
  --wandb-run-name repro-mlvu-ego-q64-tau005-epoch3-lr1e5 \
  --wandb-mode online --wandb-log-every 1
```

历史最佳 W&B run：`d9oia0vq`。

## 6. StreamingBench Real-Time 训练命令

StreamingBench 同样从 VideoMME 最佳 adapter 初始化。实时问题按各自
`query_time_seconds` 构造因果采样网格；同一视频的网格先取 union 编码，再为每道
问题索引回自己的严格 causal prefix。

```bash
export SB_RUN=results/repro-streamingbench-q64-tau005-epoch3-lr1e5

python -m AudioRouter.train_adbt_videomme \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --output-dir "$SB_RUN" \
  --epochs 3 --train-max-frames 32 --skip-video sample_332 \
  --num-queries 64 --bottleneck-hidden 256 --num-heads 8 \
  --bottleneck-stage 3 --architecture phase4 \
  --attention-temperature 0.05 --value-mode native --latent-norm none \
  --training-objective phase4 --vision-batch-size 8 --beats-batch-size 8 \
  --beats-checkpoint "$BEATS_CHECKPOINT" \
  --learning-rate 1e-5 --weight-decay 0.01 \
  --gradient-accumulation-steps 1 --max-grad-norm 1 \
  --log-every 10 --rank-log-every 30 --module-log-every 30 \
  --save-every-videos 100 --eval-every-steps 100 \
  --eval-max-videos 4 --eval-max-frames 32 \
  --init-adapter results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --wandb --wandb-project AudioRouter-mcqa --wandb-entity amd_yes \
  --wandb-run-name repro-streamingbench-q64-tau005-epoch3-lr1e5 \
  --wandb-mode online --wandb-log-every 1
```

当前最佳 checkpoint 来自成功完成的 W&B run `u1genxhe`，最终
`global_step=5982`。不要让两个训练进程写入同一个 `--output-dir`；不同温度或 seed
实验必须使用不同目录，否则 checkpoint 和中间状态会互相覆盖。

## 7. 正式验证命令

### 7.1 VideoMME

VideoMME 使用 lmms-eval、无字幕 task、官方 aggregation、batch size 1 和 greedy
decoding。`fps=auto` 表示不超过 30 分钟使用 0.5 FPS，超过 30 分钟使用 0.2 FPS。

```bash
bash scripts/run_phase5_short_ablation.sh \
  6 real \
  results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  results/verify-videomme-short-tau003 \
  0.03 videomme_short

bash scripts/run_phase5_short_ablation.sh \
  6 real \
  results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  results/verify-videomme-medium-tau003 \
  0.03 videomme_medium

bash scripts/run_phase5_short_ablation.sh \
  7 real \
  results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  results/verify-videomme-long-tau003 \
  0.03 videomme_long
```

不要给正式命令增加 `--limit`。Medium/Long 可以分别放到物理 GPU 6/7 并行；不要
在未检查实时显存前同时启动第三个完整进程。

### 7.2 MLVU Ego

```bash
python -m AudioRouter.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/mlvu_ego.json \
  --split-file results/dataset_splits/mlvu_ego_seed1234_80_20.json \
  --checkpoint results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output results/verify-mlvu-ego-tau003.json \
  --fps auto --max-frames 0 --audio-ablation real \
  --attention-temperature 0.03 \
  --vision-batch-size 8 --beats-batch-size 8
```

### 7.3 StreamingBench Real-Time

当前 checkpoint 的已验证最佳结果使用 tau=0.05。命令中显式写温度，不依赖
checkpoint 默认值：

```bash
python -m AudioRouter.eval_adbt_mcqa \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --checkpoint results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output results/verify-streamingbench-realtime-tau005.json \
  --fps auto --max-frames 0 --audio-ablation real \
  --attention-temperature 0.05 \
  --vision-batch-size 8 --beats-batch-size 8
```

MLVU/StreamingBench 评测输出同时包含：

- `overall`；
- `by_task_type`；
- 每题 prediction、target、loss 和 causal prefix frame count；
- 总采样帧数；
- 每张可见 GPU 的 peak allocated/reserved memory。

## 8. 如何确认验证有效

正式验证完成后至少检查以下内容：

```bash
python - <<'PY'
import json

for path, expected in [
    ("results/verify-mlvu-ego-tau003.json", 62),
    ("results/verify-streamingbench-realtime-tau005.json", 500),
]:
    data = json.load(open(path))
    assert data["overall"]["examples"] == expected
    assert len(data["records"]) == expected
    assert len({(r["videoID"], r["question_id"]) for r in data["records"]}) == expected
    assert all(r["greedy_correct"] == r["option_correct"] for r in data["records"])
    assert data["fps"] == "auto" and data["max_frames"] == 0
    assert data["audio_ablation"] == "real"
    print(path, data["overall"], data["sampled_frames"])
PY
```

预期覆盖：

```text
MLVU: 62 questions, 4,307 sampled frames
StreamingBench: 500 questions, 60,328 sampled frames
VideoMME: Short 210, Medium 147, Long 183
```

VideoMME 需要读取每个 lmms-eval 输出中的
`videomme_perception_score,none`，然后按问题数 210/147/183 加权计算 Overall。
不要把三个 split 的简单平均当 Overall。

还要确认：

1. checkpoint SHA256 与本文一致；
2. 没有 `--limit`；
3. 正式验证没有训练用的 32-frame cap；
4. StreamingBench 每条记录的 `sampled_prefix_frames` 对应自己的 query timestamp；
5. greedy prediction 与 A--D option-logit prediction 数量一致；
6. 输出 JSON、日志和完整 per-question records 均保存。

## 9. Where-to-Look heatmap

脚本：

```text
scripts/evaluate_where_to_look.py
```

每张图有四列：

1. 原始代表帧；
2. vision-only bottleneck attention；
3. audio-conditioned ADBT attention；
4. frozen LLaVA answer-conditioned Grad x Activation teacher。

通用参数：

```text
selection seed = 20260911
videos per split = 4
frames per video = 32
teacher questions per delayed-query video <= 3
heatmap temperature = 0.03
vision-only checkpoint = results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt
vision-only SHA256 = 47be825e896292f0efb855e1069e1277d159fd128ce06deaae7bdbe5e811f48a
```

对于 MLVU，问题位于视频结束后，最多均匀取 3 道 QA 构造 teacher。对于
StreamingBench，每个视频选择按 query time 排序的中间问题，并把视频/音频严格
截断到该问题时间；不会使用 future frame/audio。

### 9.1 VideoMME heatmap

训练集：

```bash
python scripts/evaluate_where_to_look.py \
  --split-file results/adbt_videomme_split.json --split-name train \
  --selection-seed 20260911 --question-policy auto \
  --max-teacher-questions 3 --max-videos 4 --max-frames 32 \
  --vision-checkpoint results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt \
  --audio-checkpoint results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
  --output-dir results/heatmaps/videomme/train \
  --temperature 0.03 --vision-batch-size 8 --beats-batch-size 8
```

验证集只需替换：

```text
--split-name test
--output-dir results/heatmaps/videomme/validation
```

历史 20-video VideoMME heatmaps 位于：
`results/exp02_where_to_look_20videos/`。

### 9.2 MLVU Ego heatmap

训练集：

```bash
python scripts/evaluate_where_to_look.py \
  --dataset-manifest results/dataset_manifests/mlvu_ego.json \
  --split-file results/dataset_splits/mlvu_ego_seed1234_80_20.json \
  --split-name train --selection-seed 20260911 --question-policy auto \
  --max-teacher-questions 3 --max-videos 4 --max-frames 32 \
  --vision-checkpoint results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt \
  --audio-checkpoint results/mlvu_ego_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output-dir results/heatmaps/mlvu_ego/train \
  --temperature 0.03 --vision-batch-size 8 --beats-batch-size 8
```

验证集只需替换：

```text
--split-name test
--output-dir results/heatmaps/mlvu_ego/validation
```

### 9.3 StreamingBench Real-Time heatmap

训练集：

```bash
python scripts/evaluate_where_to_look.py \
  --dataset-manifest results/dataset_manifests/streamingbench_realtime.json \
  --split-file results/dataset_splits/streamingbench_realtime_seed1234_80_20.json \
  --split-name train --selection-seed 20260911 --question-policy auto \
  --max-teacher-questions 3 --max-videos 4 --max-frames 32 \
  --vision-checkpoint results/exp01_visiononly64_train_epoch3_lr1e5/adbt_epoch_3.pt \
  --audio-checkpoint results/streamingbench_realtime_adbt_q64_tau005_epoch3_lr1e5/adbt_epoch_3.pt \
  --output-dir results/heatmaps/streamingbench_realtime/train \
  --temperature 0.03 --vision-batch-size 8 --beats-batch-size 8
```

验证集只需替换：

```text
--split-name test
--output-dir results/heatmaps/streamingbench_realtime/validation
```

当前已生成的 MLVU/StreamingBench 四个目录均含四张 PNG 和一个
`metrics.json`：

- `results/heatmaps/mlvu_ego/train/`；
- `results/heatmaps/mlvu_ego/validation/`；
- `results/heatmaps/streamingbench_realtime/train/`；
- `results/heatmaps/streamingbench_realtime/validation/`。

Heatmap 是小样本机制分析，不是 accuracy 评测。当前结果的共同趋势是 audio map
提高了 top-k teacher-token overlap，但 teacher-distribution CE/JS 更差，因此只能
报告为 mixed evidence，不能据此声称 audio 已学习到稳定的空间语义优势。

若要让 StreamingBench heatmap 与当前 67.00% 的 score-optimal operating point
一致，可把它的 `--temperature 0.03` 改成 `--temperature 0.05`，并使用新的输出
目录；不要覆盖现有跨数据集固定 tau=0.03 的可视化。

## 10. 推荐工作流

```text
检查数据和 split
  -> 使用唯一 output-dir 训练
  -> 校验 checkpoint SHA256 和内部 args
  -> short/small smoke test
  -> 固定 checkpoint、temperature、FPS
  -> 完整 validation
  -> 检查样本覆盖和 per-question records
  -> 最后再生成 heatmap/ablation
```
