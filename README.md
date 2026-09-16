# AudioRouter: Audio-Guided Visual Compression for Streaming Video Language Models

Official implementation of:

**AudioRouter: Audio-Guided Visual Compression for Streaming Video Language Models**

AudioRouter uses synchronized audio as a perception-level routing signal:
**audio determines where to look, while vision determines what the LLM sees.**

Instead of feeding audio tokens into the language model, AudioRouter uses audio to guide visual token compression while preserving the original visual-only VLM interface.

The compressed representation is:

\[
Z_t=P_tV_t
\]

where audio controls the routing weights \(P_t\), while the values remain from native visual tokens.

---

## Overview

AudioRouter is built upon a frozen video language model and learns an audio-conditioned visual bottleneck.

Main components:

- Audio-conditioned query bottleneck
- Audio-guided visual token routing
- Causal streaming inference

The model compresses:

```
196 visual tokens/frame → 64 routed visual tokens/frame
```

while maintaining strong video understanding performance.

---

## Installation

```bash
git clone https://github.com/saber23324/StreamingVideo.git
cd StreamingVideo
```

Create environment:

```bash
conda create -n AudioRouter python=3.10
conda activate AudioRouter

pip install -r requirements.txt
```

Required models:

- LLaVA-OneVision-Qwen2-7B-OV
- BEATs audio encoder

Set environment variables:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH=$(pwd)
```

---

# Quick Start

## 1. Download checkpoint

Example VideoMME checkpoint:

```
results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/
└── adbt_epoch_3.pt
```

---

## 2. Run VideoMME evaluation

```bash
bash scripts/run_phase5_short_ablation.sh \
6 \
real \
results/adbt-phase4.1-q64-tau005-epoch3-lr1e5/adbt_epoch_3.pt \
results/demo-videomme \
0.03 \
videomme_short
```

The evaluation will output:

```
results/demo-videomme/
└── results.json
```

---

# Evaluation

## VideoMME

```bash
python -m AudioRouter.eval_adbt_mcqa \
--dataset videomme \
--checkpoint PATH_TO_CHECKPOINT \
--fps auto \
--max-frames 0 \
--audio-ablation real
```

## MLVU

```bash
python -m AudioRouter.eval_adbt_mcqa \
--dataset mlvu \
--checkpoint PATH_TO_CHECKPOINT \
--fps auto \
--max-frames 0 \
--audio-ablation real
```

## StreamingBench

```bash
python -m AudioRouter.eval_adbt_mcqa \
--dataset streamingbench \
--checkpoint PATH_TO_CHECKPOINT \
--fps auto \
--max-frames 0 \
--audio-ablation real
```

---

# Training

AudioRouter only updates the audio-conditioned visual bottleneck.

Frozen:

- LLaVA language model
- Vision encoder
- Visual projector
- BEATs encoder


Trainable:

- Audio-conditioned query module
- Visual routing module


Example:

```bash
python -m AudioRouter.train_adbt_videomme \
--epochs 3 \
--num-queries 64 \
--bottleneck-hidden 256 \
--attention-temperature 0.05 \
--value-mode native \
--learning-rate 1e-5
```

---
# Reproduced Results

Default configuration:

| Setting | Value |
|-|-|
| Backbone | LLaVA-OneVision-Qwen2-7B-OV |
| Visual tokens | 196 → 64 |
| Query slots | 64 |
| Audio encoder | BEATs |
| Training | Frozen VLM + trainable bottleneck |


Main results:

| Dataset | Accuracy |
|-|-:|
| VideoMME | 60.19 |
| MLVU | 67.83 |
| StreamingBench RTVU | 72.40 |

---

# Visualization

Generate Where-to-Look visualization:

```bash
python scripts/evaluate_where_to_look.py \
--checkpoint PATH_TO_CHECKPOINT \
--output-dir results/heatmaps
```

---

# Notes

- Audio is only used for visual routing.
- No audio token is introduced into the LLM.
- The evaluation follows causal streaming settings without future audio/video access.


---

# Citation

If you find this work useful, please cite:

```bibtex{
xxx.xxx
}
```