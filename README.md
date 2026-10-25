# LiveVQA Continual Baseline: Visual Knowledge Updating

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research baseline reproducing and evaluating the visual knowledge updating experiment from **"LiveVQA: Live Visual Knowledge Seeking"** ([arXiv:2504.05288](https://arxiv.org/abs/2504.05288)). This repository fine-tunes multimodal large language models (Vision-LLMs) with parameter-efficient adapters (4-bit QLoRA, keeping the vision encoder frozen) on newly emerged visual knowledge questions, and measures retention across sequential continual updates.

---

## Executive Summary

| Dimension | Specification / Verified Finding |
| :--- | :--- |
| **Target Paper** | *LiveVQA: Live Visual Knowledge Seeking* ([arXiv:2504.05288](https://arxiv.org/abs/2504.05288)) |
| **Evaluated Model** | `Qwen/Qwen2.5-VL-3B-Instruct` (Vision encoder frozen; 4-bit NF4 QLoRA, $r=16, \alpha=32$) |
| **Evaluated Dataset** | LiveVQA (10,000 multiple-choice QA pairs; 7,519 train / 2,481 eval split, seed 0) |
| **Zero-Shot Baselines** | **Seen**: 28.3% (95% CI: [18.5%, 40.8%]) \| **Unseen**: 30.7% (95% CI: [23.9%, 38.5%]) \| **MMStar**: 46.0% (vs. 25.0% random baseline) |
| **Fine-Tuning Execution** | Google Colab T4: training loss decreased from **3.14 down to ~1.05** before session was interrupted at step 791/1,880 due to quota exhaustion. No post-training eval exists yet. |
| **Code Health & Testing** | **All unit tests passing** (`pytest tests/ -v`); fast local CPU smoke pipeline (`make smoke`). |

---

## 1. Real Colab T4 Experimental Measurements

Zero-shot baselines and fine-tuning measurements were taken on Google Colab T4 using [`scripts/livevqa_experiments.py`](scripts/livevqa_experiments.py). Full logs are preserved in [`results/colab_t4_partial.md`](results/colab_t4_partial.md).

### Base-Model Zero-Shot Accuracy (Before Training)

| Evaluation Split | Samples | Raw Score | Accuracy (%) | Wilson 95% Confidence Interval |
| :--- | :---: | :---: | :---: | :---: |
| **LiveVQA Seen Subset** | 60 | 17 / 60 | **28.3%** | [18.5%, 40.8%] |
| **LiveVQA Unseen Subset** | 150 | 46 / 150 | **30.7%** | [23.9%, 38.5%] |
| **MMStar Benchmark** | 100 | 46 / 100 | **46.0%** | [36.6%, 55.7%] |
| *Random 4-Choice Baseline* | - | - | *25.0%* | - |

*Finding*: The base 3B vision-language model is marginally above chance on LiveVQA without task adaptation.

### Fine-Tuning Execution
- **Setup**: `Qwen/Qwen2.5-VL-3B-Instruct`, 4-bit NormalFloat4 (NF4) QLoRA ($r=16, \alpha=32$), `fp16` compute, image resolution capped at `MAX_PIXELS = 200,704`.
- **Progress**: Executed on all 7,519 train samples up to **step 791 / 1,880** (effective batch size 8).
- **Loss Convergence**: Loss dropped from **3.14** (step 1) to **1.94** (step 200) to **~1.05** (step 750).
- **Status**: The run was **interrupted at step 791** due to Colab free GPU compute quota exhaustion. **No post-training evaluation exists.**
- **Numerical Stability**: Two non-finite `fp16` loss values (steps 435 and 539) were encountered; PyTorch's `GradScaler` safely skipped those gradient updates without divergence.

---

## 2. Key Engineering Findings

1. **Resolution Clamping (`MAX_PIXELS = 200,704`)**:
   Dynamic visual tokenization in Qwen2.5-VL caused CUDA OOMs on 15 GB VRAM when high-resolution images occurred. Clamping pixels strictly stabilized VRAM at ~12.8 GB / 15.0 GB.
2. **Eliminating Option Bias**:
   In early pilots without candidate option letters in the prompt, the model had no choice candidates and skewed heavily toward letter "C" (40.8% majority class artifact). Explicitly formatting lettered options `(A), (B), (C), (D)` eliminated this artifact.
3. **Image-Disjoint Splitting**:
   LiveVQA contains multiple questions per image. The dataset pipeline enforces strict image-disjoint splits ($\text{Images}_{\text{train}} \cap \text{Images}_{\text{eval}} = \emptyset$) to prevent visual data leakage.

---

## 3. Quickstart (Local Verification)

```bash
# 1. Install requirements
pip install -r requirements.txt

# 2. Run fast CPU smoke test (< 1 min, verifies training, eval, and continual learning)
make smoke

# 3. Run automated unit test suite
pytest tests/ -v
```

---

## 4. Automated Research Results Summary

> [!NOTE]
> All tables below are compiled automatically by `src/analyze.py` from verified execution records in `results/results.csv`. Hand-crafted result numbers are strictly prohibited.

<!-- RESULTS:START -->

## Automated Research Results Summary

### Real Experiments (≥ 50 Evaluation Questions)

> [!NOTE]
> **No completed full-dataset runs (N ≥ 50) have been executed in this local environment.**
> Base-model zero-shot baseline measurements on `Qwen/Qwen2.5-VL-3B-Instruct` were executed on a Google Colab T4 GPU via `scripts/livevqa_experiments.py` (documented in [`results/colab_t4_partial.md`](results/colab_t4_partial.md)). Full fine-tuning was interrupted at step 791/1880 due to Colab free GPU quota exhaustion. Full real-dataset fine-tuning requires GPU access.

### Pipeline Smoke Tests (Synthetic or Tiny Datasets, N < 50)
> [!WARNING]
> **Research Integrity Notice**: All runs below are pipeline smoke tests (<50 evaluation questions or synthetic data). Every such run is explicitly labelled **PIPELINE SMOKE TEST: sample too small for conclusions**. Per research integrity rules, these numbers verify code execution only and must NEVER be cited as research findings.

| Run ID / Model | Label | Train N | Seen Eval N | Seen Acc (95% CI) | Unseen Eval N | Unseen Acc (95% CI) | Eval Majority Baseline | Wall-Clock (s) | Loss | Device | Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_smoke.yaml` (HuggingFaceTB/SmolVLM-256M-Instruct) | **PIPELINE SMOKE TEST: sample too small for conclusions** | 12 | 12 | 0.00% [0.0%, 24.2%] | 4 | 0.00% [0.0%, 49.0%] | 50.00% (A) | 0.26s | 7.7830 | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `uncommitted` |

### Continual Learning Evaluation Matrix (Pipeline Verification)
- **Stages (Chunks K)**: 2
- **Replay Capacity**: 4 samples
- **Backward Transfer (BWT)**: 0.00%
- **Average Forgetting**: 0.00%

<!-- RESULTS:END -->

---

## 5. Limitations & Next Steps

1. **Post-Training Evaluation**: Requires dedicated GPU access to finish steps 792–1,880 and evaluate post-training retention and Backward Transfer (BWT).
2. **Precision**: Colab T4 used 4-bit NF4 QLoRA due to 15 GB VRAM; full bfloat16 cluster training remains future work.
3. **Reference Paper**: *LiveVQA: Live Visual Knowledge Seeking* ([arXiv:2504.05288](https://arxiv.org/abs/2504.05288)).
