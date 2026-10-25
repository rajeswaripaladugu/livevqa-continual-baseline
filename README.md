# LiveVQA Continual Baseline: Visual Knowledge Updating & Continual Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research baseline reproducing and investigating the visual knowledge updating experiment from **"LiveVQA: Live Visual Knowledge Seeking"** ([arXiv:2504.05288](https://arxiv.org/abs/2504.05288)). This repository fine-tunes multimodal large language models (Vision-LLMs) with parameter-efficient adapters (LoRA / DoRA, keeping the vision encoder frozen) on newly emerged visual knowledge questions. The core research questions investigate:
1. How much of fine-tuning performance reflects genuine visual knowledge acquisition versus format matching or exploiting label frequency priors?
2. How much general multimodal reasoning capability is retained or forgotten during sequential continual learning updates?

---

## Current Status

1. **Pipeline Implementation & Test Suite**: The core data pipeline (schema validation, image-disjoint splitting, options prompt construction), model loading with frozen vision encoder, shifted label masking, Wilson score 95% confidence intervals, and continual learning replay buffer are implemented and verified. All unit and integration tests pass.
2. **Base-Model Baseline Measurements**: Zero-shot baseline measurements on `Qwen/Qwen2.5-VL-3B-Instruct` were taken on a Google Colab T4 GPU across seen, unseen, and MMStar evaluation subsets using [`scripts/livevqa_experiments.py`](scripts/livevqa_experiments.py) (recorded in [`results/colab_t4_partial.md`](results/colab_t4_partial.md)).
3. **Full Fine-Tuning Status**: Fine-tuning on the full 7,519-sample LiveVQA training set was started on a Colab T4 GPU for 2 epochs (1,880 optimizer steps at effective batch size 8). The execution was **interrupted at step 791** due to free Colab GPU compute quota exhaustion. No post-training evaluation exists yet; full fine-tuning and adapter ablations require dedicated GPU access.
4. **Continual Learning Pipeline**: Sequential updating across $K$ chronological stages, reservoir experience replay, and Backward Transfer / Forgetting matrix calculation are implemented and tested on CPU smoke test runs, but have not yet been evaluated on real full-dataset benchmarks.

---

## Real Results (Partial: Google Colab T4)

Before fine-tuning was interrupted, zero-shot baseline measurements were recorded on Google Colab T4 using [`scripts/livevqa_experiments.py`](scripts/livevqa_experiments.py). Full setup details and logs are preserved in [`results/colab_t4_partial.md`](results/colab_t4_partial.md).

### Setup Summary
- **Hardware**: Google Colab T4 GPU (16 GB VRAM)
- **Model**: `Qwen/Qwen2.5-VL-3B-Instruct` with 4-bit NormalFloat4 (NF4) QLoRA, `fp16` compute dtype
- **Vision Tower**: Kept strictly frozen; LoRA rank $r = 16$, $\alpha = 32$ applied to language-model attention and MLP projections
- **Resolution**: `MAX_PIXELS = 256 * 28 * 28`
- **Data Split**: 10,000 multiple-choice questions over 3,795 unique images; image-disjoint split: 7,519 train / 2,481 test (seed 0)
- **Prompt**: Question text with lettered options, followed by: `"Answer with the letter of the correct option only (A, B, C, D or E)."`

### Zero-Shot Base-Model Performance

| Evaluation Split | Raw Score | Accuracy (%) | Wilson 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Seen Split Subset** | 17 / 60 | 28.3% | [18.5%, 40.8%] |
| **Unseen Split Subset** | 46 / 150 | 30.7% | [23.9%, 38.5%] |
| **MMStar** (First 100 validation samples) | 46 / 100 | 46.0% | [36.6%, 55.7%] |

### Training Interruption
- Fine-tuning began on all 7,519 training samples for 2 planned epochs (1,880 optimizer steps at effective batch size 8 via gradient accumulation).
- The run was **interrupted at step 791** due to Colab GPU quota exhaustion.
- **No post-training evaluation exists.**
- Two non-finite `fp16` loss values were encountered during training at step 435 and step 539; in both cases, the optimizer step was skipped by the PyTorch `GradScaler`.

---

## 1. Research Environment & System Specifications

- **Local Validation Platform**: Windows 11 CPU (Intel / AMD processor, PyTorch CPU). The modular `src/` codebase was validated on this platform using `SmolVLM-256M-Instruct` and `TinyVLM`.
- **GPU Evaluation Platform**: Google Colab T4 (16 GB VRAM) running [`scripts/livevqa_experiments.py`](scripts/livevqa_experiments.py).
- **GPU Cluster Target (Planned)**: Full multi-GPU node (RTX 3090/4090/A100) using bfloat16 LoRA (`configs/gpu_full.yaml`).
- **Audit Logging**: Every training and evaluation run records timestamp, git commit, config hash, seed, device specs, library versions (`torch`, `transformers`, `peft`), sample counts, wall-clock time, and final training loss in `results/results.csv`. Raw model outputs are saved to `results/predictions_<run_id>.jsonl`.

---

## 2. Quickstart

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/livevqa-continual-baseline.git
cd livevqa-continual-baseline
pip install -r requirements.txt
```
*(Note: Replace `<YOUR_GITHUB_USERNAME>` with your GitHub username or organization where the repository is hosted.)*

### Run Fast CPU Smoke Test (< 5 minutes)
Runs the end-to-end verification pipeline (synthetic data generation, model adapter loading, masked label training, strict evaluation, and analysis):
```bash
make smoke
```
Or manually:
```bash
# 1. Generate 16 synthetic visual QA items
python -m src.data --generate-synthetic --output data/synthetic_smoke.jsonl --num-samples 16

# 2. Train tiny VLM on CPU
python -m src.train --config configs/cpu_smoke.yaml

# 3. Evaluate seen vs unseen and audit predictions
python -m src.eval --config configs/cpu_smoke.yaml

# 4. Generate summary and update README
python -m src.analyze
```

### Run Full Test Suite
```bash
pytest tests/ -v
```

### Inspect Dataset Statistics and Majority Baseline
To inspect any dataset file, calculate unique images, option distributions, and majority-class baseline:
```bash
python -m src.data --stats data/livevqa_laptop_mini.jsonl
```

---

## 3. How to Reproduce the Colab Baseline

To run the Qwen2.5-VL-3B-Instruct baseline on a Google Colab T4 GPU (16 GB):
1. Open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) in Google Colab.
2. Select runtime type: **Python 3 with T4 GPU**.
3. Install required packages: `pip install -r requirements.txt bitsandbytes qwen-vl-utils`.
4. Run [`scripts/livevqa_experiments.py`](scripts/livevqa_experiments.py) with `--zero-shot-only` to verify base-model accuracy across seen, unseen, and MMStar subsets.

---

## 4. Methodology

```
┌────────────────────────────────────────────────────────┐
│               LiveVQA Multimodal Sample                │
│    [Image] + "Question: ... Options: A/B/C/D/E Answer:"│
└──────────────────────────┬─────────────────────────────┘
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
   ┌──────────────────┐        ┌──────────────────┐
   │  Vision Tower    │        │ LLM Backbone     │
   │  (FROZEN via     │        │ (LoRA / DoRA     │
   │   regex pattern) │        │  q/k/v/o/gate)   │
   └─────────┬────────┘        └─────────┬────────┘
             │                           │
             └─────────────┬─────────────┘
                           ▼
              Cross-Entropy Loss (Shifted)
          *Labels masked with -100 for all*
          *prompt tokens; only target answer*
          *tokens (e.g. ' C') contribute to loss*
```

### Models & Adapters
- **Target Architecture**: `Qwen/Qwen2.5-VL-3B-Instruct` (and `SmolVLM-256M-Instruct` for local CPU smoke verification).
- **Vision Tower**: Kept strictly **frozen** via module regex exclusion (`.*visual.*|.*vision.*`).
- **Adapter Configurations**:
  - Smoke test: LoRA rank $r = 8$, $\alpha = 16$.
  - Colab T4: LoRA rank $r = 16$, $\alpha = 32$.
  - Full GPU (planned): LoRA rank $r = 32$, $\alpha = 64$.
  - Target modules: Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
  - Quantization: 4-bit NF4 with float16 compute dtype via `bitsandbytes` (on CUDA).

### Split-by-Image Protocol
Visual knowledge datasets frequently contain multiple questions per image. Standard random row splitting causes severe visual data leakage. The loader in `src/data.py` groups questions by image identifier and performs an **image-disjoint split**:
$$\text{Images}(\mathcal{D}_{\text{train}}) \cap \text{Images}(\mathcal{D}_{\text{eval}}) = \emptyset$$

### Continual Learning Formulation
The dataset is split into $K$ chunks (chronological if timestamped, else sequential order). After training on stage $i \in \{1, \dots, K\}$, the model is evaluated on all chunks $j \in \{1, \dots, K\}$, yielding an accuracy matrix $R \in \mathbb{R}^{K \times K}$:
- **Backward Transfer (BWT)**:
  $$\text{BWT} = \frac{1}{K-1} \sum_{j=1}^{K-1} (R_{K,j} - R_{j,j})$$
- **Average Forgetting**:
  $$F_j = \max_{l \in \{1, \dots, K-1\}} (R_{l,j} - R_{K,j}), \quad \bar{F} = \frac{1}{K-1} \sum_{j=1}^{K-1} \max(0, F_j)$$

---

## 5. Evaluation Pitfalls & Findings

During the design and calibration of this benchmark, three empirical findings were observed:

1. **Prompt Phrasing & Free-Text Answer Degeneration**:
   - The initial prompt instruction allowed *"choice letter or short name"*. Under this prompt, the base model answered with raw option content or open-ended phrases (examples actually observed in predictions: `'1000K'`, `'Residual Connection'`, `'W1-W2'`, and letter `'A'` when the gold answer was `'D'`).
   - Evaluating these outputs with strict letter matching resulted in an apparent accuracy of ~10%. This was a prompt parsing failure rather than an inability to recognize visual information.
   - Constraining the prompt with an explicit letter-only directive (*"Answer with the letter of the correct option only (A, B, C, D or E)."*) resolved this format discrepancy.
2. **Missing Options in Raw Questions (Pre-Fix Pilot)**:
   - In raw data files, question strings did not contain the lettered options (for example, *"what is the electron temperature used for the particle injection at the beginning of the simulation?"* with gold letter `C`).
   - Without the options provided in the prompt, the model had no candidate choices and was forced into open-ended guessing.
   - In a 130-sample pilot, the gold answer distribution was heavily skewed: letter `C` accounted for 40.8% of items (C: 53, B: 26, D: 25, A: 17, E: 9). A pilot fine-tune scoring ~43% to 47% under this condition was indistinguishable from simply predicting the majority label `C`. *(This pilot is superseded, pre-fix).*
3. **Effect of Formatting Candidate Options**:
   - Explicitly formatting all candidate choices into the prompt text moved the zero-shot base model performance to ~30% on LiveVQA.
   - For the full 10,000-question dataset, the majority-class baseline should always be verified directly from the dataset file via `python -m src.data --stats <path>` to benchmark against label priors.

---

## 6. Experimental Results

> [!NOTE]
> All tables, metrics, and plots below are compiled by `src/analyze.py` from verified execution records in `results/results.csv`. Hand-crafted result numbers are prohibited.

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
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_smoke.yaml` (HuggingFaceTB/SmolVLM-256M-Instruct) | **PIPELINE SMOKE TEST: sample too small for conclusions** | 12 | 12 | 0.00% [0.0%, 24.2%] | 4 | 0.00% [0.0%, 49.0%] | 50.00% (A) | 0.26s | 7.7830 | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `uncommitted` |
| `cpu_smoke.yaml` (HuggingFaceTB/SmolVLM-256M-Instruct) | **PIPELINE SMOKE TEST: sample too small for conclusions** | 12 | 12 | 0.00% [0.0%, 24.2%] | 4 | 0.00% [0.0%, 49.0%] | 50.00% (A) | 0.65s | 7.2365 | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `uncommitted` |

### Continual Learning Evaluation Matrix (Pipeline Verification)
- **Stages (Chunks K)**: 2
- **Replay Capacity**: 4 samples
- **Backward Transfer (BWT)**: 0.00%
- **Average Forgetting**: 0.00%


<!-- RESULTS:END -->

### Generated Diagnostic Visualizations
When real or smoke experiments are analyzed, diagnostic plots are saved to `results/`:
- `results/rank_dora_comparison.png`: Comparison across LoRA ranks ($r \in \{8, 16\}$).
- `results/generalization_gap.png`: Comparison of accuracy on seen vs. unseen splits against the majority baseline.
- `results/forgetting_matrix.png`: Heatmap of the continual learning matrix.

---

## 7. Hyperparameter Configurations & Execution Status

| Experiment Tier | Model | Ranks ($r$) | Precision | Epochs | Batch Size (Effective) | Format | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **CPU Smoke Test** (`configs/cpu_smoke.yaml`) | `TinyVLM` / `SmolVLM-256M` | 8 | float32 | 1 | 4 (batch 2, accum 2) | MCQA | **Verified** |
| **Laptop Mini** (`configs/laptop_cpu.yaml`) | `SmolVLM-256M` | 16 | float32 | 1 | 4 (batch 2, accum 2) | MCQA | **Smoke test only** (24 samples) |
| **Colab T4** (`configs/colab_t4.yaml`) | `Qwen2.5-VL-3B-Instruct` | 16 | 4-bit NF4 / fp16 | 2 planned | 8 (batch 1, accum 8) | MCQA | **Zero-shot verified; fine-tuning interrupted at step 791/1,880** |
| **Full GPU Node** (`configs/gpu_full.yaml`) | `Qwen2.5-VL-3B-Instruct` | 32 | bfloat16 | 2 planned | 128 (batch 8, accum 16) | MCQA | **Planned (requires GPU cluster)** |
| **Format Ablations** (QA / QAR) | Any | - | - | - | - | QA, QAR | **Scaffolded in data loader; planned** |

---

## 8. Limitations & Differences from the Paper

1. **Quantization & Hardware Constraints**: The original paper evaluated full bfloat16 fine-tuning across high-memory GPUs (A100/H100). On a single Colab T4 (16 GB), 4-bit QLoRA is required to prevent out-of-memory errors.
2. **Effective Batch Size**: Due to memory limitations on a single T4 GPU, gradient accumulation was set to an effective batch size of 8, compared to batch sizes of 128–256 in large-scale cluster runs.
3. **Training Interruption**: Fine-tuning on Colab was cut short at step 791/1,880 due to quota limits, meaning no post-training checkpoint or fine-tuned accuracy was obtained on the full dataset.
4. **General Benchmark Scope**: For rapid auditability on Colab, the first 100 validation samples of MMStar were evaluated.

---

## 9. Continual Learning Roadmap

- [x] $K$-chunk chronological sequential updating framework.
- [x] Experience-replay buffer with configurable reservoir sampling.
- [x] Computation of Backward Transfer (BWT) and Average Forgetting matrix.
- [ ] Integration of Parameter-Allocation methods (InfLoRA / O-LoRA) to project updates into the null space of previous knowledge.
- [ ] Evaluation on full real-data continual visual streams.

---

## 10. References & Citations

- **LiveVQA**: *Live Visual Knowledge Seeking* (2025). [arXiv:2504.05288](https://arxiv.org/abs/2504.05288)
- **LoRA Learns Less and Forgets Less**: Biderman et al. (2024). [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- **CoIN**: *Continual Instruction Tuning for Large Multimodal Models* (2024). [arXiv:2403.08350](https://arxiv.org/abs/2403.08350)
- **TiC-CLIP**: *Continual Training of CLIP Models on Time-Continuous Web Data* (2023). [arXiv:2310.16226](https://arxiv.org/abs/2310.16226)
- **InfLoRA**: *Interference-Free Low-Rank Adaptation for Continual Learning* (2024). [arXiv:2404.00228](https://arxiv.org/abs/2404.00228)
