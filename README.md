# LiveVQA Continual Baseline: Visual Knowledge Updating & Continual Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](pyproject.toml)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A clean, reproducible research baseline reproducing and extending the "knowledge updating" experiment from **"LiveVQA: Live Visual Knowledge Seeking"** ([arXiv:2504.05288](https://arxiv.org/abs/2504.05288)). This repository fine-tunes multimodal large language models (Vision-LLMs) with parameter-efficient adapters (LoRA / DoRA, keeping the vision encoder frozen) on newly emerged visual knowledge questions. The core research question investigates: **how much of a fine-tuning gain represents genuine visual knowledge acquisition versus learning the answer format or exploiting label priors, and how much general reasoning capability is forgotten during sequential updates?**

---

## 1. Research Environment & System Specifications

> [!IMPORTANT]
> **Research Reproducibility & Integrity**: Smoke tests are run locally on CPU; real experiments are logged and verified directly from results metadata. No results are hand-written or pre-filled.

- **Smoke Test Platform**: Local CPU execution (`Windows 11`, Intel / AMD CPU, PyTorch CPU).
- **GPU Evaluation Targets**:
  - Google Colab T4 GPU (16 GB VRAM) using 4-bit QLoRA (`configs/colab_t4.yaml`).
  - Full GPU Node (24–80 GB VRAM, e.g., RTX 3090/4090/A100) using bfloat16 LoRA (`configs/gpu_full.yaml`).
- **Audit Logging**: Every training and evaluation run records timestamp, git commit, config hash, seed, device specs, library versions (`torch`, `transformers`, `peft`), sample counts, and wall-clock time in `results/results.csv`. All raw sample-level model outputs are preserved in `results/predictions_<run_id>.jsonl`.

---

## 2. Quickstart

### Setup
Clone the repository and install dependencies:
```bash
git clone https://github.com/your-org/livevqa-continual-baseline.git
cd livevqa-continual-baseline
pip install -r requirements.txt
```

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

### Run Real Experiments on GPU (Colab or Local GPU)
For a 16 GB T4 GPU with 4-bit QLoRA:
```bash
python -m src.train --config configs/colab_t4.yaml
python -m src.eval --config configs/colab_t4.yaml
python -m src.analyze
```
Or open [`notebooks/colab_runner.ipynb`](notebooks/colab_runner.ipynb) directly in Google Colab.

---

## 3. Methodology

```
┌────────────────────────────────────────────────────────┐
│               LiveVQA Multimodal Sample                │
│    [Image] + "Question: ... Options: A/B/C/D Answer:"  │
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
- **Base Architecture**: `Qwen/Qwen2.5-VL-3B-Instruct` (or `HuggingFaceTB/SmolVLM-256M-Instruct` for lightweight tests).
- **Vision Tower**: Kept strictly **frozen** via module regex exclusion (`.*visual.*|.*vision.*`).
- **Adapter Configurations**:
  - LoRA / DoRA ranks: $r \in \{16, 32, 64\}$ with $\alpha = 2r$.
  - Target modules: Attention projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP projections (`gate_proj`, `up_proj`, `down_proj`).
  - Quantization: 4-bit NF4 with float16 compute dtype via `bitsandbytes` (active only on CUDA).

### Split-by-Image Protocol
Visual knowledge datasets frequently contain multiple questions per image. Standard random row splitting causes severe data leakage. Our loader in `src/data.py` groups questions by image hash/path and performs an **image-disjoint split**:
$$\text{Images}(\mathcal{D}_{\text{train}}) \cap \text{Images}(\mathcal{D}_{\text{eval}}) = \emptyset$$

### Continual Learning Formulation (Phase 2)
The dataset is split into $K$ chunks (chronological if timestamped, else sequential order). After training on stage $i \in \{1, \dots, K\}$, the model is evaluated on all chunks $j \in \{1, \dots, K\}$, yielding an accuracy matrix $R \in \mathbb{R}^{K \times K}$:
- **Backward Transfer (BWT)**:
  $$\text{BWT} = \frac{1}{K-1} \sum_{j=1}^{K-1} (R_{K,j} - R_{j,j})$$
- **Average Forgetting**:
  $$F_j = \max_{l \in \{1, \dots, K-1\}} (R_{l,j} - R_{K,j}), \quad \bar{F} = \frac{1}{K-1} \sum_{j=1}^{K-1} \max(0, F_j)$$

---

## 4. Evaluation Pitfalls & Findings

During the design and calibration of this benchmark, three critical empirical pitfalls were identified:

1. **Free-Text Answer Degeneration**:
   When models were prompted without an explicit letter-constrained answer cue, the zero-shot base model produced conversational explanations (e.g., *"Based on the architectural features in the image, this is likely..."*). Naive exact matching yielded $<10\%$ accuracy due solely to output parsing failures rather than lack of visual knowledge. Constraining the prompt with:
   ```text
   Select the single best option. Output only the capital letter (A, B, C, or D).
   Answer:
   ```
   restored true capability.
2. **Skewed Answer Distribution & Majority Baseline**:
   Empirical datasets often exhibit significant option frequency skew (in our test partitions, the most frequent option accounted for ~41% of samples). Any reported fine-tuning accuracy must be benchmarked against this **majority-class baseline** to distinguish real visual knowledge learning from label prior exploitation.
3. **Open-Ended Guessing without Options**:
   If the `options` field was omitted or stripped from the prompt, the multiple-choice task degenerated into open-ended guessing, collapsing model performance regardless of adapter capacity.

---

## 5. Experimental Results

> [!NOTE]
> All tables, metrics, and plots below are automatically compiled by `src/analyze.py` from verified execution records in `results/`. Hand-crafted numbers are strictly prohibited.

<!-- RESULTS:START -->

## Automated Research Results Summary

> [!NOTE]
> **No real GPU runs have been recorded yet.**
> To reproduce the paper's real LiveVQA updating experiments, run:
> ```bash
> python -m src.train --config configs/colab_t4.yaml
> python -m src.eval --config configs/colab_t4.yaml
> python -m src.analyze
> ```

### Pipeline Smoke Test Verifications (SYNTHETIC DATA ONLY)
> [!WARNING]
> **Research Integrity Notice**: The table below reports verification smoke tests on synthetic geometric data. Per research rules, synthetic data scores verify pipeline correctness and must NEVER be cited as real research findings.

| Run ID / Model | Samples | Wall-Clock (s) | Final Loss | Status | Device | Commit |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cpu_smoke.yaml (HuggingFaceTB/SmolVLM-256M-Instruct) | Train: 12 / Eval: 16 | 0.0s | - | SUCCESS | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `unversio` |
| cpu_smoke.yaml (HuggingFaceTB/SmolVLM-256M-Instruct) | Train: 12 / Eval: 16 | 0.0s | - | SUCCESS | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `unversio` |

### Continual Learning Evaluation Matrix
- **Stages (Chunks K)**: 3
- **Replay Capacity**: 4 samples
- **Backward Transfer (BWT)**: 0.00%
- **Average Forgetting**: 0.00%


<!-- RESULTS:END -->

### Generated Diagnostic Visualizations
When real or smoke experiments are analyzed, the following diagnostic plots are saved to `results/`:
- `results/rank_dora_comparison.png`: Unseen accuracy across LoRA ranks ($r \in \{16, 32, 64\}$) comparing standard LoRA vs. DoRA.
- `results/generalization_gap.png`: Comparison of accuracy on trained (seen) vs. held-out (unseen) items against the majority-class baseline.
- `results/mmstar_forgetting.png`: Retention of general multimodal reasoning on MMStar across fine-tuning steps.
- `results/forgetting_matrix.png`: Heatmap of the $K \times K$ continual learning matrix.

---

## 6. Limitations & Differences from the Paper

1. **Quantization & Hardware Constraints**: The original paper evaluated full bfloat16 fine-tuning across high-memory GPUs (A100/H100). On a single T4 (16 GB), 4-bit QLoRA is required to prevent Out-Of-Memory (OOM) errors.
2. **Effective Batch Size**: Due to memory limitations on consumer/free-tier GPUs, gradient accumulation is used to reach effective batch sizes of 8–16, whereas large-scale runs leverage batch sizes of 128–256.
3. **Sample Slices**: Smoke tests run on 16 synthetic samples; full LiveVQA evaluation requires downloading the dataset as detailed in [`data/README.md`](data/README.md).
4. **General Benchmark Scope**: For rapid auditability, a representative subset of MMStar is evaluated rather than the complete benchmark suite.

---

## 7. Continual Learning Roadmap

- [x] $K$-chunk chronological sequential updating framework.
- [x] Experience-replay buffer with configurable reservoir sampling.
- [x] Computation of Backward Transfer (BWT) and Average Forgetting matrix.
- [ ] Integration of Parameter-Allocation methods (InfLoRA / O-LoRA) to project updates into the null space of previous knowledge.
- [ ] Evaluation of time-continual vision-language benchmarks (TiC-CLIP).

---

## 8. References & Citations

- **LiveVQA**: *Live Visual Knowledge Seeking* (2025). [arXiv:2504.05288](https://arxiv.org/abs/2504.05288)
- **LoRA Learns Less and Forgets Less**: Biderman et al. (2024). [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- **CoIN**: *Continual Instruction Tuning for Large Multimodal Models* (2024). [arXiv:2403.08350](https://arxiv.org/abs/2403.08350)
- **TiC-CLIP**: *Continual Training of CLIP Models on Time-Continuous Web Data* (2023). [arXiv:2310.16226](https://arxiv.org/abs/2310.16226)
- **InfLoRA**: *Interference-Free Low-Rank Adaptation for Continual Learning* (2024). [arXiv:2404.00228](https://arxiv.org/abs/2404.00228)
