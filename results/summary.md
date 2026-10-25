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
| `cpu_smoke.yaml` (HuggingFaceTB/SmolVLM-256M-Instruct) | **PIPELINE SMOKE TEST: sample too small for conclusions** | 12 | 12 | 0.00% [0.0%, 24.2%] | 4 | 0.00% [0.0%, 49.0%] | 50.00% (A) | 0.74s | 6.7846 | Intel64 Family 6 Model 140 Stepping 2, GenuineIntel | `uncommitted` |

### Continual Learning Evaluation Matrix (Pipeline Verification)
- **Stages (Chunks K)**: 2
- **Replay Capacity**: 4 samples
- **Backward Transfer (BWT)**: 0.00%
- **Average Forgetting**: 0.00%
