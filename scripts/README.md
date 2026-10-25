# Scripts Overview

This directory contains standalone execution scripts used for cloud GPU runs (e.g., Google Colab).

### `livevqa_experiments.py` vs. `src/` Package

- **[`scripts/livevqa_experiments.py`](livevqa_experiments.py)**:
  - A self-contained, single-file script designed specifically for free-tier **Google Colab T4 GPUs** (16 GB VRAM).
  - Configures 4-bit NormalFloat4 (NF4) QLoRA with `bitsandbytes`, frozen vision encoder, and image resolution constrained to `MAX_PIXELS = 256 * 28 * 28` to fit within T4 memory limitations.
  - Used to produce the baseline zero-shot measurements recorded in [`results/colab_t4_partial.md`](../results/colab_t4_partial.md).
  - Its fine-tuning run was interrupted at step 791/1,880 when the Colab compute quota ran out.

- **[`src/`](../src/) Package**:
  - The structured, modular research library providing full configuration management (`configs/`), split-by-image logic, prompt formatting (MCQA / QA / QAR), Wilson confidence interval computation, experience replay continual learning, and automated audit logging.
  - Tested and validated end-to-end on local CPU and smoke test environments with 100% test coverage (`tests/`).
  - Intended for systematic replication, multi-epoch runs, and cluster execution when GPU resources are available.
