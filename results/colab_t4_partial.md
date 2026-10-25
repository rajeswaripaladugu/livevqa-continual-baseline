# Colab T4 Partial Measurements (Real GPU Run)

> [!NOTE]
> The numbers below were produced directly by [`scripts/livevqa_experiments.py`](../scripts/livevqa_experiments.py) in a Google Colab environment, rather than the local [`src/`](../src/) package. Both scripts and implementations are linked for full transparency.

---

### Hardware and Setup
- **Platform**: Google Colab T4 GPU (16 GB VRAM)
- **Quantization & Precision**: 4-bit NormalFloat4 (NF4) QLoRA, `fp16` compute dtype
- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct`
- **Vision Tower**: Kept strictly frozen
- **Adapter Configuration**: LoRA rank $r = 16$, $\alpha = 32$ applied to language-model attention and MLP projections
- **Image Resolution Constraint**: `MAX_PIXELS = 256 * 28 * 28` (200,704 pixels)

---

### Dataset & Split
- **Source**: LiveVQA multiple-choice dataset file containing 10,000 valid questions across 3,795 unique images.
- **Split Protocol**: Image-disjoint split (seed 0) ensuring zero visual overlap between train and test:
  - **Train split**: 7,519 questions
  - **Test split**: 2,481 questions
- **Evaluation Prompt**: Question text formatted with lettered options, followed by:
  ```text
  Answer with the letter of the correct option only (A, B, C, D or E).
  ```

---

### Base-Model Zero-Shot Results

Evaluated on the exact same train/test partitions before fine-tuning:

| Evaluation Split | Raw Score | Accuracy (%) | Wilson 95% Confidence Interval |
| :--- | :---: | :---: | :---: |
| **Seen Subset** | 17 / 60 | 28.3% | [18.5%, 40.8%] |
| **Unseen Subset** | 46 / 150 | 30.7% | [23.9%, 38.5%] |
| **MMStar** (First 100 validation samples) | 46 / 100 | 46.0% | [36.6%, 55.7%] |

---

### Fine-Tuning Execution & Interruption

- **Planned Training**: 7,519 training samples for 2 epochs (1,880 optimizer update steps at effective batch size 8 using gradient accumulation).
- **Execution Status**: The run was **INTERRUPTED at step 791** because the free Google Colab GPU compute quota was exhausted.
- **Post-Training Evaluation**: **No post-training evaluation exists.** Because training was cut short prior to completion and the session terminated, the fine-tuned checkpoint was not evaluated.
- **Optimization Observations**: Two non-finite `fp16` losses were observed during training at **step 435** and **step 539**. In both occurrences, the optimizer step was safely skipped by the PyTorch `GradScaler`.
