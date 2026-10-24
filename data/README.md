# Dataset Setup & Instructions

## 1. LiveVQA Benchmark

The **LiveVQA** dataset is introduced in the research paper:
> *"LiveVQA: Live Visual Knowledge Seeking"* (arXiv:2504.05288).  
> Official Project Page: [https://livevqa.github.io](https://livevqa.github.io)

### How to Obtain LiveVQA
1. Visit the official project repository and release page at [https://livevqa.github.io](https://livevqa.github.io).
2. Download the question annotations (`livevqa_annotations.json` or `.jsonl`) and the visual knowledge image archive.
3. Extract images into `data/images/`.
4. Place the annotation JSON/JSONL into `data/livevqa_qa.jsonl`.

### Expected Schema
Each sample in the JSONL should follow the schema:
```json
{
  "id": "livevqa_00123",
  "image": "images/event_2024_01.jpg",
  "question": "What entity or object is featured in this 2024 update?",
  "options": {
    "A": "Option text 1",
    "B": "Option text 2",
    "C": "Option text 3",
    "D": "Option text 4"
  },
  "answer": "B",
  "date": "2024-03-15"
}
```
*Note: The dataset loader in `src/data.py` also supports `options` as a list of strings (`["Option text 1", ...]`), and questions where options are already formatted directly into the question text.*

---

## 2. General Knowledge Benchmark (MMStar)

To evaluate knowledge forgetting on general visual reasoning, we use the **MMStar** multimodal benchmark:
> *"MMStar: Are We on the Right Track for Multimodal Evaluation?"*

1. Download the MMStar evaluation subset from Hugging Face: `datasets/Lin-Chen/MMStar`.
2. Save the benchmark items to `data/mmstar_eval.jsonl`.
3. MMStar items contain multiple-choice visual questions formatted with 4 options ('A', 'B', 'C', 'D').

---

## 3. Synthetic Data (For Smoke Tests Only)

> [!NOTE]  
> In accordance with research integrity rules, **synthetic data is strictly for pipeline verification and smoke testing**. Synthetic scores must never be reported as research results.

To generate a synthetic dataset (16 geometric visual questions with rendered shapes and colors) for fast local smoke testing:
```bash
python -m src.data --generate-synthetic --output data/synthetic_smoke.jsonl --num-samples 16
```
This generates `data/synthetic_smoke.jsonl` along with rendered images in `data/synthetic_images/`.
