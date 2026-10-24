.PHONY: help setup test smoke run-t4 run-full analyze clean

PYTHON ?= python

help:
	@echo "Available targets:"
	@echo "  setup      Install dependencies from requirements.txt"
	@echo "  test       Run test suite with pytest"
	@echo "  smoke      Run fast CPU smoke test pipeline"
	@echo "  run-t4     Run Colab T4 4-bit QLoRA configuration"
	@echo "  run-full   Run Full GPU bf16 LoRA configuration"
	@echo "  analyze    Aggregate results and inject summaries into README.md"
	@echo "  clean      Clean temporary and cached files"

setup:
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests -v

smoke:
	$(PYTHON) -m src.data --generate-synthetic --output data/synthetic_smoke.jsonl --num-samples 16
	$(PYTHON) -m src.train --config configs/cpu_smoke.yaml
	$(PYTHON) -m src.eval --config configs/cpu_smoke.yaml
	$(PYTHON) -m src.analyze

run-t4:
	$(PYTHON) -m src.train --config configs/colab_t4.yaml
	$(PYTHON) -m src.eval --config configs/colab_t4.yaml
	$(PYTHON) -m src.analyze

run-full:
	$(PYTHON) -m src.train --config configs/gpu_full.yaml
	$(PYTHON) -m src.eval --config configs/gpu_full.yaml
	$(PYTHON) -m src.analyze

analyze:
	$(PYTHON) -m src.analyze

clean:
	$(PYTHON) -c "import shutil, glob, os; [shutil.rmtree(p) for p in glob.glob('**/__pycache__', recursive=True)]; [shutil.rmtree(p) for p in glob.glob('.pytest_cache') if os.path.isdir(p)]"
