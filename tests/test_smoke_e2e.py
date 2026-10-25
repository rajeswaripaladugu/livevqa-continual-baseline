"""End-to-end CPU smoke test running training, evaluation, continual learning, and analysis."""

from pathlib import Path
import yaml

from src.analyze import analyze
from src.continual import run_continual_learning
from src.data import generate_synthetic_dataset
from src.eval import evaluate
from src.train import train


def test_end_to_end_cpu_smoke(tmp_path) -> None:
    """Execute complete CPU pipeline with configs/cpu_smoke.yaml."""
    repo_root = Path(__file__).parent.parent
    config_path = repo_root / "configs" / "cpu_smoke.yaml"
    synthetic_data_path = repo_root / "data" / "synthetic_smoke.jsonl"

    # 1. Generate 16 synthetic samples
    gen_path = generate_synthetic_dataset(
        output_path=synthetic_data_path,
        num_samples=16,
        seed=0,
    )
    assert gen_path.is_file(), "Synthetic dataset file was not generated"

    # 2. Run Training on CPU
    train_out = train(config_path)
    assert "metadata" in train_out
    assert train_out["metadata"]["num_train_samples"] > 0
    assert "final_loss" in train_out
    assert Path(train_out["checkpoint_dir"]).is_dir()

    # 3. Run Evaluation on Seen and Unseen
    eval_out = evaluate(
        config_path,
        model=train_out["model"],
        processor=train_out["processor"],
        run_id="cpu_smoke_test_run",
    )
    assert "seen" in eval_out
    assert "unseen" in eval_out
    assert Path(eval_out["predictions_file"]).is_file()

    # 4. Run Continual Learning on 2 Chunks
    continual_out = run_continual_learning(
        config_path,
        num_chunks=2,
        replay_capacity=4,
        epochs_per_stage=1,
    )
    assert "accuracy_matrix" in continual_out
    assert len(continual_out["accuracy_matrix"]) == 2

    # 5. Run Results Analyzer & Plot Generation on temporary readme
    test_readme = tmp_path / "README.md"
    test_readme.write_text("<!-- RESULTS:START -->\n<!-- RESULTS:END -->\n", encoding="utf-8")
    analyze(
        results_dir_path=repo_root / "results",
        readme_path=test_readme,
    )
    assert "## Automated Research Results Summary" in test_readme.read_text(encoding="utf-8")

    # Clean up ephemeral prediction file and test row from results.csv
    pred_path = Path(eval_out["predictions_file"])
    if pred_path.is_file():
        pred_path.unlink()

    csv_path = repo_root / "results" / "results.csv"
    if csv_path.is_file():
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        filtered = [line for line in lines if "cpu_smoke_test_run" not in line]
        csv_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
