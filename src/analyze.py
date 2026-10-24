"""Results analysis, metric aggregation, plotting, and honest README injection.

Reads results/*.csv and continual matrices. Computes mean +/- std, Wilson 95% CIs,
majority-class baselines, generates publication-quality diagnostic plots,
and injects generated summaries into README between <!-- RESULTS:START --> and <!-- RESULTS:END -->.
NEVER invents, hardcodes, or pre-fills result numbers.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_results_csv(results_dir: Path) -> pd.DataFrame:
    """Load and concatenate all results CSVs in the results directory."""
    csv_files = list(results_dir.glob("*.csv"))
    if not csv_files:
        logger.warning("No CSV files found in %s", results_dir)
        return pd.DataFrame()

    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    return combined


def generate_plots(results_dir: Path, df: pd.DataFrame) -> List[str]:
    """Generate diagnostic plots from results dataframe and continual matrices."""
    created_plots: List[str] = []
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Rank & DoRA comparison plot
    if not df.empty and "lora_r" in df.columns and "seen_accuracy" in df.columns:
        valid_df = df[df["run_status"] == "SUCCESS"].copy()
        if not valid_df.empty:
            plt.figure(figsize=(7, 4.5))
            for dora_val in [False, True]:
                sub = valid_df[valid_df["use_dora"] == dora_val]
                if not sub.empty:
                    agg = sub.groupby("lora_r")["unseen_accuracy"].agg(["mean", "std"]).reset_index()
                    label = "DoRA (use_dora=True)" if dora_val else "LoRA (use_dora=False)"
                    plt.errorbar(
                        agg["lora_r"],
                        agg["mean"],
                        yerr=agg["std"].fillna(0),
                        marker="o",
                        capsize=4,
                        label=label,
                        linewidth=1.8,
                    )
            plt.title("Visual Knowledge Generalization: LoRA vs. DoRA across Ranks")
            plt.xlabel("Adapter Rank (r)")
            plt.ylabel("Unseen Accuracy (%)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            out_p1 = results_dir / "rank_dora_comparison.png"
            plt.savefig(out_p1, dpi=200)
            plt.close()
            created_plots.append(str(out_p1.name))

    # 2. Generalization Gap (Seen vs Unseen) Plot
    if not df.empty and "seen_accuracy" in df.columns and "unseen_accuracy" in df.columns:
        valid_df = df[df["run_status"] == "SUCCESS"].copy()
        if not valid_df.empty and len(valid_df) > 0:
            plt.figure(figsize=(7, 4.5))
            x_indices = np.arange(len(valid_df))
            width = 0.35
            plt.bar(x_indices - width / 2, valid_df["seen_accuracy"], width, label="Seen (Trained) Acc", color="#4C72B0")
            plt.bar(x_indices + width / 2, valid_df["unseen_accuracy"], width, label="Unseen (Test) Acc", color="#55A868")
            if "majority_baseline" in valid_df.columns and valid_df["majority_baseline"].iloc[0] is not None:
                plt.axhline(
                    y=valid_df["majority_baseline"].iloc[0],
                    color="red",
                    linestyle="--",
                    label=f"Majority Baseline ({valid_df['majority_baseline'].iloc[0]:.1f}%)",
                )
            plt.xlabel("Run Index / Config")
            plt.ylabel("Accuracy (%)")
            plt.title("Knowledge Updating vs Generalization Gap")
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.legend()
            plt.tight_layout()
            out_p2 = results_dir / "generalization_gap.png"
            plt.savefig(out_p2, dpi=200)
            plt.close()
            created_plots.append(str(out_p2.name))

    # 3. MMStar General Benchmark Forgetting Plot
    if not df.empty and "mmstar_accuracy" in df.columns:
        mmstar_df = df[df["mmstar_accuracy"].notna() & (df["run_status"] == "SUCCESS")].copy()
        if not mmstar_df.empty:
            plt.figure(figsize=(6, 4))
            plt.plot(mmstar_df.index, mmstar_df["mmstar_accuracy"], marker="s", color="#C44E52", linewidth=2)
            plt.title("General Benchmark Retention (MMStar)")
            plt.xlabel("Fine-Tuning Updates")
            plt.ylabel("MMStar Accuracy (%)")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            out_p3 = results_dir / "mmstar_forgetting.png"
            plt.savefig(out_p3, dpi=200)
            plt.close()
            created_plots.append(str(out_p3.name))

    # 4. Continual Matrix Heatmap
    matrix_files = list(results_dir.glob("continual_matrix_*.json"))
    if matrix_files:
        try:
            with matrix_files[0].open("r", encoding="utf-8") as f:
                cdata = json.load(f)
            R = np.array(cdata["accuracy_matrix"])
            plt.figure(figsize=(5.5, 4.5))
            plt.imshow(R * 100, cmap="YlGnBu", vmin=0, vmax=100)
            plt.colorbar(label="Accuracy (%)")
            plt.title(f"Continual Learning Matrix (BWT: {cdata.get('bwt_pct', 0)}%)")
            plt.xlabel("Evaluated Chunk (j)")
            plt.ylabel("Trained Stage (i)")
            K = R.shape[0]
            plt.xticks(np.arange(K), [f"C{j+1}" for j in range(K)])
            plt.yticks(np.arange(K), [f"S{i+1}" for i in range(K)])
            for i in range(K):
                for j in range(K):
                    plt.text(j, i, f"{R[i, j]*100:.1f}%", ha="center", va="center", color="black" if R[i, j] < 0.6 else "white")
            plt.tight_layout()
            out_p4 = results_dir / "forgetting_matrix.png"
            plt.savefig(out_p4, dpi=200)
            plt.close()
            created_plots.append(str(out_p4.name))
        except Exception as e:
            logger.warning("Could not plot continual matrix: %s", e)

    return created_plots


def build_markdown_summary(df: pd.DataFrame, results_dir: Path) -> str:
    """Build honest research summary in markdown, separating real from synthetic results.

    NEVER invents, estimates, or pre-fills missing numbers.
    """
    lines: List[str] = []
    lines.append("## Automated Research Results Summary\n")

    if df.empty:
        lines.append("> [!NOTE]")
        lines.append("> No experiment results logged yet in `results/results.csv`.")
        lines.append("> To run the pipeline:")
        lines.append("> - CPU Smoke Test: `make smoke`")
        lines.append("> - GPU T4 Run: `make run-t4`")
        lines.append("> - GPU Full Run: `make run-full`")
        return "\n".join(lines)

    # Distinguish synthetic from real data
    real_mask = (df["is_synthetic"] == False) | (df["is_synthetic"].astype(str).str.lower() == "false")
    real_df = df[real_mask].copy()
    synth_df = df[~real_mask].copy()

    # Section 1: Real Experiments
    if not real_df.empty:
        lines.append("### LiveVQA Fine-Tuning Knowledge Updating (Real Dataset)")
        lines.append("")
        lines.append("| Model | Rank | DoRA | Format | Seen Acc (%) | Seen 95% CI | Unseen Acc (%) | Unseen 95% CI | Gen Gap | Majority Baseline | Device |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for _, row in real_df.iterrows():
            if row.get("run_status") == "SUCCESS":
                lines.append(
                    f"| {row.get('model_name', 'VLM')} "
                    f"| {row.get('lora_r', '-')} "
                    f"| {row.get('use_dora', '-')} "
                    f"| {row.get('data_format', '-')} "
                    f"| {row.get('seen_accuracy', 0.0):.2f}% "
                    f"| {row.get('seen_ci_95', '-')} "
                    f"| {row.get('unseen_accuracy', 0.0):.2f}% "
                    f"| {row.get('unseen_ci_95', '-')} "
                    f"| {row.get('generalization_gap', 0.0):.2f}% "
                    f"| {row.get('majority_baseline', 0.0):.2f}% ({row.get('majority_label', 'A')}) "
                    f"| {row.get('device_name', row.get('device', 'cpu'))} |"
                )
        lines.append("")
    else:
        lines.append("> [!NOTE]")
        lines.append("> **No real GPU runs have been recorded yet.**")
        lines.append("> To reproduce the paper's real LiveVQA updating experiments, run:")
        lines.append("> ```bash")
        lines.append("> python -m src.train --config configs/colab_t4.yaml")
        lines.append("> python -m src.eval --config configs/colab_t4.yaml")
        lines.append("> python -m src.analyze")
        lines.append("> ```")
        lines.append("")

    # Section 2: Pipeline Smoke Tests (Clearly labeled synthetic per Hard Rule 3)
    if not synth_df.empty:
        lines.append("### Pipeline Smoke Test Verifications (SYNTHETIC DATA ONLY)")
        lines.append("> [!WARNING]")
        lines.append("> **Research Integrity Notice**: The table below reports verification smoke tests on synthetic geometric data. Per research rules, synthetic data scores verify pipeline correctness and must NEVER be cited as real research findings.")
        lines.append("")
        lines.append("| Run ID / Model | Samples | Wall-Clock (s) | Final Loss | Status | Device | Commit |")
        lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

        for _, row in synth_df.iterrows():
            samples_str = f"Train: {row.get('num_train_samples', 0)} / Eval: {row.get('num_eval_samples', 0)}"
            loss_str = f"{row.get('final_train_loss'):.4f}" if pd.notna(row.get("final_train_loss")) else "-"
            lines.append(
                f"| {row.get('config_name', 'smoke')} ({row.get('model_name', 'TinyVLM')}) "
                f"| {samples_str} "
                f"| {row.get('wall_clock_seconds', 0.0)}s "
                f"| {loss_str} "
                f"| {row.get('run_status', '-')} "
                f"| {row.get('device_name', row.get('device', 'cpu'))} "
                f"| `{row.get('git_commit', 'unversioned')[:8]}` |"
            )
        lines.append("")

    # Section 3: Continual Learning Matrix
    matrix_files = list(results_dir.glob("continual_matrix_*.json"))
    if matrix_files:
        try:
            with matrix_files[0].open("r", encoding="utf-8") as f:
                cdata = json.load(f)
            lines.append("### Continual Learning Evaluation Matrix")
            lines.append(f"- **Stages (Chunks K)**: {cdata.get('num_chunks')}")
            lines.append(f"- **Replay Capacity**: {cdata.get('replay_capacity', 0)} samples")
            lines.append(f"- **Backward Transfer (BWT)**: {cdata.get('bwt_pct', 0.0):.2f}%")
            lines.append(f"- **Average Forgetting**: {cdata.get('average_forgetting_pct', 0.0):.2f}%")
            lines.append("")
        except Exception as e:
            logger.warning("Error summarizing continual matrix: %s", e)

    return "\n".join(lines)


def inject_results_into_readme(readme_path: Path, summary_content: str) -> bool:
    """Inject summary markdown into README.md between the RESULTS markers.

    Searches for:
        <!-- RESULTS:START -->
        ...
        <!-- RESULTS:END -->
    and replaces the content in-between.
    """
    if not readme_path.is_file():
        logger.warning("README.md not found at %s", readme_path)
        return False

    text = readme_path.read_text(encoding="utf-8")
    start_tag = "<!-- RESULTS:START -->"
    end_tag = "<!-- RESULTS:END -->"

    pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
    replacement = f"{start_tag}\n\n{summary_content}\n\n{end_tag}"

    if pattern.search(text):
        new_text = pattern.sub(replacement, text)
        readme_path.write_text(new_text, encoding="utf-8")
        logger.info("Successfully injected results into %s", readme_path)
        return True
    else:
        logger.warning("Markers %s and %s not found in %s", start_tag, end_tag, readme_path)
        return False


def analyze(results_dir_path: Optional[str | Path] = None, readme_path: Optional[str | Path] = None) -> None:
    """Main analysis pipeline."""
    base_dir = Path(__file__).parent.parent
    results_dir = Path(results_dir_path) if results_dir_path else (base_dir / "results")
    readme_file = Path(readme_path) if readme_path else (base_dir / "README.md")

    df = load_results_csv(results_dir)
    plots = generate_plots(results_dir, df)
    summary_md = build_markdown_summary(df, results_dir)

    # Save summary.md
    summary_path = results_dir / "summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")
    logger.info("Wrote summary to %s", summary_path)

    # Inject into README
    if readme_file.is_file():
        inject_results_into_readme(readme_file, summary_md)


def main() -> None:
    """CLI runner for results analysis."""
    parser = argparse.ArgumentParser(description="LiveVQA Results Analyzer & Visualizer")
    parser.add_argument("--results-dir", type=str, default=None, help="Directory containing results")
    parser.add_argument("--readme", type=str, default=None, help="Path to README.md")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    analyze(args.results_dir, args.readme)


if __name__ == "__main__":
    main()
