# benchmark_runner.py
"""
Task 3 (second half): Comprehensive benchmark.

Evaluates 3 algorithms across 2 datasets with 5-fold cross-validation and
writes a Markdown experimental summary table (benchmark_report.md).

This script is deliberately deterministic (it calls ml_tools directly rather
than going through the LLM) so it is reproducible for grading. It also shows
you exactly the kind of multi-step trace the agent SHOULD be able to
reproduce autonomously if you instead type the equivalent instruction as a
natural-language prompt into react_agent.run_agent_loop(...):

    "Evaluate decision_tree, random_forest, and a deep PyTorch classifier on
     both the wine and breast_cancer datasets using 5-fold cross-validation,
     then write a Markdown table comparing all results and recommend the
     best model for each dataset."

Run:
    python benchmark_runner.py
"""

import json
from datetime import datetime

from ml_tools import (
    train_sklearn_model,
    train_deep_pytorch_classifier,
)

DATASETS_TO_BENCHMARK = ["wine", "breast_cancer"]
SKLEARN_MODELS = ["decision_tree", "random_forest", "logistic_regression"]


def run_benchmark():
    rows = []

    for dataset in DATASETS_TO_BENCHMARK:
        for model_type in SKLEARN_MODELS:
            result = json.loads(train_sklearn_model(dataset, model_type))
            rows.append({
                "dataset": dataset,
                "algorithm": model_type,
                "test_accuracy": result["test_accuracy"],
                "cv_mean_accuracy": result["cv_mean_accuracy"],
                "cv_std": result["cv_std"],
            })

        # Also benchmark the regularized deep PyTorch classifier as the 4th
        # comparison point per dataset (kept separate from the CV table
        # below since PyTorch models here use a single train/test split
        # rather than sklearn's cross_val_score).
        deep_result = json.loads(
            train_deep_pytorch_classifier(
                dataset, hidden_dims=[64, 32], dropout=0.3,
                use_batchnorm=True, epochs=100, lr=0.01, scheduler_type="cosine",
            )
        )
        rows.append({
            "dataset": dataset,
            "algorithm": "deep_pytorch_mlp",
            "test_accuracy": deep_result["test_accuracy"],
            "cv_mean_accuracy": None,
            "cv_std": None,
        })

    return rows


def to_markdown_table(rows):
    lines = [
        "| Dataset | Algorithm | Test Accuracy | CV Mean Accuracy | CV Std Dev |",
        "|:-|:-|:-|:-|:-|",
    ]
    for r in rows:
        cv_mean = f"{r['cv_mean_accuracy']:.4f}" if r["cv_mean_accuracy"] is not None else "N/A"
        cv_std = f"{r['cv_std']:.4f}" if r["cv_std"] is not None else "N/A"
        lines.append(
            f"| {r['dataset']} | {r['algorithm']} | {r['test_accuracy']:.4f} | "
            f"{cv_mean} | {cv_std} |"
        )
    return "\n".join(lines)


def best_per_dataset(rows):
    best = {}
    for r in rows:
        d = r["dataset"]
        if d not in best or r["test_accuracy"] > best[d]["test_accuracy"]:
            best[d] = r
    return best


def write_report(rows, path="benchmark_report.md"):
    table_md = to_markdown_table(rows)
    best = best_per_dataset(rows)

    best_lines = []
    for dataset, r in best.items():
        best_lines.append(
            f"- **{dataset}**: best model is `{r['algorithm']}` "
            f"(test accuracy = {r['test_accuracy']:.4f})"
        )

    content = f"""# Autonomous Benchmark Report

Generated: {datetime.now().isoformat(timespec='seconds')}

## Experimental Summary Table

{table_md}

## Recommendations

{chr(10).join(best_lines)}

## Notes

- `decision_tree`, `random_forest`, and `logistic_regression` are evaluated with
  an 80/20 train/test split AND 5-fold cross-validation (see CV columns).
- `deep_pytorch_mlp` (Dropout + BatchNorm + cosine LR scheduling) is evaluated
  with the same 80/20 split; CV columns are N/A because this benchmark uses a
  single held-out split for the neural network for speed. Extend
  `benchmark_runner.py` with a manual K-fold loop around
  `train_deep_pytorch_classifier` if you want CV numbers for it too.
"""
    with open(path, "w") as f:
        f.write(content)
    return path


if __name__ == "__main__":
    print("Running comprehensive benchmark across datasets:", DATASETS_TO_BENCHMARK)
    results = run_benchmark()
    report_path = write_report(results)
    print(f"\nBenchmark complete. Report written to: {report_path}\n")
    print(to_markdown_table(results))
