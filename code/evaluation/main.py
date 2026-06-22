"""
Evaluation pipeline — runs the claim analyzer on dataset/sample_claims.csv
and compares predictions against expected outputs.

Usage:
    python code/evaluation/main.py [--dataset-dir dataset]
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from main import REPO_ROOT, OUTPUT_COLUMNS, process_claims

EXACT_COLUMNS = [
    "claim_status", "issue_type", "object_part",
    "severity", "evidence_standard_met", "valid_image",
]


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flags_match(pred: str, gold: str) -> bool:
    pred_set = {f.strip() for f in pred.split(";") if f.strip()}
    gold_set = {f.strip() for f in gold.split(";") if f.strip()}
    return pred_set == gold_set


def evaluate(predictions: list[dict], ground_truth: list[dict]) -> dict:
    n = min(len(predictions), len(ground_truth))
    if len(predictions) != len(ground_truth):
        print(f"Warning: {len(predictions)} predictions vs {len(ground_truth)} ground truth rows")

    metrics = {col: {"correct": 0, "total": n} for col in EXACT_COLUMNS + ["risk_flags"]}
    per_row = []

    for pred, gold in zip(predictions[:n], ground_truth[:n]):
        row_result = {"user_id": gold.get("user_id", "?")}

        for col in EXACT_COLUMNS:
            p = pred.get(col, "").strip().lower()
            g = gold.get(col, "").strip().lower()
            match = p == g
            metrics[col]["correct"] += int(match)
            row_result[col] = {"pred": p, "gold": g, "match": match}

        p_flags = pred.get("risk_flags", "none").strip().lower()
        g_flags = gold.get("risk_flags", "none").strip().lower()
        flag_ok = flags_match(p_flags, g_flags)
        metrics["risk_flags"]["correct"] += int(flag_ok)
        row_result["risk_flags"] = {"pred": p_flags, "gold": g_flags, "match": flag_ok}

        per_row.append(row_result)

    return {"metrics": metrics, "per_row": per_row, "n": n}


def print_report(result: dict):
    metrics = result["metrics"]
    n = result["n"]

    print(f"\n=== Evaluation Report ({n} rows) ===")
    print(f"{'Field':<30} {'Correct':>7} {'Total':>7} {'Accuracy':>9}")
    print("-" * 57)

    total_correct = total_items = 0
    for col, m in metrics.items():
        acc = m["correct"] / m["total"] if m["total"] else 0
        print(f"{col:<30} {m['correct']:>7} {m['total']:>7} {acc:>8.1%}")
        total_correct += m["correct"]
        total_items += m["total"]

    overall = total_correct / total_items if total_items else 0
    print("-" * 57)
    print(f"{'Overall (all fields)':<30} {total_correct:>7} {total_items:>7} {overall:>8.1%}")

    print("\n=== Per-Row Details ===")
    for row in result["per_row"]:
        mismatches = [col for col, v in row.items() if col != "user_id" and not v["match"]]
        status = "OK" if not mismatches else f"MISMATCH: {', '.join(mismatches)}"
        print(f"  {row['user_id']}: {status}")
        for col in mismatches:
            v = row[col]
            print(f"    {col}: pred={v['pred']!r}  gold={v['gold']!r}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate on sample_claims.csv")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--delay", type=float, default=4.0)
    args = parser.parse_args()

    dataset_dir = REPO_ROOT / args.dataset_dir
    sample_path = dataset_dir / "sample_claims.csv"
    output_path = REPO_ROOT / "evaluation" / "eval_output.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Running evaluation on {sample_path}...")
    process_claims(sample_path, dataset_dir, output_path, delay=args.delay)

    ground_truth = load_csv(sample_path)
    predictions = load_csv(output_path)

    result = evaluate(predictions, ground_truth)
    print_report(result)


if __name__ == "__main__":
    main()
