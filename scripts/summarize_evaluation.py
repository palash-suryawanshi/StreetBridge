#!/usr/bin/env python3
"""Build a compact evaluation report from YOLO artifacts and dataset labels."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_class_names(dataset_root: Path) -> list[str]:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    class_names: dict[int, str] = {}
    in_names = False
    for raw_line in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        if line.strip() == "names:":
            in_names = True
            continue
        if not in_names:
            continue
        if not line.startswith("  "):
            break
        key, value = line.strip().split(":", 1)
        class_names[int(key.strip())] = value.strip()

    if not class_names:
        raise ValueError(f"No class names found in {data_yaml}")

    return [class_names[idx] for idx in sorted(class_names)]


def read_label_counts(dataset_root: Path, split: str, class_names: list[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    labels_dir = dataset_root / "labels" / split
    for label_file in sorted(labels_dir.glob("*.txt")):
        for line in label_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            cls_id = int(float(line.split()[0]))
            counts[class_names[cls_id]] += 1
    return {name: counts.get(name, 0) for name in class_names}


def read_image_count(dataset_root: Path, split: str) -> int:
    return len([p for p in (dataset_root / "images" / split).glob("*") if p.is_file()])


def read_training_results(results_csv: Path) -> dict:
    with results_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows found in {results_csv}")

    best = max(rows, key=lambda row: float(row.get("metrics/mAP50-95(B)", "0") or 0))
    last = rows[-1]
    return {
        "epochs_recorded": len(rows),
        "best_epoch": int(best["epoch"]),
        "best_precision": float(best["metrics/precision(B)"]),
        "best_recall": float(best["metrics/recall(B)"]),
        "best_map50": float(best["metrics/mAP50(B)"]),
        "best_map50_95": float(best["metrics/mAP50-95(B)"]),
        "last_epoch": int(last["epoch"]),
        "last_map50": float(last["metrics/mAP50(B)"]),
        "last_map50_95": float(last["metrics/mAP50-95(B)"]),
    }


def build_support_notes(label_counts: dict[str, int]) -> list[str]:
    notes: list[str] = []
    for class_name, count in label_counts.items():
        if count == 0:
            notes.append(f"{class_name}: no ground-truth boxes in this split")
        elif count < 10:
            notes.append(f"{class_name}: very low support ({count} boxes)")
        elif count < 25:
            notes.append(f"{class_name}: limited support ({count} boxes)")
    return notes


def build_report(
    dataset_root: Path,
    split: str,
    run_dir: Path,
    eval_dir: Path,
) -> dict:
    results = read_training_results(run_dir / "results.csv")
    class_names = read_class_names(dataset_root)
    label_counts = read_label_counts(dataset_root, split, class_names)
    image_count = read_image_count(dataset_root, split)
    support_notes = build_support_notes(label_counts)

    artifact_paths = {
        "confusion_matrix": str((eval_dir / "confusion_matrix.png").resolve()),
        "confusion_matrix_normalized": str(
            (eval_dir / "confusion_matrix_normalized.png").resolve()
        ),
        "pr_curve": str((eval_dir / "BoxPR_curve.png").resolve()),
        "precision_curve": str((eval_dir / "BoxP_curve.png").resolve()),
        "recall_curve": str((eval_dir / "BoxR_curve.png").resolve()),
        "f1_curve": str((eval_dir / "BoxF1_curve.png").resolve()),
    }
    sample_pairs = []
    for idx in range(3):
        label_img = eval_dir / f"{split}_batch{idx}_labels.jpg"
        pred_img = eval_dir / f"{split}_batch{idx}_pred.jpg"
        if label_img.exists() and pred_img.exists():
            sample_pairs.append(
                {
                    "labels": str(label_img.resolve()),
                    "predictions": str(pred_img.resolve()),
                }
            )

    return {
        "dataset_root": str(dataset_root.resolve()),
        "split": split,
        "run_dir": str(run_dir.resolve()),
        "eval_dir": str(eval_dir.resolve()),
        "image_count": image_count,
        "ground_truth_box_counts": label_counts,
        "support_notes": support_notes,
        "training_summary": results,
        "artifact_paths": artifact_paths,
        "sample_prediction_pairs": sample_pairs,
    }


def report_to_markdown(report: dict) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Dataset: `{report['dataset_root']}`",
        f"- Split: `{report['split']}`",
        f"- Images: {report['image_count']}",
        f"- Run directory: `{report['run_dir']}`",
        f"- Evaluation directory: `{report['eval_dir']}`",
        "",
        "## Training Summary",
        "",
        f"- Best epoch by mAP50-95: {report['training_summary']['best_epoch']}",
        f"- Best precision: {report['training_summary']['best_precision']:.4f}",
        f"- Best recall: {report['training_summary']['best_recall']:.4f}",
        f"- Best mAP50: {report['training_summary']['best_map50']:.4f}",
        f"- Best mAP50-95: {report['training_summary']['best_map50_95']:.4f}",
        "",
        "## Ground-Truth Support",
        "",
    ]

    for class_name, count in report["ground_truth_box_counts"].items():
        lines.append(f"- {class_name}: {count} boxes")

    lines.extend(["", "## Support Notes", ""])
    if report["support_notes"]:
        lines.extend(f"- {note}" for note in report["support_notes"])
    else:
        lines.append("- All classes have at least 25 ground-truth boxes in this split.")

    lines.extend(["", "## Key Artifacts", ""])
    for name, path in report["artifact_paths"].items():
        lines.append(f"- {name}: `{path}`")

    lines.extend(["", "## Visual Error Analysis", ""])
    if report["sample_prediction_pairs"]:
        for pair in report["sample_prediction_pairs"]:
            lines.append(f"- labels: `{pair['labels']}`")
            lines.append(f"- predictions: `{pair['predictions']}`")
    else:
        lines.append("- No paired prediction preview images were found.")

    lines.extend(
        [
            "",
            "## What To Check",
            "",
            "- Compare confusion matrices to see which classes are being confused.",
            "- Inspect PR, precision, recall, and F1 curves for threshold behavior.",
            "- Review the prediction preview images for repeated false positives and misses.",
            "- Pay special attention to low-support classes because their metrics will be noisy.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize YOLO evaluation artifacts")
    parser.add_argument("--dataset", required=True, help="Path to YOLO dataset root")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--run-dir", required=True, help="Training run directory")
    parser.add_argument("--eval-dir", required=True, help="YOLO val output directory")
    parser.add_argument("--out-json", required=True, help="Output JSON report path")
    parser.add_argument("--out-md", required=True, help="Output Markdown report path")
    args = parser.parse_args()

    report = build_report(
        dataset_root=Path(args.dataset),
        split=args.split,
        run_dir=Path(args.run_dir),
        eval_dir=Path(args.eval_dir),
    )
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path(args.out_md).write_text(report_to_markdown(report), encoding="utf-8")
    print(f"Wrote JSON report: {args.out_json}")
    print(f"Wrote Markdown report: {args.out_md}")


if __name__ == "__main__":
    main()
