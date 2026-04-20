#!/usr/bin/env python3
"""Create a class-balanced YOLO training split with targeted augmentation."""

from __future__ import annotations

import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageEnhance

SRC = Path(os.environ.get("STREETBRIDGE_BALANCE_SRC", "StreetBridge/datasets/yolo_homeless_4class"))
OUT = Path(os.environ.get("STREETBRIDGE_BALANCE_OUT", "StreetBridge/datasets/yolo_homeless_4class_balanced"))
SEED = 42
CLASS_NAMES = [
    "homeless_tent",
    "homeless_cart",
    "homeless_person",
]
TARGET_CLASS_IDS = [1, 2]  # homeless_cart, homeless_person
AUGMENTATION_NAMES = [
    "hflip",
    "color",
    "light",
]


def read_labels(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        rows.append([float(p) for p in parts])
    return rows


def write_labels(path: Path, rows: list[list[float]]) -> None:
    lines = []
    for r in rows:
        cls = int(r[0])
        lines.append(f"{cls} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def has_class(rows: list[list[float]], cls_id: int) -> bool:
    return any(int(r[0]) == cls_id for r in rows)


def present_class_ids(rows: list[list[float]]) -> set[int]:
    return {int(r[0]) for r in rows}


def hflip_rows(rows: list[list[float]]) -> list[list[float]]:
    flipped = []
    for r in rows:
        cls, x, y, w, h = r
        flipped.append([cls, 1.0 - x, y, w, h])
    return flipped


def ensure_clean_copy() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(SRC, OUT)


def resolve_image_for_label(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png"):
        candidate = img_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def augment_train_split() -> dict[str, int]:
    random.seed(SEED)
    img_dir = OUT / "images" / "train"
    lbl_dir = OUT / "labels" / "train"

    label_files = sorted(lbl_dir.glob("*.txt"))
    target_files: list[tuple[Path, set[int]]] = []
    for lf in label_files:
        rows = read_labels(lf)
        class_ids = present_class_ids(rows) & set(TARGET_CLASS_IDS)
        if class_ids:
            target_files.append((lf, class_ids))

    generated_counts = Counter()
    for lf, class_ids in target_files:
        stem = lf.stem
        src_img = resolve_image_for_label(img_dir, stem)
        if src_img is None:
            continue

        rows = read_labels(lf)
        target_suffix = "_".join(CLASS_NAMES[cls_id] for cls_id in sorted(class_ids))
        with Image.open(src_img) as img:
            img = img.convert("RGB")

            # Aug 1: horizontal flip (geometric, bbox adjusted)
            out1 = img.transpose(Image.FLIP_LEFT_RIGHT)
            out1_name = f"{stem}__aug_{target_suffix}_hflip.jpg"
            out1.save(img_dir / out1_name, format="JPEG", quality=92)
            write_labels(lbl_dir / f"{Path(out1_name).stem}.txt", hflip_rows(rows))
            generated_counts["augmented_images"] += 1

            # Aug 2: contrast/color jitter (same bbox)
            c = random.uniform(0.85, 1.25)
            s = random.uniform(0.85, 1.25)
            out2 = ImageEnhance.Contrast(img).enhance(c)
            out2 = ImageEnhance.Color(out2).enhance(s)
            out2_name = f"{stem}__aug_{target_suffix}_color.jpg"
            out2.save(img_dir / out2_name, format="JPEG", quality=92)
            write_labels(lbl_dir / f"{Path(out2_name).stem}.txt", rows)
            generated_counts["augmented_images"] += 1

            # Aug 3: brightness + slight sharpness (same bbox)
            b = random.uniform(0.85, 1.20)
            sh = random.uniform(0.85, 1.25)
            out3 = ImageEnhance.Brightness(img).enhance(b)
            out3 = ImageEnhance.Sharpness(out3).enhance(sh)
            out3_name = f"{stem}__aug_{target_suffix}_light.jpg"
            out3.save(img_dir / out3_name, format="JPEG", quality=92)
            write_labels(lbl_dir / f"{Path(out3_name).stem}.txt", rows)
            generated_counts["augmented_images"] += 1
            generated_counts["source_images_augmented"] += 1
            for cls_id in class_ids:
                generated_counts[f"source_images_with_{CLASS_NAMES[cls_id]}"] += 1

    # refresh data.yaml path
    data_yaml = OUT / "data.yaml"
    text = data_yaml.read_text(encoding="utf-8")
    new_text = text.splitlines()
    new_text[0] = f"path: {OUT.resolve()}"
    data_yaml.write_text("\n".join(new_text) + "\n", encoding="utf-8")

    return dict(generated_counts)


def collect_dataset_summary() -> dict:
    images_root = OUT / "images"
    labels_root = OUT / "labels"

    split_image_counts: dict[str, int] = {}
    split_box_counts: dict[str, dict[str, int]] = {}
    total_box_counts: Counter[str] = Counter()

    for split in ["train", "val", "test"]:
        image_files = sorted(
            p for p in (images_root / split).glob("*") if p.is_file()
        )
        split_image_counts[split] = len(image_files)

        class_counts: Counter[str] = Counter()
        for lf in sorted((labels_root / split).glob("*.txt")):
            rows = read_labels(lf)
            for row in rows:
                cls_id = int(row[0])
                class_name = CLASS_NAMES[cls_id]
                class_counts[class_name] += 1
                total_box_counts[class_name] += 1

        split_box_counts[split] = {
            class_name: class_counts.get(class_name, 0)
            for class_name in CLASS_NAMES
        }

    train_augmented_images = len(
        [p for p in (images_root / "train").glob("*") if "__aug_" in p.stem]
    )

    return {
        "source_dataset": str(SRC.resolve()),
        "balanced_dataset": str(OUT.resolve()),
        "class_names": CLASS_NAMES,
        "target_class_ids": TARGET_CLASS_IDS,
        "target_class_names": [CLASS_NAMES[cls_id] for cls_id in TARGET_CLASS_IDS],
        "seed": SEED,
        "augmentation_variants": AUGMENTATION_NAMES,
        "split_image_counts": split_image_counts,
        "split_box_counts": split_box_counts,
        "total_box_counts": {
            class_name: total_box_counts.get(class_name, 0)
            for class_name in CLASS_NAMES
        },
        "train_augmented_images": train_augmented_images,
    }


def main() -> None:
    ensure_clean_copy()
    augmentation_stats = augment_train_split()
    summary = collect_dataset_summary()
    summary["augmentation_stats"] = augmentation_stats
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Balanced dataset ready: {OUT}")
    print(f"Summary written: {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
