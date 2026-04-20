#!/usr/bin/env python3
"""Convert VIA CSV and COCO JSON annotations into YOLO dataset structure."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image

# Mapping from source CSV labels to target 3-class schema.
# "Encampments" is retained as the shelter proxy for the tent class because it
# is the closest labeled concept available in the source data.
BASE_CLASS_MAP = {
    "Encampments": "homeless_tent",
    "Homeless Cart": "homeless_cart",
    "People": "homeless_person",
}

AMBIGUOUS_PERSON_LABELS = {
    "Homeless",
    "Homelessss",
}

CLASS_NAMES = [
    "homeless_tent",
    "homeless_cart",
    "homeless_person",
]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float


@dataclass
class ImageRecord:
    src_image: Path
    out_name: str
    boxes: List[Box]
    group_id: str


def parse_jsonish(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}


def is_coco_dataset(payload: object) -> bool:
    return (
        isinstance(payload, dict)
        and isinstance(payload.get("images"), list)
        and isinstance(payload.get("annotations"), list)
        and isinstance(payload.get("categories"), list)
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


def to_yolo_box(shape: dict, img_w: int, img_h: int) -> Tuple[float, float, float, float] | None:
    if shape.get("name") != "rect":
        return None

    x = float(shape.get("x", 0))
    y = float(shape.get("y", 0))
    w = float(shape.get("width", 0))
    h = float(shape.get("height", 0))

    if w <= 1 or h <= 1:
        return None

    x1 = clamp(x, 0.0, img_w - 1.0)
    y1 = clamp(y, 0.0, img_h - 1.0)
    x2 = clamp(x + w, 1.0, img_w)
    y2 = clamp(y + h, 1.0, img_h)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    wn = bw / img_w
    hn = bh / img_h

    return xc, yc, wn, hn


def coco_bbox_to_yolo_box(bbox: list[float], img_w: int, img_h: int) -> Tuple[float, float, float, float] | None:
    if len(bbox) != 4:
        return None

    x, y, w, h = [float(v) for v in bbox]
    if w <= 1 or h <= 1:
        return None

    x1 = clamp(x, 0.0, img_w - 1.0)
    y1 = clamp(y, 0.0, img_h - 1.0)
    x2 = clamp(x + w, 1.0, img_w)
    y2 = clamp(y + h, 1.0, img_h)

    bw = x2 - x1
    bh = y2 - y1
    if bw <= 1 or bh <= 1:
        return None

    xc = (x1 + x2) / 2.0 / img_w
    yc = (y1 + y2) / 2.0 / img_h
    wn = bw / img_w
    hn = bh / img_h

    return xc, yc, wn, hn


def collect_csv_records(src_root: Path, class_map: dict[str, str]) -> Tuple[List[ImageRecord], Counter, Counter]:
    grouped: Dict[Tuple[Path, str], List[Box]] = defaultdict(list)
    raw_counter: Counter = Counter()
    mapped_counter: Counter = Counter()

    for csv_path in sorted(src_root.rglob("*.csv")):
        folder = csv_path.parent
        folder_slug = folder.name.replace(" ", "_").replace("-", "_")

        with csv_path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = (row.get("filename") or "").strip()
                if not filename:
                    continue

                image_path = folder / filename
                if not image_path.exists():
                    continue

                shape = parse_jsonish(row.get("region_shape_attributes") or "{}")
                attrs = parse_jsonish(row.get("region_attributes") or "{}")
                src_label = (attrs.get("Homeless") or "").strip() if isinstance(attrs, dict) else ""
                if not src_label:
                    continue

                raw_counter[src_label] += 1
                mapped_label = class_map.get(src_label)
                if mapped_label is None:
                    continue

                with Image.open(image_path) as img:
                    img_w, img_h = img.size

                yolo = to_yolo_box(shape, img_w, img_h)
                if yolo is None:
                    continue

                class_id = CLASS_TO_ID[mapped_label]
                grouped[(image_path, folder_slug)].append(Box(class_id=class_id, x=yolo[0], y=yolo[1], w=yolo[2], h=yolo[3]))
                mapped_counter[mapped_label] += 1

    records: List[ImageRecord] = []
    for (img_path, folder_slug), boxes in grouped.items():
        out_name = f"{folder_slug}__{img_path.name}"
        group_id = derive_group_id(folder_slug, img_path)
        records.append(
            ImageRecord(
                src_image=img_path,
                out_name=out_name,
                boxes=boxes,
                group_id=group_id,
            )
        )

    return records, raw_counter, mapped_counter


def collect_coco_records(src_root: Path) -> Tuple[List[ImageRecord], Counter, Counter]:
    grouped: Dict[Tuple[Path, str], List[Box]] = defaultdict(list)
    raw_counter: Counter = Counter()
    mapped_counter: Counter = Counter()

    for json_path in sorted(src_root.rglob("*.json")):
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        if not is_coco_dataset(payload):
            continue

        folder = json_path.parent
        folder_slug = folder.relative_to(src_root).as_posix().replace("/", "_").replace(" ", "_").replace("-", "_")

        categories = payload["categories"]
        images = payload["images"]
        annotations = payload["annotations"]

        category_name_by_id = {}
        for category in categories:
            if not isinstance(category, dict):
                continue
            category_id = category.get("id")
            category_name = (category.get("name") or "").strip()
            if category_id is None or not category_name:
                continue
            category_name_by_id[category_id] = category_name

        image_meta_by_id = {}
        for image in images:
            if not isinstance(image, dict):
                continue
            image_id = image.get("id")
            file_name = (image.get("file_name") or "").strip()
            if image_id is None or not file_name:
                continue
            image_meta_by_id[image_id] = image

        for ann in annotations:
            if not isinstance(ann, dict):
                continue

            image_id = ann.get("image_id")
            category_id = ann.get("category_id")
            bbox = ann.get("bbox")
            image_meta = image_meta_by_id.get(image_id)
            category_name = category_name_by_id.get(category_id, "")

            if image_meta is None or not category_name or not isinstance(bbox, list):
                continue

            raw_counter[category_name] += 1

            mapped_label = category_name if category_name in CLASS_TO_ID else None
            if mapped_label is None:
                continue

            image_path = folder / image_meta["file_name"]
            if not image_path.exists():
                continue

            img_w = int(image_meta.get("width") or 0)
            img_h = int(image_meta.get("height") or 0)
            if img_w <= 0 or img_h <= 0:
                with Image.open(image_path) as img:
                    img_w, img_h = img.size

            yolo = coco_bbox_to_yolo_box(bbox, img_w, img_h)
            if yolo is None:
                continue

            class_id = CLASS_TO_ID[mapped_label]
            grouped[(image_path, folder_slug)].append(Box(class_id=class_id, x=yolo[0], y=yolo[1], w=yolo[2], h=yolo[3]))
            mapped_counter[mapped_label] += 1

    records: List[ImageRecord] = []
    for (img_path, folder_slug), boxes in grouped.items():
        out_name = f"{folder_slug}__{img_path.name}"
        group_id = derive_group_id(folder_slug, img_path)
        records.append(
            ImageRecord(
                src_image=img_path,
                out_name=out_name,
                boxes=boxes,
                group_id=group_id,
            )
        )

    return records, raw_counter, mapped_counter


def collect_records(src_root: Path, class_map: dict[str, str]) -> Tuple[List[ImageRecord], Counter, Counter]:
    csv_records, csv_raw_counter, csv_mapped_counter = collect_csv_records(src_root, class_map)
    coco_records, coco_raw_counter, coco_mapped_counter = collect_coco_records(src_root)

    records = csv_records + coco_records
    raw_counter = csv_raw_counter + coco_raw_counter
    mapped_counter = csv_mapped_counter + coco_mapped_counter
    return records, raw_counter, mapped_counter


def derive_group_id(folder_slug: str, image_path: Path) -> str:
    """
    Group together related Street View images so adjacent views from the same
    source panorama/location stay in the same split.

    Raw image names in this project follow a pattern like:
        2017_4_<panorama_or_location_id>_90.jpeg

    The trailing token is typically the camera heading (0/90/180/270). For
    honest evaluation, we drop that heading token and group by the remaining
    stem plus the source folder.
    """
    stem_parts = [
        part for part in image_path.stem.split("_")
        if part not in {"0", "90", "180", "270"} and not (part.startswith("h") and part[1:].isdigit())
    ]
    grouped_stem = "_".join(stem_parts) if stem_parts else image_path.stem
    return f"{folder_slug}__{grouped_stem}"


def split_records(records: List[ImageRecord], seed: int, train: float, val: float, test: float) -> Dict[str, List[ImageRecord]]:
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rng = random.Random(seed)
    grouped_records: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        grouped_records[record.group_id].append(record)

    groups = list(grouped_records.items())
    rng.shuffle(groups)

    total_records = len(records)
    train_target = total_records * train
    val_target = total_records * val

    splits: Dict[str, List[ImageRecord]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    split_sizes = {
        "train": 0,
        "val": 0,
        "test": 0,
    }

    for _, group_items in groups:
        group_size = len(group_items)

        if split_sizes["train"] + group_size <= train_target:
            split_name = "train"
        elif split_sizes["val"] + group_size <= val_target:
            split_name = "val"
        else:
            split_name = "test"

        splits[split_name].extend(group_items)
        split_sizes[split_name] += group_size

    return splits


def build_class_map(ambiguous_person_policy: str) -> dict[str, str]:
    """
    Build the label-remapping policy used during dataset preparation.

    Policies:
      - exclude: drop ambiguous person-only labels for cleaner supervision
      - person: include ambiguous labels as homeless_person
    """
    class_map = dict(BASE_CLASS_MAP)

    if ambiguous_person_policy == "person":
        for label in AMBIGUOUS_PERSON_LABELS:
            class_map[label] = "homeless_person"
    elif ambiguous_person_policy != "exclude":
        raise ValueError(
            f"Unsupported ambiguous person policy: {ambiguous_person_policy}"
        )

    return class_map


def write_yolo_dataset(output_root: Path, splits: Dict[str, List[ImageRecord]]) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)

    stats = {
        "images": {},
        "boxes": Counter(),
    }

    for split, items in splits.items():
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        stats["images"][split] = len(items)

        for rec in items:
            dst_image = img_dir / rec.out_name
            shutil.copy2(rec.src_image, dst_image)

            label_path = lbl_dir / f"{Path(rec.out_name).stem}.txt"
            with label_path.open("w", encoding="utf-8") as f:
                for b in rec.boxes:
                    f.write(f"{b.class_id} {b.x:.6f} {b.y:.6f} {b.w:.6f} {b.h:.6f}\n")
                    stats["boxes"][CLASS_NAMES[b.class_id]] += 1

    data_yaml = output_root / "data.yaml"
    yaml_lines = [
        f"path: {output_root.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_lines.append("")
    data_yaml.write_text(
        "\n".join(yaml_lines),
        encoding="utf-8",
    )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from VIA CSV and COCO JSON files")
    parser.add_argument("--src", default="StreetBridge/annotated_images", help="Source folder containing VIA CSV/COCO JSON annotations + images")
    parser.add_argument("--out", default="StreetBridge/datasets/yolo_homeless_4class", help="Output YOLO dataset root")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument(
        "--ambiguous-person-policy",
        choices=["exclude", "person"],
        default="exclude",
        help=(
            "How to handle ambiguous source labels like 'Homeless' and "
            "'Homelessss'. 'exclude' drops them for cleaner supervision; "
            "'person' remaps them to homeless_person."
        ),
    )
    args = parser.parse_args()

    src_root = Path(args.src)
    out_root = Path(args.out)

    class_map = build_class_map(args.ambiguous_person_policy)

    records, raw_counter, mapped_counter = collect_records(src_root, class_map)
    if not records:
        raise SystemExit("No valid labeled records found.")

    splits = split_records(records, seed=args.seed, train=args.train, val=args.val, test=args.test)
    stats = write_yolo_dataset(out_root, splits)

    included_source_labels = sorted(set(class_map.keys()) | set(CLASS_TO_ID.keys()))

    summary = {
        "records": len(records),
        "groups": len({record.group_id for record in records}),
        "raw_label_counts": dict(raw_counter),
        "mapped_label_counts": dict(mapped_counter),
        "ambiguous_person_policy": args.ambiguous_person_policy,
        "included_source_labels": included_source_labels,
        "excluded_source_labels": sorted(
            label for label in raw_counter.keys() if label not in included_source_labels
        ),
        "split_image_counts": stats["images"],
        "split_box_counts": dict(stats["boxes"]),
        "class_names": CLASS_NAMES,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Dataset prepared:", out_root)
    print("Images:", stats["images"])
    print("Boxes:")
    for name in CLASS_NAMES:
        print(f"  {name}: {stats['boxes'][name]}")
    print("data.yaml:", out_root / "data.yaml")


if __name__ == "__main__":
    main()
