"""
exporter.py - CSV and ZIP export logic for annotation results.
"""

import io
import zipfile
import csv
from typing import Optional

import pandas as pd


def export_csv(
    fetched_images: list[dict],
    annotations: dict[str, list[dict]],
) -> bytes:
    """
    Build a CSV file containing all annotation data.

    Each row represents one detection (or one unannotated image if no detections).
    Columns: filename, lat, lng, label, confidence, bbox_x1, bbox_y1, bbox_x2,
             bbox_y2, source, note.

    Args:
        fetched_images: List of image info dicts from session state.
        annotations: Dict mapping filename -> list of annotation dicts.

    Returns:
        CSV content as UTF-8 encoded bytes.
    """
    rows = []

    for item in fetched_images:
        fname = item["filename"]
        lat = item["lat"]
        lng = item["lng"]
        anns = annotations.get(fname, [])

        if anns:
            for ann in anns:
                bbox = ann.get("bbox") or [None, None, None, None]
                rows.append(
                    {
                        "filename": fname,
                        "latitude": lat,
                        "longitude": lng,
                        "label": ann.get("label", ""),
                        "confidence": ann.get("confidence", ""),
                        "bbox_x1": bbox[0] if bbox else None,
                        "bbox_y1": bbox[1] if bbox else None,
                        "bbox_x2": bbox[2] if bbox else None,
                        "bbox_y2": bbox[3] if bbox else None,
                        "source": ann.get("source", "auto"),
                        "coco_class": ann.get("coco_class", ""),
                        "note": ann.get("note", ""),
                    }
                )
        else:
            # Include unannotated images so all fetched points appear in the CSV
            rows.append(
                {
                    "filename": fname,
                    "latitude": lat,
                    "longitude": lng,
                    "label": None,
                    "confidence": None,
                    "bbox_x1": None,
                    "bbox_y1": None,
                    "bbox_x2": None,
                    "bbox_y2": None,
                    "source": None,
                    "coco_class": None,
                    "note": None,
                }
            )

    df = pd.DataFrame(rows)

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def export_zip(
    fetched_images: list[dict],
    annotations: dict[str, list[dict]],
) -> bytes:
    """
    Build a ZIP archive containing:
      - All annotated images (JPEG, or originals if not yet annotated).
      - annotations.csv with all detection data.

    Args:
        fetched_images: List of image info dicts from session state.
        annotations: Dict mapping filename -> list of annotation dicts.

    Returns:
        ZIP file content as bytes.
    """
    zip_buf = io.BytesIO()

    with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:

        # Write images
        for item in fetched_images:
            fname = item["filename"]
            # Prefer annotated version if available
            img_bytes = item.get("annotated_bytes") or item.get("image_bytes")
            if img_bytes:
                zf.writestr(f"images/{fname}", img_bytes)

        # Write annotations CSV
        csv_bytes = export_csv(fetched_images, annotations)
        zf.writestr("annotations.csv", csv_bytes)

    zip_buf.seek(0)
    return zip_buf.read()
