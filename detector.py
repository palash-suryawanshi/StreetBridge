"""
detector.py - YOLOv8 object detection for homelessness indicators.

This module prefers a locally trained custom model:
    0: homeless_tent
    1: homeless_cart
    2: homeless_person

If those weights are not available, it falls back to the default YOLOv8 COCO
model and only exposes the supported proxy class:
    0: person   -> homeless_person
"""

import io
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
RUNS_ROOT = PROJECT_ROOT / "runs" / "detect"
MODEL_OVERRIDE_ENV = "STREETBRIDGE_MODEL_PATH"
DEFAULT_FALLBACK_MODEL = "yolov8n.pt"

# Prefer the strongest known project runs first.
MODEL_CANDIDATE_DIRS = [
    "homeless4_accuracy_v5_finetune",
    "homeless4_accuracy_v3",
    "homeless4_accuracy_v2",
    "homeless4_balanced_v1",
    "homeless4_baseline",
]

COCO_FALLBACK_CLASS_MAP = {
    0: "homeless_person",
}

ALL_TARGET_CLASSES = {
    "homeless_tent",
    "homeless_cart",
    "homeless_person",
}


def _resolve_candidate_run_dir(name: str) -> Path:
    direct = RUNS_ROOT / name / "weights" / "best.pt"
    nested = RUNS_ROOT / "runs" / "detect" / name / "weights" / "best.pt"
    if direct.exists():
        return direct
    return nested


def resolve_default_model_path() -> tuple[Optional[str], Optional[str]]:
    """
    Resolve the default custom model path and a short reason string.

    Selection order:
      1. Explicit STREETBRIDGE_MODEL_PATH override, if it exists
      2. Preferred known run names, strongest-first
      3. No custom weights found -> fallback to COCO model
    """
    override = os.environ.get(MODEL_OVERRIDE_ENV)
    if override:
        override_path = Path(override).expanduser()
        if override_path.exists():
            return str(override_path), f"override:{MODEL_OVERRIDE_ENV}"
        return None, f"missing_override:{override_path}"

    for run_name in MODEL_CANDIDATE_DIRS:
        candidate = _resolve_candidate_run_dir(run_name)
        if candidate.exists():
            return str(candidate), f"run:{run_name}"

    return None, None


CUSTOM_WEIGHTS_PATH, CUSTOM_WEIGHTS_SOURCE = resolve_default_model_path()


class HomelessnessDetector:
    """
    Wraps a YOLOv8 model and translates detections into homelessness-indicator
    annotations.
    """

    def __init__(self, conf_threshold: float = 0.35, model_path: Optional[str] = None):
        """
        Args:
            conf_threshold: Minimum confidence score to keep a detection.
            model_path: Path to .pt weights. Overrides CUSTOM_WEIGHTS_PATH.
        """
        self.conf_threshold = conf_threshold
        self._model = None
        self._custom_model_path = Path(CUSTOM_WEIGHTS_PATH) if CUSTOM_WEIGHTS_PATH else None
        requested_path = Path(model_path).expanduser() if model_path else self._custom_model_path
        if model_path and not requested_path.exists():
            raise FileNotFoundError(f"Requested model weights not found: {requested_path}")
        self._uses_custom_model = bool(
            requested_path
            and requested_path.exists()
            and requested_path.name != DEFAULT_FALLBACK_MODEL
        )
        self._model_path = str(requested_path) if requested_path else DEFAULT_FALLBACK_MODEL
        self._class_map: dict[int, str] = {}
        self.model_description = (
            f"custom:{Path(self._model_path).name}"
            if self._uses_custom_model
            else f"fallback:{self._model_path}"
        )
        self._load_model()

    def _load_model(self):
        """Lazy-load the YOLO model (downloads weights on first run if needed)."""
        try:
            from ultralytics import YOLO
            self._model = YOLO(self._model_path)
            self._class_map = (
                self._custom_model_class_map()
                if self._uses_custom_model
                else COCO_FALLBACK_CLASS_MAP
            )
        except ImportError:
            raise ImportError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model: {e}")

    def _custom_model_class_map(self) -> dict[int, str]:
        names = getattr(self._model, "names", {})
        return {
            int(cls_id): str(label)
            for cls_id, label in names.items()
            if str(label) in ALL_TARGET_CLASSES
        }

    def detect(self, image_bytes: bytes) -> list[dict]:
        """
        Run detection on a single image supplied as raw bytes.

        Args:
            image_bytes: JPEG/PNG image data.

        Returns:
            List of detection dicts, each with keys:
                label       (str)   : target class name
                confidence  (float) : 0.0 – 1.0
                bbox        (list)  : [x1, y1, x2, y2] pixel coordinates
                coco_class  (str)   : original COCO class name
                source      (str)   : 'auto'
        """
        if self._model is None:
            return []

        # Decode image bytes to a PIL image then to numpy array
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(pil_img)

        # Run inference
        results = self._model(img_array, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                coco_cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Only keep classes supported by the active model mapping
                if coco_cls_id not in self._class_map:
                    continue

                target_label = self._class_map[coco_cls_id]
                coco_name = result.names.get(coco_cls_id, str(coco_cls_id))
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append(
                    {
                        "label": target_label,
                        "confidence": round(conf, 4),
                        "bbox": [round(x1), round(y1), round(x2), round(y2)],
                        "coco_class": coco_name,
                        "source": "auto",
                        "note": self._coverage_note(target_label),
                    }
                )

        return detections

    @staticmethod
    def _coverage_note(label: str) -> str:
        """Return a warning note for classes that need a custom model."""
        if label in {"homeless_tent", "homeless_cart"}:
            return "⚠️ Requires custom model for reliable detection"
        return ""

    @property
    def missing_classes(self) -> list[str]:
        """Return the list of target classes not covered by the current model."""
        if self._uses_custom_model:
            return []
        return sorted(ALL_TARGET_CLASSES - set(self._class_map.values()))
