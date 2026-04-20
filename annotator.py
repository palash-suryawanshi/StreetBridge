"""
annotator.py - Bounding box drawing utilities using Pillow.

Draws coloured bounding boxes and labels on Street View images for both
automated YOLO detections and manual user annotations.
"""

import io
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

# ── Colour palette per label ──────────────────────────────────────────────────
LABEL_COLOURS = {
    "homeless_person":   "#FF4B4B",   # red
    "homeless_bicycle":  "#4B9EFF",   # blue
    "homeless_tent":     "#FFB347",   # orange
    "homeless_cart":     "#A8FF4B",   # green
    "manual":            "#FF69B4",   # pink (fallback for unknown manual labels)
}

DEFAULT_COLOUR = "#FFFFFF"


def _hex_to_rgb(hex_colour: str) -> tuple[int, int, int]:
    """Convert a #RRGGBB hex string to an (R, G, B) tuple."""
    h = hex_colour.lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def _get_font(size: int = 14):
    """
    Load a font for label text. Falls back to the PIL default if no TTF is found.
    The PIL default is small but always available.
    """
    try:
        # Try common system font paths
        for path in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/Library/Fonts/Arial Bold.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]:
            try:
                return ImageFont.truetype(path, size)
            except (IOError, OSError):
                continue
        # Fallback: PIL built-in (no size control)
        return ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()


def draw_annotations(image_bytes: bytes, detections: list[dict]) -> bytes:
    """
    Draw bounding boxes and labels on an image for a list of detections.

    Each detection dict should have:
        label       (str)         : class name
        confidence  (float)       : 0.0 – 1.0
        bbox        (list | None) : [x1, y1, x2, y2] pixel coords
        source      (str)         : 'auto' or 'manual'

    Args:
        image_bytes: Raw JPEG/PNG image bytes.
        detections: List of detection dicts from the detector or manual input.

    Returns:
        JPEG bytes of the annotated image.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font_label = _get_font(14)
    font_small = _get_font(11)

    for det in detections:
        label = det.get("label", "unknown")
        conf = det.get("confidence", 1.0)
        bbox = det.get("bbox")           # May be None for manual annotations without bbox
        source = det.get("source", "auto")

        # Choose colour
        colour_hex = LABEL_COLOURS.get(label, LABEL_COLOURS.get("manual", DEFAULT_COLOUR))
        colour_rgb = _hex_to_rgb(colour_hex)

        # Build label text
        if source == "auto":
            label_text = f"{label} {conf:.0%}"
        else:
            label_text = f"[manual] {label}"

        if bbox:
            x1, y1, x2, y2 = bbox

            # Draw bounding box (3px border)
            for offset in range(3):
                draw.rectangle(
                    [x1 - offset, y1 - offset, x2 + offset, y2 + offset],
                    outline=colour_rgb,
                )

            # Draw label background
            try:
                text_bbox = draw.textbbox((x1, y1 - 18), label_text, font=font_label)
                bg_box = [
                    text_bbox[0] - 2,
                    text_bbox[1] - 2,
                    text_bbox[2] + 2,
                    text_bbox[3] + 2,
                ]
            except AttributeError:
                # Older Pillow versions don't have textbbox
                bg_box = [x1, y1 - 20, x1 + len(label_text) * 8, y1]

            draw.rectangle(bg_box, fill=colour_rgb)
            draw.text(
                (x1, y1 - 18),
                label_text,
                fill=(0, 0, 0),
                font=font_label,
            )
        else:
            # Manual annotation without bbox: draw a text badge in the corner
            _draw_corner_badge(draw, img.size, label_text, colour_rgb, font_small)

    return _to_bytes(img)


def draw_manual_annotation(
    image_bytes: bytes,
    label: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> bytes:
    """
    Draw a single manual bounding box annotation on an image.

    Args:
        image_bytes: Raw image bytes.
        label: Annotation label string.
        x1, y1, x2, y2: Bounding box corners in pixel coordinates.

    Returns:
        JPEG bytes of the annotated image.
    """
    ann = {
        "label": label,
        "confidence": 1.0,
        "bbox": [x1, y1, x2, y2],
        "source": "manual",
    }
    return draw_annotations(image_bytes, [ann])


def _draw_corner_badge(
    draw: ImageDraw.ImageDraw,
    img_size: tuple[int, int],
    text: str,
    colour: tuple[int, int, int],
    font,
    margin: int = 8,
):
    """
    Draw a coloured text badge in the bottom-left corner.
    Used for manual annotations that have no bounding box coordinates.
    """
    img_w, img_h = img_size
    # Estimate text size
    try:
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
    except AttributeError:
        tw = len(text) * 7
        th = 14

    x0 = margin
    y0 = img_h - th - margin * 2
    x1 = x0 + tw + margin
    y1 = img_h - margin

    draw.rectangle([x0, y0, x1, y1], fill=colour)
    draw.text((x0 + margin // 2, y0 + 2), text, fill=(0, 0, 0), font=font)


def _to_bytes(img: Image.Image) -> bytes:
    """Convert a PIL image to JPEG bytes."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()
