"""
app.py - Main Streamlit application for GSV Homelessness Annotation Tool
San Diego County Street View Analysis
"""

import streamlit as st
import pandas as pd
import json
import os
import io
from collections import Counter
from pathlib import Path
import tempfile
from PIL import Image

try:
    from streamlit.elements import image as st_image
    from streamlit.elements.lib.image_utils import image_to_url as st_image_to_url
    from streamlit.elements.lib.layout_utils import LayoutConfig
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st_image = None
    st_image_to_url = None
    LayoutConfig = None
    st_canvas = None

# Local modules
from gsv_fetcher import fetch_gsv_image, generate_grid_points, check_image_valid
from detector import (
    HomelessnessDetector,
    CUSTOM_WEIGHTS_PATH,
    CUSTOM_WEIGHTS_SOURCE,
    MODEL_OVERRIDE_ENV,
)
from annotator import draw_annotations, LABEL_COLOURS
from exporter import export_csv, export_zip
from project_overview import render_project_overview

HEADING_OPTIONS = {
    "Auto": None,
    "North (0°)": 0,
    "East (90°)": 90,
    "South (180°)": 180,
    "West (270°)": 270,
}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GSV Homelessness Detector - San Diego",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load API key from env or Streamlit secrets ────────────────────────────────
def get_api_key() -> str:
    """Try secrets first, then environment variable, then session state."""
    try:
        return st.secrets["GOOGLE_STREET_VIEW_API_KEY"]
    except Exception:
        pass
    env_key = os.getenv("GOOGLE_STREET_VIEW_API_KEY", "")
    if env_key:
        return env_key
    return st.session_state.get("api_key", "")


# ── Session state initialisation ──────────────────────────────────────────────
def init_session():
    defaults = {
        "fetched_images": [],       # list of dicts: {lat, lng, image, filename}
        "annotations": {},          # keyed by filename: list of annotation dicts
        "detection_done": False,
        "api_key": "",
        "selected_bounds": None,    # {"north":, "south":, "east":, "west":}
        "selected_heading_labels": ["Auto"],
        "fetch_debug": {},
        "review_group": "Detected by model",
        "review_index_by_group": {},
        "delete_canvas_version_by_file": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

if st_image is not None and st_image_to_url is not None:
    def _compat_image_to_url(image, width, clamp, channels, output_format, image_id):
        """Bridge old canvas calls to Streamlit's newer image_to_url signature."""
        layout_config = LayoutConfig(width=width) if LayoutConfig is not None else width
        return st_image_to_url(
            image=image,
            layout_config=layout_config,
            clamp=clamp,
            channels=channels,
            output_format=output_format,
            image_id=image_id,
        )

    st_image.image_to_url = _compat_image_to_url


def has_auto_detection(filename: str) -> bool:
    """Return True when the current image has at least one model detection."""
    anns = st.session_state.annotations.get(filename, [])
    return any(ann.get("source") == "auto" for ann in anns)


def get_review_groups(images: list[dict]) -> dict[str, list[dict]]:
    """Split images into detected and non-detected groups for review."""
    detected = [item for item in images if has_auto_detection(item["filename"])]
    non_detected = [item for item in images if not has_auto_detection(item["filename"])]
    return {
        "Detected by model": detected,
        "No model detections": non_detected,
    }


def get_canvas_bbox(canvas_result, scale_ratio: float, img_w: int, img_h: int):
    """Convert the most recent drawn rectangle into original-image coordinates."""
    if not canvas_result or not canvas_result.json_data:
        return None

    objects = canvas_result.json_data.get("objects") or []
    rects = [obj for obj in objects if obj.get("type") == "rect"]
    if not rects:
        return None

    rect = rects[-1]
    left = float(rect.get("left", 0))
    top = float(rect.get("top", 0))
    width = float(rect.get("width", 0)) * float(rect.get("scaleX", 1))
    height = float(rect.get("height", 0)) * float(rect.get("scaleY", 1))

    x1 = max(0, min(int(round(left / scale_ratio)), img_w - 1))
    y1 = max(0, min(int(round(top / scale_ratio)), img_h - 1))
    x2 = max(1, min(int(round((left + width) / scale_ratio)), img_w))
    y2 = max(1, min(int(round((top + height) / scale_ratio)), img_h))

    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """Convert #RRGGBB to an rgba(...) CSS string."""
    hex_colour = hex_colour.lstrip("#")
    r, g, b = (int(hex_colour[i: i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def build_delete_canvas_drawing(annotations: list[dict], scale_ratio: float) -> dict:
    """Build a canvas JSON payload showing current annotation boxes."""
    objects = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        width = max(1, int(round((x2 - x1) * scale_ratio)))
        height = max(1, int(round((y2 - y1) * scale_ratio)))
        left = int(round(x1 * scale_ratio))
        top = int(round(y1 * scale_ratio))
        colour = LABEL_COLOURS.get(ann.get("label"), LABEL_COLOURS.get("manual", "#FF69B4"))
        objects.append(
            {
                "type": "rect",
                "version": "4.4.0",
                "originX": "left",
                "originY": "top",
                "left": left,
                "top": top,
                "width": width,
                "height": height,
                "fill": hex_to_rgba(colour, 0.12),
                "stroke": colour,
                "strokeWidth": 3,
                "strokeUniform": True,
                "transparentCorners": False,
                "angle": 0,
                "scaleX": 1,
                "scaleY": 1,
            }
        )
    return {"version": "4.4.0", "objects": objects}


def annotation_matches_canvas_object(annotation: dict, canvas_obj: dict, scale_ratio: float) -> bool:
    """Return True when a canvas rectangle corresponds to the annotation bbox."""
    bbox = annotation.get("bbox")
    if not bbox or canvas_obj.get("type") != "rect":
        return False
    x1, y1, x2, y2 = bbox
    expected_left = x1 * scale_ratio
    expected_top = y1 * scale_ratio
    expected_width = (x2 - x1) * scale_ratio
    expected_height = (y2 - y1) * scale_ratio

    actual_left = float(canvas_obj.get("left", 0))
    actual_top = float(canvas_obj.get("top", 0))
    actual_width = float(canvas_obj.get("width", 0)) * float(canvas_obj.get("scaleX", 1))
    actual_height = float(canvas_obj.get("height", 0)) * float(canvas_obj.get("scaleY", 1))

    tolerance = max(4.0, scale_ratio * 6.0)
    return (
        abs(expected_left - actual_left) <= tolerance
        and abs(expected_top - actual_top) <= tolerance
        and abs(expected_width - actual_width) <= tolerance
        and abs(expected_height - actual_height) <= tolerance
    )


def remove_deleted_canvas_annotations(
    annotations: list[dict],
    canvas_result,
    scale_ratio: float,
) -> list[dict]:
    """Keep only annotations whose boxes still exist on the delete canvas."""
    if not canvas_result or not canvas_result.json_data:
        return annotations

    remaining_objects = [
        obj for obj in (canvas_result.json_data.get("objects") or []) if obj.get("type") == "rect"
    ]

    matched_object_indexes = set()
    updated_annotations = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox:
            updated_annotations.append(ann)
            continue

        match_idx = next(
            (
                idx for idx, obj in enumerate(remaining_objects)
                if idx not in matched_object_indexes
                and annotation_matches_canvas_object(ann, obj, scale_ratio)
            ),
            None,
        )
        if match_idx is not None:
            matched_object_indexes.add(match_idx)
            updated_annotations.append(ann)

    return updated_annotations


def get_last_canvas_point(canvas_result, scale_ratio: float):
    """Extract the newest clicked point from a point-mode canvas."""
    if not canvas_result or not canvas_result.json_data:
        return None

    objects = canvas_result.json_data.get("objects") or []
    circles = [obj for obj in objects if obj.get("type") == "circle"]
    if not circles:
        return None

    circle = circles[-1]
    radius = float(circle.get("radius", 0))
    stroke_width = float(circle.get("strokeWidth", 0))
    left = float(circle.get("left", 0))
    top = float(circle.get("top", 0))

    center_x = left + radius + (stroke_width / 2.0)
    center_y = top
    return [center_x / scale_ratio, center_y / scale_ratio]


def remove_annotation_at_point(annotations: list[dict], point_xy: list[float]) -> list[dict]:
    """Remove the smallest bbox annotation containing the clicked point."""
    px, py = point_xy
    bbox_candidates = []
    for idx, ann in enumerate(annotations):
        bbox = ann.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = bbox
        if x1 <= px <= x2 and y1 <= py <= y2:
            area = max(1, (x2 - x1) * (y2 - y1))
            bbox_candidates.append((area, idx))

    if not bbox_candidates:
        return annotations

    _, remove_idx = min(bbox_candidates, key=lambda item: item[0])
    return [ann for idx, ann in enumerate(annotations) if idx != remove_idx]


def canvas_supported() -> bool:
    """Return True when the drawable canvas package is available and compatible."""
    return (
        st_canvas is not None
        and st_image is not None
        and hasattr(st_image, "image_to_url")
    )

# ── Sidebar configuration ─────────────────────────────────────────────────────
st.sidebar.title("⚙️ Configuration")

# API Key input (masked)
api_key_input = st.sidebar.text_input(
    "Google Street View API Key",
    value=st.session_state.api_key,
    type="password",
    help="Get a key at console.cloud.google.com. Enable Street View Static API.",
)
if api_key_input:
    st.session_state.api_key = api_key_input

API_KEY = get_api_key()

# Grid density
grid_spacing_m = st.sidebar.slider(
    "Grid spacing (meters between points)",
    min_value=50,
    max_value=500,
    value=150,
    step=25,
    help="Smaller values = more images fetched. Warning: large areas + small spacing = many API calls.",
)

# Confidence threshold
conf_threshold = st.sidebar.slider(
    "Detection confidence threshold",
    min_value=0.1,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

# Image size cap for display
max_images = st.sidebar.number_input(
    "Max images to fetch (safety cap)",
    min_value=1,
    max_value=100,
    value=20,
)

selected_heading_labels = st.sidebar.multiselect(
    "Street View headings",
    options=list(HEADING_OPTIONS.keys()),
    default=st.session_state.selected_heading_labels,
    help=(
        "Fetch one or more camera directions per grid point. Multiple headings "
        "increase coverage but also increase API calls and runtime."
    ),
)
if not selected_heading_labels:
    selected_heading_labels = ["Auto"]
st.session_state.selected_heading_labels = selected_heading_labels
selected_headings = [HEADING_OPTIONS[label] for label in selected_heading_labels]

per_point_views = len(selected_headings)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Target classes:**\n"
    "- `homeless_tent`\n"
    "- `homeless_cart`\n"
    "- `homeless_person`"
)

st.sidebar.markdown("---")
if CUSTOM_WEIGHTS_PATH:
    model_name = Path(CUSTOM_WEIGHTS_PATH).name
    source_text = CUSTOM_WEIGHTS_SOURCE or "custom"
    st.sidebar.caption(
        f"Using custom StreetBridge YOLO weights: `{model_name}` ({source_text})."
    )
else:
    if CUSTOM_WEIGHTS_SOURCE and CUSTOM_WEIGHTS_SOURCE.startswith("missing_override:"):
        missing_path = CUSTOM_WEIGHTS_SOURCE.split(":", 1)[1]
        st.sidebar.warning(
            f"{MODEL_OVERRIDE_ENV} was set, but the weights were not found at "
            f"`{missing_path}`. Falling back to the default YOLO model."
        )
    else:
        st.sidebar.caption(
            "No project-trained weights found. Falling back to the default YOLO model."
        )

# ── Main title ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .app-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0 0 0.55rem 0;
        padding: 0.45rem 0 0.35rem 0;
        overflow: visible;
    }
    .app-title-icon {
        font-size: clamp(2rem, 2.4vw, 2.6rem);
        line-height: 1;
        flex: 0 0 auto;
    }
    .app-title {
        font-size: clamp(2rem, 2.7vw, 3rem);
        font-weight: 800;
        line-height: 1.22;
        letter-spacing: -0.02em;
        margin: 0;
        color: inherit;
        word-break: normal;
        overflow-wrap: anywhere;
        overflow: visible;
        display: block;
    }
    .app-subtitle {
        font-size: 1.08rem;
        line-height: 1.55;
        color: rgba(250, 250, 250, 0.92);
        margin-top: 0.15rem;
        margin-bottom: 0.8rem;
        max-width: 1000px;
    }
    </style>
    <div class="app-header">
        <span class="app-title-icon">🏙️</span>
        <div class="app-title">GSV Homelessness Indicator Detector</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="app-subtitle">Select an area in San Diego County, fetch Street View images, '
    "run automated object detection, manually review/correct annotations, then export "
    "results.</div>",
    unsafe_allow_html=True,
)

# ── Tab layout ────────────────────────────────────────────────────────────────
tab_overview, tab_map, tab_images, tab_annotate, tab_export = st.tabs(
    [
        "0. Project Overview",
        "1. Select Area",
        "2. Fetch & Detect",
        "3. Review & Annotate",
        "4. Export",
    ]
)

with tab_overview:
    render_project_overview(Path(__file__).resolve().parent)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Map selection
# ═════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.subheader("Draw a bounding box on the map")
    st.markdown(
        "Use the **rectangle tool** (□) in the map toolbar to draw a selection area. "
        "The coordinates will appear below after you draw."
    )

    try:
        import folium
        from streamlit_folium import st_folium
        from folium.plugins import Draw

        # San Diego County centre
        SD_LAT, SD_LNG = 32.7157, -117.1611

        m = folium.Map(location=[SD_LAT, SD_LNG], zoom_start=11)

        # Add draw plugin (rectangle + polygon only for simplicity)
        draw = Draw(
            draw_options={
                "polyline": False,
                "polygon": False,
                "circle": False,
                "marker": False,
                "circlemarker": False,
                "rectangle": True,
            },
            edit_options={"edit": True},
        )
        draw.add_to(m)

        # Render the map; capture drawn geometry
        map_output = st_folium(m, width=900, height=500, returned_objects=["all_drawings"])

        # Parse the drawn rectangle bounds
        if map_output and map_output.get("all_drawings"):
            drawings = map_output["all_drawings"]
            if drawings:
                last = drawings[-1]
                geom = last.get("geometry", {})
                coords = geom.get("coordinates", [[]])[0]  # list of [lng, lat] pairs
                if coords:
                    lngs = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    bounds = {
                        "north": max(lats),
                        "south": min(lats),
                        "east": max(lngs),
                        "west": min(lngs),
                    }
                    st.session_state.selected_bounds = bounds
                    st.success(
                        f"Selected bounds: N={bounds['north']:.4f}, S={bounds['south']:.4f}, "
                        f"E={bounds['east']:.4f}, W={bounds['west']:.4f}"
                    )

                    # Estimate number of sample points
                    pts = generate_grid_points(bounds, grid_spacing_m)
                    n_images = min(len(pts), max_images) * per_point_views
                    st.info(
                        f"Grid will produce **{len(pts)}** sample points "
                        f"(capped at **{min(len(pts), max_images)}** points by the safety limit). "
                        f"With **{per_point_views}** heading(s), up to **{n_images}** images "
                        f"may be requested."
                    )

        # Manual coordinate fallback
        with st.expander("Or enter coordinates manually"):
            col1, col2 = st.columns(2)
            with col1:
                north = st.number_input("North latitude", value=32.74, format="%.6f")
                south = st.number_input("South latitude", value=32.70, format="%.6f")
            with col2:
                east = st.number_input("East longitude", value=-117.13, format="%.6f")
                west = st.number_input("West longitude", value=-117.19, format="%.6f")
            if st.button("Use these coordinates"):
                st.session_state.selected_bounds = {
                    "north": north, "south": south, "east": east, "west": west
                }
                st.success("Bounds set manually.")

    except ImportError:
        st.error(
            "streamlit-folium is not installed. Run: pip install streamlit-folium folium"
        )

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Fetch images and run detection
# ═════════════════════════════════════════════════════════════════════════════
with tab_images:
    st.subheader("Fetch Street View Images & Run Detection")

    bounds = st.session_state.selected_bounds

    if not bounds:
        st.warning("Go to **Tab 1** and select an area first.")
    elif not API_KEY:
        st.warning("Enter your Google Street View API key in the sidebar.")
    else:
        col_fetch, col_detect = st.columns(2)

        with col_fetch:
            if st.button("🌐 Fetch GSV Images", use_container_width=True):
                pts = generate_grid_points(bounds, grid_spacing_m)
                pts = pts[:max_images]

                st.session_state.fetched_images = []
                st.session_state.annotations = {}
                st.session_state.detection_done = False
                st.session_state.fetch_debug = {}

                progress = st.progress(0, text="Fetching images...")
                fetched = []
                fetch_status_counts: Counter[str] = Counter()

                total_requests = max(len(pts) * per_point_views, 1)
                request_idx = 0
                for point_idx, (lat, lng) in enumerate(pts, start=1):
                    for heading in selected_headings:
                        request_idx += 1
                        img_bytes, filename, fetch_status = fetch_gsv_image(
                            lat,
                            lng,
                            API_KEY,
                            heading=heading,
                        )
                        fetch_status_counts[fetch_status] += 1
                        if img_bytes and check_image_valid(img_bytes):
                            fetched.append({
                                "lat": lat,
                                "lng": lng,
                                "heading": heading,
                                "heading_label": (
                                    "Auto" if heading is None else f"{heading}°"
                                ),
                                "image_bytes": img_bytes,
                                "filename": filename,
                                "annotated_bytes": None,
                            })
                        elif img_bytes:
                            fetch_status_counts["filtered_invalid_image"] += 1
                        progress.progress(
                            request_idx / total_requests,
                            text=(
                                f"Fetching point {point_idx}/{len(pts)} "
                                f"view {request_idx}/{total_requests} — valid: {len(fetched)}"
                            ),
                        )

                progress.empty()
                st.session_state.fetched_images = fetched
                st.session_state.fetch_debug = dict(fetch_status_counts)
                st.success(
                    f"Fetched **{len(fetched)}** valid images from "
                    f"**{len(pts)}** points across **{per_point_views}** heading(s)."
                )
                if not fetched:
                    st.warning(
                        "No valid images were kept. See fetch diagnostics below for "
                        "the reason breakdown."
                    )
                if fetch_status_counts:
                    debug_df = pd.DataFrame(
                        [
                            {"status": status, "count": count}
                            for status, count in sorted(fetch_status_counts.items())
                        ]
                    )
                    st.markdown("#### Fetch Diagnostics")
                    st.dataframe(debug_df, use_container_width=True, hide_index=True)

        with col_detect:
            if st.session_state.fetched_images:
                if st.button("🤖 Run YOLO Detection", use_container_width=True):
                    detector = HomelessnessDetector(conf_threshold=conf_threshold)
                    progress = st.progress(0, text="Running detection...")

                    for i, item in enumerate(st.session_state.fetched_images):
                        detections = detector.detect(item["image_bytes"])
                        annotated = draw_annotations(item["image_bytes"], detections)
                        st.session_state.fetched_images[i]["annotated_bytes"] = annotated

                        # Store detections as base annotation
                        st.session_state.annotations[item["filename"]] = detections

                        progress.progress(
                            (i + 1) / len(st.session_state.fetched_images),
                            text=f"Detecting {i+1}/{len(st.session_state.fetched_images)}",
                        )

                    progress.empty()
                    st.session_state.detection_done = True
                    st.success("Detection complete.")
            elif st.session_state.fetch_debug:
                st.info(
                    "Run YOLO Detection is hidden because no valid images were fetched."
                )

        # Gallery display
        imgs = st.session_state.fetched_images
        if imgs:
            st.markdown(f"### Image Gallery ({len(imgs)} images)")
            cols_per_row = 3
            for row_start in range(0, len(imgs), cols_per_row):
                row_items = imgs[row_start: row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, item in zip(cols, row_items):
                    with col:
                        display_bytes = item["annotated_bytes"] or item["image_bytes"]
                        st.image(display_bytes, use_container_width=True)
                        st.caption(
                            f"📍 {item['lat']:.5f}, {item['lng']:.5f} "
                            f"• view {item.get('heading_label', 'Auto')}"
                        )

                        # Show detection summary
                        anns = st.session_state.annotations.get(item["filename"], [])
                        if anns:
                            labels = [a["label"] for a in anns]
                            st.caption(f"Detections: {', '.join(labels)}")
                        elif st.session_state.detection_done:
                            st.caption("No detections.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Manual annotation review
# ═════════════════════════════════════════════════════════════════════════════
with tab_annotate:
    st.subheader("Review and Manually Correct Annotations")

    imgs = st.session_state.fetched_images
    if not imgs:
        st.warning("Fetch images first (Tab 2).")
    else:
        MANUAL_LABELS = [
            "homeless_person",
            "homeless_cart",
            "homeless_tent",
        ]

        review_groups = get_review_groups(imgs)
        available_groups = [
            group_name for group_name, group_items in review_groups.items() if group_items
        ]

        if not available_groups:
            st.info("No images are ready for review yet.")
            st.stop()

        if st.session_state.review_group not in available_groups:
            st.session_state.review_group = available_groups[0]

        selected_group = st.radio(
            "Review queue",
            options=available_groups,
            horizontal=True,
            key="review_group",
            help="Review model-detected images separately from images with no detections.",
        )
        group_items = review_groups[selected_group]
        group_filenames = [item["filename"] for item in group_items]
        review_indexes = st.session_state.review_index_by_group
        current_index = review_indexes.get(selected_group, 0)
        current_index = max(0, min(current_index, len(group_items) - 1))
        review_indexes[selected_group] = current_index
        select_key = f"review_select_{selected_group}"

        nav_prev, nav_status, nav_next = st.columns([1, 2, 1])
        with nav_prev:
            if st.button(
                "← Previous",
                key=f"prev_{selected_group}",
                use_container_width=True,
                disabled=current_index == 0,
            ):
                new_index = current_index - 1
                review_indexes[selected_group] = new_index
                st.session_state[select_key] = group_filenames[new_index]
                st.rerun()
        with nav_status:
            st.caption(
                f"{selected_group}: image {current_index + 1} of {len(group_items)}"
            )
        with nav_next:
            if st.button(
                "Next →",
                key=f"next_{selected_group}",
                use_container_width=True,
                disabled=current_index >= len(group_items) - 1,
            ):
                new_index = current_index + 1
                review_indexes[selected_group] = new_index
                st.session_state[select_key] = group_filenames[new_index]
                st.rerun()

        expected_file = group_filenames[review_indexes[selected_group]]
        if st.session_state.get(select_key) not in group_filenames:
            st.session_state[select_key] = expected_file

        selected_file = st.selectbox(
            "Image in current review queue",
            options=group_filenames,
            index=group_filenames.index(st.session_state[select_key]),
            key=select_key,
        )
        review_indexes[selected_group] = group_filenames.index(selected_file)

        selected_item = next((x for x in imgs if x["filename"] == selected_file), None)
        if selected_item:
            col_img, col_ann = st.columns([2, 1])
            current_anns = st.session_state.annotations.get(selected_file, [])
            display_bytes = selected_item["annotated_bytes"] or selected_item["image_bytes"]
            display_image = Image.open(io.BytesIO(display_bytes))
            img_w, img_h = display_image.size
            delete_mode_enabled = bool(st.session_state.get(f"delete_mode_{selected_file}", False))
            delete_canvas_versions = st.session_state.delete_canvas_version_by_file
            delete_canvas_version = delete_canvas_versions.get(selected_file, 0)

            with col_ann:
                st.markdown("#### Current Annotations")

                if current_anns:
                    ann_df = pd.DataFrame(current_anns)[
                        ["label", "confidence", "source"]
                    ]
                    st.dataframe(ann_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No annotations for this image yet.")

                st.markdown("#### Add Manual Annotations")
                manual_label = st.selectbox("Label", MANUAL_LABELS, key=f"ml_{selected_file}")
                manual_note = st.text_input(
                    "Optional note", key=f"mn_{selected_file}", placeholder="e.g. confident sighting"
                )
                manual_colour = LABEL_COLOURS.get(
                    manual_label,
                    LABEL_COLOURS.get("manual", "#FF69B4"),
                )
                st.caption(f"Image size: {img_w} x {img_h}")
                bbox_annotations = [ann for ann in current_anns if ann.get("bbox")]
                label_only_annotations = [ann for ann in current_anns if not ann.get("bbox")]

                bbox_mode = st.radio(
                    "Manual annotation type",
                    options=["Bounding box", "Label only"],
                    key=f"bbox_mode_{selected_file}",
                    horizontal=True,
                    help=(
                        "Use a bounding box for spatially precise corrections, or "
                        "choose label-only when the object is visible but hard to box."
                    ),
                )
                if bbox_mode == "Bounding box" and canvas_supported():
                    st.caption("Draw the new box directly on the main image.")

            bbox = None
            delete_canvas_result = None
            delete_scale_ratio = 1.0
            with col_img:
                if delete_mode_enabled and canvas_supported():
                    delete_canvas_width = min(img_w, 900)
                    delete_scale_ratio = delete_canvas_width / img_w if img_w else 1
                    delete_canvas_height = max(1, int(round(img_h * delete_scale_ratio)))
                    delete_canvas_image = display_image.resize(
                        (delete_canvas_width, delete_canvas_height)
                    )
                    st.caption(
                        "Click once inside the box you want to remove on the main image."
                    )
                    delete_canvas_result = st_canvas(
                        fill_color="rgba(255, 255, 255, 0.0)",
                        stroke_width=2,
                        stroke_color="#4B9EFF",
                        background_image=delete_canvas_image,
                        update_streamlit=True,
                        height=delete_canvas_height,
                        width=delete_canvas_width,
                        drawing_mode="point",
                        point_display_radius=4,
                        display_toolbar=True,
                        key=f"delete_canvas_{selected_file}_{delete_canvas_version}",
                    )
                elif bbox_mode == "Bounding box" and canvas_supported():
                    canvas_width = min(img_w, 900)
                    scale_ratio = canvas_width / img_w if img_w else 1
                    canvas_height = max(1, int(round(img_h * scale_ratio)))
                    canvas_image = display_image.resize((canvas_width, canvas_height))

                    st.caption(
                        "Draw directly on the main image. The newest rectangle will be used."
                    )
                    canvas_result = st_canvas(
                        fill_color=hex_to_rgba(manual_colour, 0.18),
                        stroke_width=3,
                        stroke_color=manual_colour,
                        background_image=canvas_image,
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode="rect",
                        display_toolbar=True,
                        key=f"canvas_{selected_file}",
                    )
                    bbox = get_canvas_bbox(
                        canvas_result, scale_ratio=scale_ratio, img_w=img_w, img_h=img_h
                    )
                else:
                    st.image(display_bytes, caption=selected_file, use_container_width=True)

                st.caption(
                    f"Location: {selected_item['lat']:.6f}, {selected_item['lng']:.6f} "
                    f"• view {selected_item.get('heading_label', 'Auto')}"
                )

            with col_ann:
                if bbox_mode == "Bounding box":
                    st.markdown("#### Bounding Box")
                    if canvas_supported():
                        if bbox:
                            st.caption(
                                f"Selected box: x1={bbox[0]}, y1={bbox[1]}, "
                                f"x2={bbox[2]}, y2={bbox[3]}"
                            )
                        else:
                            st.caption("Draw a bounding box on the main image to select it.")
                    else:
                        st.warning(
                            "Mouse-drag annotation is unavailable with the current "
                            "Streamlit/canvas package combination. Using coordinate "
                            "inputs instead."
                        )
                        x1_default = 0
                        y1_default = 0
                        x2_default = min(img_w, max(1, img_w // 2))
                        y2_default = min(img_h, max(1, img_h // 2))

                        col_x1, col_y1 = st.columns(2)
                        with col_x1:
                            x1 = st.number_input(
                                "x1",
                                min_value=0,
                                max_value=max(img_w - 1, 0),
                                value=x1_default,
                                step=1,
                                key=f"x1_{selected_file}",
                            )
                        with col_y1:
                            y1 = st.number_input(
                                "y1",
                                min_value=0,
                                max_value=max(img_h - 1, 0),
                                value=y1_default,
                                step=1,
                                key=f"y1_{selected_file}",
                            )

                        col_x2, col_y2 = st.columns(2)
                        with col_x2:
                            x2 = st.number_input(
                                "x2",
                                min_value=1,
                                max_value=img_w,
                                value=x2_default,
                                step=1,
                                key=f"x2_{selected_file}",
                            )
                        with col_y2:
                            y2 = st.number_input(
                                "y2",
                                min_value=1,
                                max_value=img_h,
                                value=y2_default,
                                step=1,
                                key=f"y2_{selected_file}",
                            )
                        bbox = [int(x1), int(y1), int(x2), int(y2)]

                if st.button("➕ Add Manual Annotation", key=f"btn_{selected_file}"):
                    if bbox_mode == "Bounding box":
                        if not bbox:
                            st.error("Draw a bounding box on the image before saving.")
                            st.stop()
                        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                            st.error("Bounding box must satisfy x2 > x1 and y2 > y1.")
                            st.stop()

                    new_ann = {
                        "label": manual_label,
                        "confidence": 1.0,
                        "source": "manual",
                        "note": manual_note,
                        "bbox": bbox,
                    }
                    if selected_file not in st.session_state.annotations:
                        st.session_state.annotations[selected_file] = []
                    st.session_state.annotations[selected_file].append(new_ann)

                    updated_bytes = draw_annotations(
                        selected_item["image_bytes"],
                        st.session_state.annotations[selected_file],
                    )
                    idx = next(
                        (i for i, x in enumerate(imgs) if x["filename"] == selected_file), None
                    )
                    if idx is not None:
                        st.session_state.fetched_images[idx]["annotated_bytes"] = updated_bytes
                    st.success(f"Added '{manual_label}'.")
                    st.rerun()

                st.markdown("---")
                st.markdown("#### Remove Annotations")
                if bbox_annotations and canvas_supported():
                    delete_mode_enabled = st.toggle(
                        "Delete boxed annotations on image",
                        key=f"delete_mode_{selected_file}",
                        help="Click once inside a box on the image, then press the delete button.",
                    )

                if delete_mode_enabled:
                    st.caption(
                        "Click once inside the box you want to remove on the main image, "
                        "then press the button below."
                    )
                    clicked_point = get_last_canvas_point(
                        delete_canvas_result, scale_ratio=delete_scale_ratio
                    )
                    if clicked_point:
                        st.caption(
                            f"Selected point: x={int(round(clicked_point[0]))}, "
                            f"y={int(round(clicked_point[1]))}"
                        )
                    else:
                        st.caption("No box selected yet. Click once inside the box to remove.")

                    clear_delete_selection = st.button(
                        "Reset box selection",
                        key=f"clear_delete_point_{selected_file}",
                    )
                    if clear_delete_selection:
                        delete_canvas_versions[selected_file] = delete_canvas_version + 1
                        st.rerun()

                    if st.button("Delete selected boxed annotation", key=f"apply_delete_{selected_file}"):
                        if not clicked_point:
                            st.info("Click once inside a boxed annotation before deleting it.")
                        else:
                            updated_anns = remove_annotation_at_point(current_anns, clicked_point)
                            if len(updated_anns) == len(current_anns):
                                st.info("No boxed annotation matched that click.")
                            else:
                                st.session_state.annotations[selected_file] = updated_anns
                                updated_bytes = draw_annotations(
                                    selected_item["image_bytes"],
                                    updated_anns,
                                )
                                idx = next(
                                    (i for i, x in enumerate(imgs) if x["filename"] == selected_file),
                                    None,
                                )
                                if idx is not None:
                                    st.session_state.fetched_images[idx]["annotated_bytes"] = updated_bytes
                                st.success("Removed the selected boxed annotation.")
                                st.rerun()

                if label_only_annotations:
                    st.markdown("#### Remove Label-Only Annotations")
                    removal_options = []
                    for idx, ann in enumerate(current_anns):
                        if ann.get("bbox"):
                            continue
                        bbox_text = (
                            f"bbox={ann['bbox']}"
                            if ann.get("bbox")
                            else "label-only"
                        )
                        removal_options.append(
                            f"{idx}: {ann.get('label', 'unknown')} "
                            f"({ann.get('source', 'auto')}, {bbox_text})"
                        )
                    selected_removal = st.selectbox(
                        "Select label-only annotation to remove",
                        options=removal_options,
                        key=f"remove_sel_{selected_file}",
                    )
                    if st.button("Remove selected annotation", key=f"remove_btn_{selected_file}"):
                        remove_idx = int(selected_removal.split(":", 1)[0])
                        updated_anns = [
                            ann for idx, ann in enumerate(current_anns) if idx != remove_idx
                        ]
                        st.session_state.annotations[selected_file] = updated_anns
                        updated_bytes = draw_annotations(
                            selected_item["image_bytes"],
                            updated_anns,
                        )
                        idx = next(
                            (i for i, x in enumerate(imgs) if x["filename"] == selected_file), None
                        )
                        if idx is not None:
                            st.session_state.fetched_images[idx]["annotated_bytes"] = updated_bytes
                        st.success("Removed selected annotation.")
                        st.rerun()

                # Clear all annotations button
                if current_anns:
                    if st.button("🗑️ Clear all annotations", key=f"clear_{selected_file}"):
                        st.session_state.annotations[selected_file] = []
                        idx = next(
                            (i for i, x in enumerate(imgs) if x["filename"] == selected_file), None
                        )
                        if idx is not None:
                            st.session_state.fetched_images[idx]["annotated_bytes"] = None
                        st.rerun()

        # Quick overview table across all images
        st.markdown("---")
        st.markdown("### Annotation Summary (all images)")
        rows = []
        for item in imgs:
            anns = st.session_state.annotations.get(item["filename"], [])
            label_str = ", ".join(set(a["label"] for a in anns)) if anns else "none"
            rows.append(
                {
                    "filename": item["filename"],
                    "lat": item["lat"],
                    "lng": item["lng"],
                    "heading": item.get("heading_label", "Auto"),
                    "labels": label_str,
                    "count": len(anns),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Export
# ═════════════════════════════════════════════════════════════════════════════
with tab_export:
    st.subheader("Export Annotations and Annotated Images")

    imgs = st.session_state.fetched_images
    if not imgs:
        st.warning("Nothing to export yet. Fetch and annotate images first.")
    else:
        col_csv, col_zip = st.columns(2)

        with col_csv:
            st.markdown("#### CSV Export")
            st.markdown(
                "Downloads a CSV with one row per detection: "
                "filename, lat/lng, label, confidence, bounding box, source."
            )
            csv_bytes = export_csv(imgs, st.session_state.annotations)
            st.download_button(
                label="⬇️ Download annotations.csv",
                data=csv_bytes,
                file_name="gsv_annotations.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_zip:
            st.markdown("#### ZIP Export")
            st.markdown(
                "Downloads a ZIP containing all annotated images "
                "plus the annotations CSV."
            )
            if st.button("📦 Prepare ZIP", use_container_width=True):
                with st.spinner("Building ZIP..."):
                    zip_bytes = export_zip(imgs, st.session_state.annotations)
                st.download_button(
                    label="⬇️ Download annotated_images.zip",
                    data=zip_bytes,
                    file_name="gsv_annotated_images.zip",
                    mime="application/zip",
                    use_container_width=True,
                )

        # Preview of the CSV data
        st.markdown("---")
        st.markdown("### Data Preview")
        rows = []
        for item in imgs:
            anns = st.session_state.annotations.get(item["filename"], [])
            if anns:
                for ann in anns:
                    rows.append(
                        {
                            "filename": item["filename"],
                            "lat": item["lat"],
                            "lng": item["lng"],
                            "heading": item.get("heading_label", "Auto"),
                            "label": ann.get("label"),
                            "confidence": ann.get("confidence"),
                            "source": ann.get("source", "auto"),
                            "bbox": str(ann.get("bbox")),
                        }
                    )
            else:
                rows.append(
                    {
                        "filename": item["filename"],
                        "lat": item["lat"],
                        "lng": item["lng"],
                        "heading": item.get("heading_label", "Auto"),
                        "label": None,
                        "confidence": None,
                        "source": None,
                        "bbox": None,
                    }
                )
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
