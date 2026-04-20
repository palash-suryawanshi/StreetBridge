"""
project_overview.py - Streamlit project website tab for StreetBridge.
"""

from __future__ import annotations

import json
import textwrap
import io
import base64
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps
import csv


PROJECT_TITLE = "StreetBridge"
PROJECT_SUBTITLE = "Google Street View Homelessness Indicator Detection in San Diego County"

TEAM_MEMBERS = [
    {
        "name": "Pranjal Patel",
        "role": "Project Team Member",
        "focus": "Project framing, presentation, website narrative, and capstone reporting.",
        "photo_candidates": [
            "assets/team/pranjal_patel.jpg",
            "assets/team/pranjal_patel.jpeg",
            "assets/team/pranjal_patel.png",
        ],
        "focus_y": 0.24,
    },
    {
        "name": "Palash Suryawanshi",
        "role": "Project Team Member",
        "focus": "Model pipeline, data workflow, evaluation artifacts, and app integration.",
        "photo_candidates": [
            "assets/team/palash_suryawanshi.jpg",
            "assets/team/palash_suryawanshi.jpeg",
            "assets/team/palash_suryawanshi.png",
        ],
        "focus_y": 0.18,
    },
]


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_training_results(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _get_best_metric(rows: list[dict], key: str) -> tuple[float, int | None]:
    best_value = 0.0
    best_epoch = None
    for row in rows:
        try:
            value = float(row[key])
            epoch = int(float(row["epoch"]))
        except Exception:
            continue
        if best_epoch is None or value > best_value:
            best_value = value
            best_epoch = epoch
    return best_value, best_epoch


def _find_existing_path(project_root: Path, relative_candidates: list[str]) -> Path | None:
    for relative in relative_candidates:
        candidate = project_root / relative
        if candidate.exists():
            return candidate
    return None


def _render_metric_card(label: str, value: str, help_text: str = ""):
    help_attr = f' title="{help_text}"' if help_text else ""
    st.markdown(
        f"""
        <div class="sb-card sb-metric"{help_attr}>
            <div class="sb-metric-value">{value}</div>
            <div class="sb-metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _prepare_team_photo(photo_path: Path, focus_y: float = 0.22) -> Image.Image:
    image = Image.open(photo_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    target_ratio = 1.0
    width, height = image.size
    current_ratio = width / height if height else target_ratio

    if current_ratio > target_ratio:
        crop_width = int(height * target_ratio)
        left = max(0, (width - crop_width) // 2)
        image = image.crop((left, 0, left + crop_width, height))
    else:
        crop_height = int(width / target_ratio)
        extra_height = max(0, height - crop_height)
        top = int(extra_height * focus_y)
        top = max(0, min(top, extra_height))
        image = image.crop((0, top, width, top + crop_height))

    return image


def _image_to_data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _render_member_card(member: dict, project_root: Path):
    st.markdown('<div class="sb-card sb-team-card">', unsafe_allow_html=True)
    photo_path = _find_existing_path(project_root, member["photo_candidates"])
    if photo_path:
        photo = _prepare_team_photo(photo_path, member.get("focus_y", 0.22)).resize((240, 240))
        photo_data_url = _image_to_data_url(photo)
        st.markdown(
            f'<img src="{photo_data_url}" alt="{member["name"]}" class="sb-team-photo" />',
            unsafe_allow_html=True,
        )
    else:
        initials = "".join(part[0] for part in member["name"].split()[:2]).upper()
        st.markdown(
            f"""
            <div class="sb-photo-placeholder">
                <span>{initials}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "Add a photo at "
            f"`{member['photo_candidates'][0]}` to replace this placeholder."
        )

    st.markdown(f"### {member['name']}")
    st.markdown(f"**{member['role']}**")
    st.write(member["focus"])
    st.markdown("</div>", unsafe_allow_html=True)


def _render_artifact_image(title: str, artifact_path: Path):
    if artifact_path.exists():
        st.markdown(f"#### {title}")
        st.image(str(artifact_path), use_container_width=True)


def _render_simple_card(title: str, body_html: str):
    cleaned_body = textwrap.dedent(body_html).strip()
    st.markdown(
        f"""
        <div class="sb-card sb-content-card">
            <h3>{title}</h3>
            {cleaned_body}
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_slide_header(number: str, title: str, subtitle: str = ""):
    subtitle_html = f'<p class="sb-slide-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f"""
        <div class="sb-slide-header">
            <div class="sb-slide-kicker">Slide {number}</div>
            <h2>{title}</h2>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_overview(project_root: Path):
    dataset_summary = _load_json(
        project_root / "datasets" / "yolo_homeless_4class" / "summary.json"
    )
    clean_dataset_summary = _load_json(
        project_root / "datasets" / "yolo_homeless_4class_clean" / "summary.json"
    )
    balanced_dataset_summary = _load_json(
        project_root / "datasets" / "yolo_homeless_4class_clean_balanced" / "summary.json"
    )
    active_run_dir = project_root / "runs" / "detect" / "homeless4_accuracy_v5_finetune"
    training_rows = _load_training_results(active_run_dir / "results.csv")
    evaluation_report = _load_json(
        project_root
        / "runs"
        / "eval"
        / "homeless4_accuracy_v5_finetune_test_4class"
        / "evaluation_report.json"
    )

    split_counts = dataset_summary.get("split_image_counts", {})
    total_boxes = dataset_summary.get("split_box_counts", {})
    test_box_counts = evaluation_report.get("ground_truth_box_counts", {})
    test_metrics = {
        "precision": 0.857,
        "recall": 0.670,
        "map50": 0.764,
        "map50_95": 0.488,
    }
    original_records = int(dataset_summary.get("records", 0))
    clean_images = sum(clean_dataset_summary.get("split_image_counts", {}).values())
    final_balanced_images = sum(balanced_dataset_summary.get("split_image_counts", {}).values())
    final_balanced_boxes = sum(balanced_dataset_summary.get("total_box_counts", {}).values())
    augmentation_images = int(balanced_dataset_summary.get("train_augmented_images", 0))
    class_names = dataset_summary.get("class_names", [])
    best_precision, best_precision_epoch = _get_best_metric(training_rows, "metrics/precision(B)")
    best_recall, best_recall_epoch = _get_best_metric(training_rows, "metrics/recall(B)")
    best_map50, best_map50_epoch = _get_best_metric(training_rows, "metrics/mAP50(B)")
    best_map50_95, best_map50_95_epoch = _get_best_metric(training_rows, "metrics/mAP50-95(B)")
    training_summary = {
        "best_epoch": best_map50_epoch,
        "best_precision": best_precision,
        "best_precision_epoch": best_precision_epoch,
        "best_recall": best_recall,
        "best_recall_epoch": best_recall_epoch,
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
        "best_map50_epoch": best_map50_epoch,
        "best_map50_95_epoch": best_map50_95_epoch,
        "epochs_recorded": len(training_rows),
    }
    artifact_paths = {
        "confusion_matrix": str(active_run_dir / "confusion_matrix.png"),
        "confusion_matrix_normalized": str(active_run_dir / "confusion_matrix_normalized.png"),
        "pr_curve": evaluation_report.get("artifact_paths", {}).get(
            "pr_curve",
            str(active_run_dir / "BoxPR_curve.png"),
        ),
        "precision_curve": evaluation_report.get("artifact_paths", {}).get(
            "precision_curve",
            str(active_run_dir / "BoxP_curve.png"),
        ),
        "recall_curve": evaluation_report.get("artifact_paths", {}).get(
            "recall_curve",
            str(active_run_dir / "BoxR_curve.png"),
        ),
        "f1_curve": evaluation_report.get("artifact_paths", {}).get(
            "f1_curve",
            str(active_run_dir / "BoxF1_curve.png"),
        ),
        "results_plot": str(active_run_dir / "results.png"),
    }

    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1200px;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
            gap: 0.35rem;
        }
        .sb-hero {
            position: relative;
            overflow: hidden;
            padding: 2.35rem 2.35rem 2rem 2.35rem;
            border-radius: 30px;
            background:
                radial-gradient(circle at 12% 18%, rgba(255, 210, 120, 0.22), transparent 22%),
                radial-gradient(circle at 88% 14%, rgba(108, 214, 255, 0.22), transparent 24%),
                radial-gradient(circle at 78% 78%, rgba(70, 164, 192, 0.18), transparent 28%),
                linear-gradient(135deg, #081924 0%, #0f3143 42%, #155a5f 100%);
            color: #f5fbff;
            border: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 22px 52px rgba(13, 30, 45, 0.22);
            margin-bottom: 1.4rem;
        }
        .sb-hero::before {
            content: "";
            position: absolute;
            inset: auto -4% -32% auto;
            width: 380px;
            height: 380px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.02) 55%, transparent 70%);
            filter: blur(8px);
        }
        .sb-hero::after {
            content: "";
            position: absolute;
            top: -30px;
            right: 90px;
            width: 180px;
            height: 180px;
            border: 1px solid rgba(255,255,255,0.09);
            border-radius: 26px;
            transform: rotate(18deg);
            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
        }
        .sb-kicker {
            letter-spacing: 0.08em;
            text-transform: uppercase;
            font-size: 0.82rem;
            font-weight: 700;
            opacity: 0.82;
            position: relative;
            z-index: 1;
        }
        .sb-hero h1 {
            font-size: 3.25rem;
            margin: 0.35rem 0 0.55rem 0;
            line-height: 1.05;
            letter-spacing: -0.03em;
            position: relative;
            z-index: 1;
        }
        .sb-hero p {
            font-size: 1.06rem;
            max-width: 840px;
            margin-bottom: 0;
            position: relative;
            z-index: 1;
            color: rgba(245, 251, 255, 0.92);
        }
        .sb-hero-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.65rem;
            margin-top: 1rem;
            position: relative;
            z-index: 1;
        }
        .sb-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.45rem 0.75rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.10);
            border: 1px solid rgba(255, 255, 255, 0.14);
            font-size: 0.86rem;
            color: #f7fbfd;
            backdrop-filter: blur(6px);
        }
        .sb-card {
            background:
                linear-gradient(180deg, rgba(255,255,255,0.99), rgba(247,251,252,0.98));
            border: 1px solid rgba(22, 72, 84, 0.09);
            border-radius: 24px;
            padding: 1.15rem 1.15rem;
            box-shadow: 0 16px 34px rgba(12, 41, 52, 0.08);
            backdrop-filter: blur(4px);
        }
        .sb-metric {
            min-height: 146px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            position: relative;
            overflow: hidden;
        }
        .sb-metric::after {
            content: "";
            position: absolute;
            right: -18px;
            top: -18px;
            width: 86px;
            height: 86px;
            border-radius: 24px;
            background: linear-gradient(180deg, rgba(45, 139, 167, 0.10), rgba(255, 193, 112, 0.12));
            transform: rotate(20deg);
        }
        .sb-metric-value {
            font-size: 2.15rem;
            font-weight: 700;
            color: #113a4c;
            line-height: 1;
            margin-bottom: 0.5rem;
            position: relative;
            z-index: 1;
        }
        .sb-metric-label {
            color: #42606d;
            font-size: 0.95rem;
            position: relative;
            z-index: 1;
        }
        .sb-section {
            margin-top: 1.15rem;
            margin-bottom: 0.85rem;
        }
        .sb-section h2 {
            color: #10394b;
            margin-bottom: 0.3rem;
        }
        .sb-slide-header {
            margin: 1.6rem 0 0.85rem 0;
        }
        .sb-slide-kicker {
            display: inline-block;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(25, 94, 116, 0.09);
            color: #1c6276;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.55rem;
        }
        .sb-slide-header h2 {
            margin: 0;
            color: #eef6fb;
            font-size: 2rem;
            letter-spacing: -0.02em;
        }
        .sb-slide-subtitle {
            margin: 0.45rem 0 0 0;
            max-width: 860px;
            color: rgba(236, 244, 248, 0.84);
            font-size: 1rem;
            line-height: 1.6;
        }
        .sb-section p, .sb-section li {
            color: #233841;
            line-height: 1.62;
        }
        .sb-section ul {
            margin-bottom: 0;
        }
        .sb-team-photo {
            border-radius: 999px;
            width: 190px;
            height: 190px;
            object-fit: cover;
            margin: 0 auto 1rem auto;
            display: block;
            border: 4px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 12px 28px rgba(8, 25, 36, 0.18);
        }
        .sb-team-card {
            position: relative;
            overflow: hidden;
            text-align: center;
        }
        .sb-team-card::after {
            content: "";
            position: absolute;
            left: -18px;
            bottom: -18px;
            width: 110px;
            height: 110px;
            border-radius: 32px;
            background: linear-gradient(180deg, rgba(255, 191, 105, 0.08), rgba(29, 117, 139, 0.10));
            transform: rotate(18deg);
        }
        .sb-photo-placeholder {
            width: 190px;
            height: 190px;
            border-radius: 999px;
            background:
                radial-gradient(circle at top, rgba(255, 210, 120, 0.16), transparent 32%),
                linear-gradient(145deg, rgba(38, 110, 132, 0.16), rgba(17, 56, 74, 0.08)),
                repeating-linear-gradient(
                    -45deg,
                    rgba(26, 91, 111, 0.07),
                    rgba(26, 91, 111, 0.07) 12px,
                    rgba(255, 255, 255, 0.28) 12px,
                    rgba(255, 255, 255, 0.28) 24px
                );
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 1rem auto;
            border: 4px solid rgba(255, 255, 255, 0.9);
            box-shadow: 0 12px 28px rgba(8, 25, 36, 0.18);
        }
        .sb-photo-placeholder span {
            width: 92px;
            height: 92px;
            border-radius: 50%;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            background: #164a5e;
            color: white;
            font-weight: 700;
            font-size: 2rem;
        }
        .sb-swatch {
            display: inline-block;
            width: 0.9rem;
            height: 0.9rem;
            border-radius: 999px;
            margin-right: 0.5rem;
            vertical-align: middle;
        }
        .sb-feature-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 1rem;
        }
        .sb-feature {
            position: relative;
            overflow: hidden;
            min-height: 210px;
        }
        .sb-feature::before {
            content: "";
            position: absolute;
            inset: 0 auto auto 0;
            width: 100%;
            height: 6px;
            background: linear-gradient(90deg, #ffbf69, #5cc7df, #2f7f93);
        }
        .sb-feature h3 {
            margin-top: 0.35rem;
            margin-bottom: 0.5rem;
            color: #10394b;
        }
        .sb-feature p {
            margin-bottom: 0;
        }
        .sb-stat-banner {
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 1rem;
            align-items: stretch;
        }
        .sb-highlight-panel {
            padding: 1.35rem;
            border-radius: 24px;
            background:
                radial-gradient(circle at top right, rgba(92, 199, 223, 0.16), transparent 28%),
                linear-gradient(135deg, rgba(15,49,67,0.96), rgba(21,90,95,0.94));
            color: #f3fbfe;
            box-shadow: 0 18px 36px rgba(11, 33, 48, 0.18);
        }
        .sb-highlight-panel h3 {
            color: #ffffff;
            margin-bottom: 0.35rem;
        }
        .sb-highlight-panel p, .sb-highlight-panel li {
            color: rgba(243, 251, 254, 0.92);
        }
        .sb-highlight-panel ul {
            margin-bottom: 0;
        }
        .sb-outline-panel {
            padding: 1.15rem;
            border-radius: 24px;
            border: 1px solid rgba(20, 69, 83, 0.10);
            background:
                linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,251,252,0.98));
        }
        .sb-outline-panel h3 {
            margin-bottom: 0.35rem;
            color: #10394b;
        }
        .sb-content-card {
            height: 100%;
        }
        .sb-content-card h3 {
            color: #10394b;
            margin-bottom: 0.7rem;
        }
        .sb-slide-card {
            padding: 1.35rem 1.35rem;
        }
        .sb-slide-card h3 {
            color: #10394b;
            margin-bottom: 0.6rem;
        }
        .sb-slide-card p, .sb-slide-card li {
            color: #233841;
            line-height: 1.65;
        }
        .sb-class-list {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-top: 0.35rem;
        }
        .sb-class-item {
            display: flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.02rem;
            color: #1f3640;
        }
        .sb-class-dot {
            width: 0.95rem;
            height: 0.95rem;
            border-radius: 999px;
            flex: 0 0 auto;
            box-shadow: 0 0 0 5px rgba(17, 57, 76, 0.04);
        }
        .sb-code-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.32rem 0.6rem;
            border-radius: 12px;
            background: #f2f7f9;
            border: 1px solid rgba(19, 63, 79, 0.08);
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            color: #14506a;
            font-size: 0.96rem;
        }
        .sb-metric-list {
            display: flex;
            flex-direction: column;
            gap: 0.8rem;
            margin: 0.3rem 0 1rem 0;
        }
        .sb-metric-row {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            gap: 1rem;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid rgba(20, 69, 83, 0.08);
        }
        .sb-metric-row:last-child {
            border-bottom: none;
            padding-bottom: 0;
        }
        .sb-metric-name {
            color: #26424f;
            font-weight: 600;
        }
        .sb-metric-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.58rem;
            border-radius: 12px;
            background: rgba(33, 140, 102, 0.10);
            color: #177754;
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 0.95rem;
            font-weight: 700;
        }
        .sb-support-note {
            margin: 0.85rem 0 0.75rem 0;
            padding: 0.9rem 1rem;
            border-radius: 16px;
            background: linear-gradient(90deg, rgba(58, 149, 255, 0.14), rgba(58, 149, 255, 0.06));
            border: 1px solid rgba(58, 149, 255, 0.14);
            color: #21598b;
            font-weight: 500;
        }
        @media (max-width: 980px) {
            .sb-feature-grid,
            .sb-stat-banner {
                grid-template-columns: 1fr;
            }
            .sb-hero h1 {
                font-size: 2.5rem;
            }
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="sb-hero" style="margin-top: 1.2rem;">
            <div class="sb-kicker">Capstone Project Website</div>
            <h1>{PROJECT_TITLE}</h1>
            <p>{PROJECT_SUBTITLE}. In this project, we developed a geospatial and computer vision
            workflow to identify visible homelessness-related indicators in Street View imagery and
            support structured urban analysis in San Diego County.</p>
            <div class="sb-hero-badges">
                <span class="sb-badge">Geospatial Sampling</span>
                <span class="sb-badge">YOLO Detection</span>
                <span class="sb-badge">Human-in-the-Loop Review</span>
                <span class="sb-badge">Presentation-Ready Analytics</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="sb-section sb-feature-grid">
            <div class="sb-card sb-feature">
                <h3>End-to-End System</h3>
                <p>We built a complete workflow that connects image collection, dataset building, model inference, manual review, and final export inside one application.</p>
            </div>
            <div class="sb-card sb-feature">
                <h3>Research-Ready Pipeline</h3>
                <p>We designed dataset preparation, cleaning, balancing, augmentation, and evaluation steps so the work reflects the full project pipeline, not just one final checkpoint.</p>
            </div>
            <div class="sb-card sb-feature">
                <h3>Presentation-Ready Output</h3>
                <p>The final system presents our methods, project scale, results, and visual outputs clearly for faculty review, live demonstration, and capstone reporting.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(2)
    with metric_cols[0]:
        _render_metric_card(
            "Final Balanced Images",
            f"{final_balanced_images:,}",
            "Expanded dataset after cleaning, regrouping, and targeted augmentation.",
        )
    with metric_cols[1]:
        _render_metric_card(
            "Total Balanced Boxes",
            f"{final_balanced_boxes:,}",
            "Total annotations across the final balanced dataset.",
        )

    st.markdown(
        """
        <div class="sb-section sb-stat-banner">
            <div class="sb-highlight-panel">
                <h3>Project Significance</h3>
                <p>Our goal was not only to train a detector, but to deliver a practical analytics workflow that connects geospatial sampling, visual detection, manual validation, and reporting in a form that can be demonstrated and extended.</p>
            </div>
            <div class="sb-outline-panel">
                <h3>Core Strengths</h3>
                <ul>
                    <li>Real-world social issue</li>
                    <li>Strong technical workflow</li>
                    <li>Interactive review interface</li>
                    <li>Presentation-ready outputs</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_slide_header(
        "01",
        "Project Overview",
        "This presentation summarizes our capstone problem, workflow, model results, and the significance of the final system."
    )
    intro_col, team_col = st.columns([1.2, 1])
    with intro_col:
        st.markdown('<div class="sb-card sb-slide-card">', unsafe_allow_html=True)
        st.markdown("### Project Summary")
        st.write(
            "We developed StreetBridge as a capstone project to examine whether visible indicators "
            "in Street View imagery can support street-level analysis of homelessness-related "
            "conditions. Our system combines map-based area selection, image retrieval, model "
            "inference, manual review, and exportable outputs in one reproducible workflow. "
            "Our selected production-ready detector for the 4-class workflow is "
            "`homeless4_accuracy_v5_finetune`."
        )
        st.markdown("### Why This Project Matters")
        st.markdown(
            "- Homelessness is a visible and urgent urban issue.\n"
            "- Manual image review is time-intensive and difficult to scale.\n"
            "- A structured computer vision workflow can support faster, more consistent analysis.\n"
            "- Human review remains essential for validation and interpretation."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with team_col:
        st.markdown('<div class="sb-card sb-slide-card">', unsafe_allow_html=True)
        st.markdown("### Team Responsibilities")
        st.markdown(
            "- `Pranjal Patel`: presentation development, report framing, project website content, and project communication.\n"
            "- `Palash Suryawanshi`: data pipeline, model evaluation, technical integration, and application workflow."
        )
        st.markdown("### Research Goals")
        st.markdown(
            "- Develop a reproducible Street View sampling pipeline.\n"
            "- Train and refine a 4-class model for visible homelessness-related indicators.\n"
            "- Preserve human oversight through review and annotation.\n"
            "- Produce structured outputs for maps, charts, and reporting."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    _render_slide_header(
        "02",
        "Team",
        "Our project combines technical implementation, model analysis, reporting, and presentation design."
    )
    member_cols = st.columns(2)
    for col, member in zip(member_cols, TEAM_MEMBERS):
        with col:
            _render_member_card(member, project_root)

    _render_slide_header(
        "03",
        "Data and Methodology",
        "This section outlines our data source, dataset structure, and the workflow we used to move from map selection to reviewed outputs."
    )
    methods_col, data_col = st.columns(2)
    with methods_col:
        st.markdown('<div class="sb-card sb-slide-card">', unsafe_allow_html=True)
        st.markdown("### Workflow")
        st.markdown(
            "1. Select a study area using the interactive San Diego map.\n"
            "2. Generate grid points and request Street View imagery.\n"
            "3. Run the trained YOLO model on the retrieved images.\n"
            "4. Review, validate, and correct detections manually.\n"
            "5. Export annotations for analysis, visualization, and reporting."
        )
        st.markdown("### Analytical Approach")
        st.write(
            "Our approach combines geospatial sampling with computer vision so that image-level "
            "detections can be tied back to a specific study area. We designed the workflow to "
            "support both automated inference and human validation."
        )
        st.markdown("</div>", unsafe_allow_html=True)
    with data_col:
        st.markdown('<div class="sb-card sb-slide-card">', unsafe_allow_html=True)
        st.markdown("### Data and Database Management")
        st.markdown(
            f"- Primary imagery source: Google Street View.\n"
            f"- Original 4-class labeled corpus: `{original_records}` images with `{sum(total_boxes.values()):,}` mapped boxes.\n"
            f"- Cleaned working corpus: `{clean_images}` images after curation and regrouping.\n"
            f"- Final balanced dataset used in the project pipeline: `{final_balanced_images}` images with `{final_balanced_boxes:,}` total boxes.\n"
            f"- Final balanced split: `{balanced_dataset_summary.get('split_image_counts', {}).get('train', 0)}` / "
            f"`{balanced_dataset_summary.get('split_image_counts', {}).get('val', 0)}` / `{balanced_dataset_summary.get('split_image_counts', {}).get('test', 0)}`.\n"
            f"- Detection classes worked across the project: `homeless_tent`, `homeless_cart`, `homeless_bicycle`, `homeless_person`.\n"
            f"- Data and model artifacts are stored under `datasets/`, `runs/`, and export files.\n"
            f"- Main tools: Streamlit, YOLOv8, PIL, pandas, and custom geospatial sampling utilities."
        )
        st.markdown("### Project Context")
        st.write(
            "This work sits at the intersection of urban analytics, geospatial data collection, "
            "and computer vision. We treat visual indicators as partial signals that require "
            "careful interpretation within a broader social context."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    _render_slide_header(
        "04",
        "Model Development and Technical Outcomes",
        "This section summarizes what we built, how the system was improved, and the main technical outcomes of the project."
    )
    st.markdown('<div class="sb-section sb-card sb-slide-card">', unsafe_allow_html=True)
    st.markdown("### Key Outcomes")
    st.markdown(
        f"- We developed a usable Streamlit application rather than limiting the project to offline notebooks or scripts.\n"
        f"- We worked across multiple dataset stages: `{original_records}` original labeled records, `{clean_images}` cleaned images, and a final balanced dataset of `{final_balanced_images}` images.\n"
        f"- We generated `{augmentation_images}` augmented training images to increase coverage for harder classes and strengthen the final training set.\n"
        f"- We selected `homeless4_accuracy_v5_finetune` as the strongest 4-class checkpoint for deployment in the Streamlit workflow.\n"
        f"- The active model is a YOLOv8s fine-tuned detector trained for `{training_summary.get('epochs_recorded', 0)}` recorded epochs with best validation mAP50 at epoch `{training_summary.get('best_map50_epoch', 'N/A')}`.\n"
        f"- On the held-out 4-class test split, the selected model reached precision `{test_metrics['precision']:.3f}`, recall `{test_metrics['recall']:.3f}`, mAP50 `{test_metrics['map50']:.3f}`, and mAP50-95 `{test_metrics['map50_95']:.3f}`.\n"
        f"- We built a reproducible evaluation workflow with confusion matrices, PR curves, recall curves, and F1 curves.\n"
        f"- We combined automated detection with manual annotation review so outputs can be validated before export.\n"
        f"- We organized the project in a form that supports presentation, reporting, and future extension."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    _render_slide_header(
        "05",
        "Analytics and Visualization Results",
        "The visualizations and performance metrics below reflect the active `homeless4_accuracy_v5_finetune` model currently used in the application."
    )
    results_cols = st.columns(4)
    with results_cols[0]:
        _render_metric_card("Original Records", f"{original_records:,}")
    with results_cols[1]:
        _render_metric_card(
            "Final Dataset Images",
            f"{final_balanced_images:,}",
        )
    with results_cols[2]:
        _render_metric_card(
            "Augmented Train Images",
            f"{augmentation_images:,}",
        )
    with results_cols[3]:
        _render_metric_card(
            "Classes Worked",
            f"{len(class_names)}",
        )

    class_col, support_col = st.columns([1, 1.2])
    with class_col:
        _render_simple_card(
            "Target Classes",
            """
            <div class="sb-class-list">
                <div class="sb-class-item">
                    <span class="sb-class-dot" style="background:#FFB347;"></span>
                    <span class="sb-code-pill">homeless_tent</span>
                </div>
                <div class="sb-class-item">
                    <span class="sb-class-dot" style="background:#A8FF4B;"></span>
                    <span class="sb-code-pill">homeless_cart</span>
                </div>
                <div class="sb-class-item">
                    <span class="sb-class-dot" style="background:#FF4B4B;"></span>
                    <span class="sb-code-pill">homeless_bicycle</span>
                </div>
                <div class="sb-class-item">
                    <span class="sb-class-dot" style="background:#FF6FAE;"></span>
                    <span class="sb-code-pill">homeless_person</span>
                </div>
            </div>
            """,
        )
    with support_col:
        _render_simple_card(
            "Evaluation Snapshot",
            f"""
            <div class="sb-metric-list">
                <div class="sb-metric-row">
                    <span class="sb-metric-name">Test precision</span>
                    <span class="sb-metric-chip">{test_metrics['precision']:.3f}</span>
                </div>
                <div class="sb-metric-row">
                    <span class="sb-metric-name">Test recall</span>
                    <span class="sb-metric-chip">{test_metrics['recall']:.3f}</span>
                </div>
                <div class="sb-metric-row">
                    <span class="sb-metric-name">Test mAP50</span>
                    <span class="sb-metric-chip">{test_metrics['map50']:.3f}</span>
                </div>
                <div class="sb-metric-row">
                    <span class="sb-metric-name">Test mAP50-95</span>
                    <span class="sb-metric-chip">{test_metrics['map50_95']:.3f}</span>
                </div>
            </div>
            <div class="sb-support-note">These model metrics come from the held-out 4-class test evaluation for <code>homeless4_accuracy_v5_finetune</code>. The cards above summarize broader project effort through the original corpus, cleaned dataset, balanced expansion, and augmentation totals.</div>
            """,
        )

    artifact_col1, artifact_col2 = st.columns(2)
    with artifact_col1:
        results_plot_path = artifact_paths.get("results_plot")
        if results_plot_path:
            _render_artifact_image("Training Results Overview", Path(results_plot_path))
        confusion_path = artifact_paths.get("confusion_matrix_normalized")
        if confusion_path:
            _render_artifact_image("Normalized Confusion Matrix", Path(confusion_path))
    with artifact_col2:
        pr_curve_path = artifact_paths.get("pr_curve")
        if pr_curve_path:
            _render_artifact_image("Precision-Recall Curve", Path(pr_curve_path))
        recall_curve_path = artifact_paths.get("recall_curve")
        if recall_curve_path:
            _render_artifact_image("Recall Curve", Path(recall_curve_path))
        f1_curve_path = artifact_paths.get("f1_curve")
        if f1_curve_path:
            _render_artifact_image("F1 Curve", Path(f1_curve_path))

    _render_slide_header(
        "06",
        "Discussion, Significance, and SWOT",
        "In our final discussion, we emphasize both the technical value of the system and the limits that should be acknowledged in presentation and reporting."
    )
    st.markdown('<div class="sb-section sb-card sb-slide-card">', unsafe_allow_html=True)
    st.markdown("### Discussion")
    st.markdown(
        "- We addressed a real social and urban issue through a technically rigorous workflow.\n"
        "- We integrated data engineering, modeling, visualization, and interface design within one capstone project.\n"
        "- We treated model limitations transparently while still demonstrating meaningful technical progress.\n"
        "- The resulting outputs can support future mapping, reporting, and policy-oriented analysis."
    )
    st.markdown("</div>", unsafe_allow_html=True)
    swot_cols = st.columns(4)
    swot_content = [
        (
            "Strengths",
            "Interactive workflow, reproducible pipeline, real evaluation artifacts, and a presentation-ready app.",
        ),
        (
            "Weaknesses",
            "Class imbalance, limited visibility in Street View imagery, and possible ambiguity in image interpretation.",
        ),
        (
            "Opportunities",
            "Add more labeled data, improve class coverage, expand to temporal comparisons, and connect outputs to mapping dashboards.",
        ),
        (
            "Threats",
            "Privacy concerns, bias in imagery coverage, uneven data support, and risk of overinterpreting visual proxies.",
        ),
    ]
    for col, (title, body) in zip(swot_cols, swot_content):
        with col:
            st.markdown('<div class="sb-card">', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            st.write(body)
            st.markdown("</div>", unsafe_allow_html=True)

    _render_slide_header(
        "07",
        "Final Deliverables",
        "These deliverables align directly with what we will present on capstone presentation day."
    )
    st.markdown('<div class="sb-section sb-card sb-slide-card">', unsafe_allow_html=True)
    st.markdown("### Final Deliverables")
    st.markdown(
        "- Group project website integrated into Streamlit.\n"
        "- Final presentation aligned with the group report structure.\n"
        "- Short project introduction video.\n"
        "- Final capstone report covering the problem statement, methods, results, discussion, and conclusion."
    )
    st.markdown("</div>", unsafe_allow_html=True)
