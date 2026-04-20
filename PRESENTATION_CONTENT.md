# StreetBridge Presentation Content

## Slide 1 - Project Introduction
**Title:** StreetBridge: Detecting Visible Homelessness Indicators from Google Street View

**Subtitle:** A computer vision workflow for Street View image sampling, object detection, review, and export

**Main points:**
- StreetBridge is an end-to-end project that uses Google Street View imagery and object detection to identify visible homelessness-related indicators across selected areas in San Diego County.
- The system combines data preparation, model training, evaluation, and a Streamlit-based application for interactive review and export.
- Our goal was not just to train a model, but to build a usable workflow that supports data collection, automated screening, manual correction, and presentation-ready outputs.

**Suggested visual:**
- Use a clean app screenshot or title collage from the Streamlit app.
- Optional image path: `runs/detect/homeless4_accuracy_v5_finetune/val_batch0_pred.jpg`

---

## Slide 2 - Team Introduction
**Title:** Team

**Main points:**
- **Pranjal Patel**: Project framing, presentation narrative, website content, and capstone reporting.
- **Palash Suryawanshi**: Model pipeline, dataset workflow, evaluation artifacts, and app integration.
- The project was developed as a combination of research, engineering, and user-facing product design.

**Suggested visual:**
- Use team photos side by side.
- Image paths:
  - `assets/team/pranjal_patel.jpeg`
  - `assets/team/palash_suryawanshi.jpeg`

---

## Slide 3 - Problem Statement
**Title:** Why This Project Matters

**Main points:**
- Homelessness is a visible and spatially distributed urban issue, but traditional reporting is slow, expensive, and difficult to scale.
- Google Street View offers a historical and geographically rich image source that can support large-area visual analysis.
- We explored whether computer vision can help detect visible indicators such as tents, carts, bicycles, and people in Street View scenes.
- The broader idea is to support faster screening and structured review, not to replace human judgment.

**Suggested visual:**
- Use one example Street View image with visible objects of interest.
- Example image path: `annotated_images/Gst-Part1 Annotation-CSV/2022_10_qsrivFqc-a7GAyaoT0f7Qg_180.jpeg`

---

## Slide 4 - Project Goals and Scope
**Title:** What StreetBridge Was Designed To Do

**Main points:**
- Fetch Street View images from a selected map region.
- Detect homelessness-related indicators using a trained YOLOv8 model.
- Let users review detections and manually correct annotations.
- Export results as CSV and ZIP packages for downstream analysis.
- Create a repeatable training and evaluation pipeline that can be improved over time.

**Suggested visual:**
- Show a simple workflow diagram: Select Area -> Fetch Images -> Detect -> Review -> Export.
- Optional supporting screenshot: `runs/detect/homeless4_accuracy_v5_finetune/results.png`

---

## Slide 5 - Dataset and Labels
**Title:** Dataset Overview

**Main points:**
- The original YOLO dataset summary shows **462 labeled records**.
- The original mapped classes were:
  - `homeless_tent`: **884**
  - `homeless_cart`: **270**
  - `homeless_person`: **283**
  - `homeless_bicycle`: **56**
- The original split counts were:
  - Train: **323 images**
  - Validation: **92 images**
  - Test: **47 images**
- This class distribution immediately showed an imbalance problem, especially for bicycles and carts.

**Suggested visual:**
- Create a bar chart from the original class counts.
- Source file for numbers: `datasets/yolo_homeless_4class/summary.json`

---

## Slide 6 - Data Cleaning and Label Policy
**Title:** Improving Data Quality Before Training

**Main points:**
- We created a cleaner dataset version by converting annotations with a stricter label policy.
- Ambiguous labels such as `Homeless`, `Homelessss`, and low-consistency bicycle labels were excluded in the clean pipeline.
- The clean grouped dataset contained:
  - **935 records**
  - **605 grouped image sets**
  - Train: **654 images**
  - Validation: **187 images**
  - Test: **94 images**
- The clean pipeline focused on stronger supervision for:
  - `homeless_tent`
  - `homeless_cart`
  - `homeless_person`

**Suggested visual:**
- Before/after table: original labels vs cleaned labels.
- Source file for counts and policy: `datasets/yolo_homeless_4class_clean/summary.json`

---

## Slide 7 - Class Balancing and Augmentation
**Title:** Handling Class Imbalance

**Main points:**
- After cleaning, we built a balanced training dataset to improve weaker categories.
- Augmentation targeted images containing `homeless_cart` and `homeless_person`.
- Augmentation methods included:
  - horizontal flip
  - color adjustment
  - lighting adjustment
- The balanced dataset expanded training from **654** to **1,593** training images.
- It added **939 augmented images** generated from **313 source images**.
- Final total box counts became:
  - `homeless_tent`: **3,194**
  - `homeless_cart`: **1,442**
  - `homeless_person`: **1,171**

**Suggested visual:**
- Show a bar chart comparing class counts before and after balancing.
- Source file for numbers: `datasets/yolo_homeless_4class_clean_balanced/summary.json`

---

## Slide 8 - Training Pipeline
**Title:** Model Training Setup

**Main points:**
- The reproducible training pipeline is scripted in `scripts/run_accuracy_pipeline.sh`.
- The main model configuration used:
  - Base model: `yolov8s.pt`
  - Epochs: **60**
  - Image size: **832**
  - Batch size: **8**
  - Optimizer: **AdamW**
  - Seed: **42**
- The pipeline automatically:
  - rebuilds the clean dataset
  - rebuilds the balanced dataset
  - trains the model
  - evaluates on the held-out test split
- This made the project reproducible and easier to iterate on.

**Suggested visual:**
- Use a pipeline diagram or a screenshot of the training artifact plot.
- Suggested image path: `runs/detect/homeless4_accuracy_v5_finetune/results.png`

---

## Slide 9 - Model Training History
**Title:** Training Progress Across Experiments

**Main points:**
- We tracked multiple YOLO experiments, including:
  - baseline model
  - accuracy-focused retraining
  - finetuned model
- On the held-out 4-class test evaluation:
  - **Baseline**: best mAP50 **0.5065**, best mAP50-95 **0.2258**
  - **Accuracy v3**: best mAP50 **0.4924**, best mAP50-95 **0.2164**
  - **Accuracy v5 finetune**: best mAP50 **0.5013**, best mAP50-95 **0.2161**
- The finetuned run achieved the strongest recall among the compared runs at **0.5764**.
- The best epoch for the finetuned run occurred at **epoch 29**.

**Suggested visual:**
- Best option: line chart of precision, recall, mAP50, and mAP50-95 across epochs using `results.csv`.
- Source file: `runs/detect/homeless4_accuracy_v5_finetune/results.csv`
- Supporting evaluation reports:
  - `runs/eval/homeless4_baseline_test_4class/evaluation_report.md`
  - `runs/eval/homeless4_accuracy_v3_test_4class/evaluation_report.md`
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/evaluation_report.md`

---

## Slide 10 - Final Evaluation Results
**Title:** How Well the Model Performed

**Main points:**
- Final evaluated test set size: **47 images**
- Ground-truth support on the 4-class test split:
  - `homeless_tent`: **86 boxes**
  - `homeless_cart`: **33 boxes**
  - `homeless_bicycle`: **6 boxes**
  - `homeless_person`: **38 boxes**
- Finetuned model best metrics:
  - Precision: **0.4832**
  - Recall: **0.5764**
  - mAP50: **0.5013**
  - mAP50-95: **0.2161**
- The evaluation also showed that bicycle support was extremely low, so results for that class are noisy and less reliable.

**Suggested visual:**
- Use the confusion matrix and one curve side by side.
- Image paths:
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/confusion_matrix.png`
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/BoxPR_curve.png`

---

## Slide 11 - Qualitative Results
**Title:** Example Predictions and Visual Error Analysis

**Main points:**
- Quantitative metrics are useful, but visual inspection was critical for understanding model behavior.
- The model was better at detecting larger or more distinct tents and carts than subtle or heavily occluded objects.
- Small objects, crowded scenes, and ambiguous human presence remained challenging.
- Visual review helped us identify where manual correction inside the app still matters.

**Suggested visual:**
- Use 2 to 4 prediction examples on a single slide.
- Recommended image paths:
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/val_batch0_pred.jpg`
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/val_batch1_pred.jpg`
  - `runs/eval/homeless4_accuracy_v5_finetune_test_4class/val_batch2_pred.jpg`

---

## Slide 12 - Application Demo Workflow
**Title:** Streamlit Application Workflow

**Main points:**
- The Streamlit app turns the model into a usable demo tool.
- Users can:
  - select an area on a map
  - fetch Street View images for one or multiple headings
  - run automatic detection
  - review grouped detections
  - draw manual boxes or remove incorrect detections
  - export final annotations as CSV or ZIP
- This makes the project stronger than a model-only submission because it demonstrates a complete user workflow.

**Suggested visual:**
- Use screenshots from the app tabs if available, or show one annotated detection view.
- Supporting file reference: `app.py`

---

## Slide 13 - Project Structure and Engineering Design
**Title:** Repository Structure

**Main points:**
- The project is organized into clear functional modules:
  - `app.py`: Streamlit frontend
  - `gsv_fetcher.py`: Google Street View image retrieval
  - `detector.py`: YOLO model loading and inference
  - `annotator.py`: bounding-box drawing and review support
  - `exporter.py`: CSV and ZIP export
  - `scripts/`: dataset preparation, balancing, training, and evaluation automation
  - `datasets/`: prepared YOLO datasets
  - `runs/`: training and evaluation outputs
- This structure separates user interface, model logic, data preparation, and experiment artifacts.
- That separation makes the project easier to maintain, debug, and present.

**Suggested visual:**
- Recreate the repository tree as a clean diagram.
- Source reference: `README.md`

---

## Slide 14 - Video Demo
**Title:** Video Demo

**Main points to say while the video plays:**
- The demo shows the complete user flow from map selection to detection review and export.
- It highlights how StreetBridge can move from raw location input to structured, reviewable annotation output.
- While the model is useful for automatic screening, the app also preserves human review for quality control.

**Suggested visual:**
- Embed your screen recording or a clickable screenshot with a play icon.
- Best screenshot source: take one from the running app while showing detections and review mode.

---

## Slide 15 - Limitations and Lessons Learned
**Title:** What We Learned

**Main points:**
- Class imbalance remained a major challenge, especially for rare classes like `homeless_bicycle`.
- Street View scenes contain occlusion, distance variation, and visual ambiguity, which makes object detection difficult.
- Detection of visible indicators does not directly measure homelessness itself, so outputs must be interpreted carefully.
- The project showed that data quality and label policy can matter as much as model selection.
- Building a review workflow was important because automated predictions still need human verification.

**Suggested visual:**
- Use the normalized confusion matrix or a false-positive example.
- Suggested image path: `runs/eval/homeless4_accuracy_v5_finetune_test_4class/confusion_matrix_normalized.png`

---

## Slide 16 - Future Work and Closing
**Title:** Future Improvements

**Main points:**
- Expand the dataset with more balanced examples for rare classes.
- Improve annotation consistency and class definitions.
- Explore stronger model variants and threshold tuning.
- Add richer geospatial summarization and hotspot visualization for exported detections.
- Extend the app with historical comparisons and dashboard-style analytics.

**Closing line:**
- StreetBridge demonstrates that computer vision can support structured urban visual analysis when paired with careful data preparation, transparent evaluation, and human review.

**Suggested visual:**
- Use a final collage of app output, prediction samples, and evaluation artifacts.

---

## Optional Backup Slide - Model Comparison
**Title:** YOLO vs Faster R-CNN Comparison

**Main points:**
- The repository includes a comparison workflow between YOLO and Faster R-CNN outputs.
- In the saved comparison summary, YOLO produced detections across the sampled images, while Faster R-CNN often produced no detections in the compared examples.
- This supports the project decision to center the final application around YOLO.

**Suggested visual:**
- Use one side-by-side comparison pair.
- Example image paths:
  - `Faster_R_CNN/comparison_outputs/frcnn/17thStreet_Annotation_CSV__2016_11_duKxSuqkju_KnvjpKO6yEg_270.jpeg`
  - `Faster_R_CNN/comparison_outputs/yolo/17thStreet_Annotation_CSV__2016_11_duKxSuqkju_KnvjpKO6yEg_270.jpeg`

