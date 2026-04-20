# StreetBridge: GSV Homelessness Indicator Detection

StreetBridge is a Google Street View and YOLOv8 project for detecting visible homelessness-related indicators in San Diego County imagery. This repository supports the app, model inference, evaluation, and the scripts used to prepare and train the model.

## What This Public Repo Includes

- the Streamlit app
- the YOLO-based detector
- training and evaluation scripts
- documentation for running the app on a new device

## What This Public Repo Does Not Include

To protect privacy and keep the repo lightweight, the public version should not include:

- raw Google Street View images used for training or prediction
- annotation image folders
- dataset exports
- generated prediction preview images
- private API keys or local secret files

If you clone this project, you can still run the app and use the public model, but you will need your own Google Street View API key.

## Repository Structure

```text
StreetBridge/
├── app.py
├── annotator.py
├── detector.py
├── exporter.py
├── gsv_fetcher.py
├── project_overview.py
├── requirements.txt
├── scripts/
│   ├── prepare_yolo_dataset.py
│   ├── build_balanced_dataset.py
│   ├── run_accuracy_pipeline.sh
│   ├── run_evaluation_pipeline.sh
│   └── summarize_evaluation.py
├── assets/
└── runs/
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/palash-suryawanshi/StreetBridge.git
cd YOUR_REPO
```

### 2. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add your Google Street View API key

You need a Google Street View Static API key to fetch images in the app.

You can provide it in either of these ways:

Option A: environment variable

macOS / Linux:

```bash
export GOOGLE_STREET_VIEW_API_KEY="your_api_key_here"
```

Windows PowerShell:

```powershell
$env:GOOGLE_STREET_VIEW_API_KEY="your_api_key_here"
```

Option B: Streamlit secrets

Create a local file at `.streamlit/secrets.toml`:

```toml
GOOGLE_STREET_VIEW_API_KEY = "your_api_key_here"
```

Do not commit `.streamlit/secrets.toml`.

### 5. Run the Streamlit app

```bash
streamlit run app.py
```

After the app starts:

- select an area on the map
- fetch Street View images with your API key
- run detection
- review or correct annotations
- export CSV or ZIP results

## Model Setup

The app supports a public trained model and also supports fallback behavior when custom weights are missing.

### Default model selection order

The detector looks for weights in this order:

1. `STREETBRIDGE_MODEL_PATH`
2. `runs/detect/homeless4_accuracy_v5_finetune/weights/best.pt`
3. `runs/detect/homeless4_accuracy_v3/weights/best.pt`
4. `runs/detect/homeless4_accuracy_v2/weights/best.pt`
5. `runs/detect/homeless4_balanced_v1/weights/best.pt`
6. `runs/detect/homeless4_baseline/weights/best.pt`
7. fallback to `yolov8n.pt`

### Use a specific model file

If you want to point the app to a specific trained model:

macOS / Linux:

```bash
export STREETBRIDGE_MODEL_PATH="/full/path/to/best.pt"
streamlit run app.py
```

Windows PowerShell:

```powershell
$env:STREETBRIDGE_MODEL_PATH="C:\full\path\to\best.pt"
streamlit run app.py
```

### Important note about fallback behavior

If no custom model is available, the app can still run with the YOLO fallback model, but accuracy will be lower and some classes are not reliably detected.

Fallback cannot reliably detect:

- `homeless_tent`
- `homeless_cart`

## Reproducible Use on Another Device

If you want another person to clone this repo and get the same inference behavior:

1. share the same code version
2. share the same public model weight file
3. install the same dependencies
4. set `STREETBRIDGE_MODEL_PATH` to that model file, or keep it in the expected `runs/detect/.../weights/best.pt` location

This is the recommended way to keep model behavior consistent across devices without sharing private training images.

## Running Evaluation

If the public model weights and safe evaluation artifacts are included, users can rerun evaluation on a device that also has access to the dataset used for evaluation.

Recommended command:

```bash
bash scripts/run_evaluation_pipeline.sh
```

Useful overrides:

```bash
RUN_NAME=homeless4_accuracy_v5_finetune bash scripts/run_evaluation_pipeline.sh
```

```bash
MODEL_PATH=/full/path/to/best.pt RUN_NAME=my_model_eval bash scripts/run_evaluation_pipeline.sh
```

This produces:

- confusion matrices
- PR, precision, recall, and F1 curves
- `evaluation_report.json`
- `evaluation_report.md`

Note: if the dataset is private and not included in the repo, others cannot fully reproduce training or evaluation from scratch.

## Training Pipeline

The project also includes scripts for rebuilding the dataset, balancing classes, training, and evaluation.

Main training command:

```bash
bash scripts/run_accuracy_pipeline.sh
```

By default this script:

- rebuilds the clean grouped-split dataset
- rebuilds the balanced dataset
- trains using `yolov8s.pt`
- evaluates on the held-out test split

Default training configuration:

- model: `yolov8s.pt`
- epochs: `60`
- image size: `832`
- batch size: `8`
- optimizer: `AdamW`
- seed: `42`

Useful overrides:

```bash
DEVICE=mps bash scripts/run_accuracy_pipeline.sh
```

```bash
DEVICE=0 RUN_NAME=homeless4_accuracy_gpu EPOCHS=100 bash scripts/run_accuracy_pipeline.sh
```

Important: these training steps require the original annotated image data, which should remain private and is not required just to run the app.

## App Workflow

### Tab 1: Select Area

- Draw a rectangle on the map, or enter coordinates manually.
- The app estimates the number of Street View images that may be requested.

### Tab 2: Fetch and Detect

- Choose one or more Street View headings in the sidebar.
- Fetch images from the selected area.
- Run the detector on all fetched views.

### Tab 3: Review and Annotate

- Review automatic detections image by image.
- Add manual annotations as bounding boxes or label-only corrections.
- Remove individual annotations or clear all annotations for an image.

### Tab 4: Export

- Download CSV results
- Download a ZIP of annotated images and the CSV

## API and Usage Notes

- Enable the Google Street View Static API in Google Cloud before using the app.
- The app can fetch multiple headings per point, so API usage increases with wider heading selection.
- Use the sidebar safety cap to control fetch volume and demo cost.

## Privacy Notes

To keep this project safe to publish:

- keep raw Street View images out of GitHub
- keep training datasets out of GitHub
- keep prediction preview images out of GitHub
- keep `.streamlit/secrets.toml` and `.env` files out of GitHub
- only publish model weights and metrics if you are comfortable making them public

## Ethical Notes

This project should be used carefully. Street-level detection of vulnerable populations can create privacy and misuse risks. Any real deployment should account for:

- privacy and data minimization
- local legal and ethical review
- Google Maps Platform usage terms
- human oversight for interpretation and downstream decisions

## License

MIT
