# civicLy

civicLy is a lightweight waste analysis and recycling workflow app:
- Upload an image and run waste classification with a fine-tuned `microsoft/resnet-50`.
- View top predictions in a result page with a pie chart and routing map.
- Create and buy marketplace listings for recyclable materials.

## Tech Stack

- Backend: Go (`net/http`, SQLite via `modernc.org/sqlite`)
- Frontend: HTML/CSS/vanilla JS
- ML inference: Python (`torch`, `transformers`, `Pillow`)

## Project Structure

- `main.go`: API + static file server + SQLite persistence
- `predict.py`: Runs model inference on uploaded images
- `train.py`: Training script for the classifier
- `waste_classifier_model/`: Saved model/config artifacts
- `index.html`, `sell.html`, `buy.html`, `result.html`: UI pages
- `app.js`, `style.css`: Frontend behavior and styles

## Prerequisites

- Go 1.22+
- Python 3.10+ (recommended)
- Internet access for first-time Python package/model downloads

## Setup

1. Create and prepare a Python environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install transformers Pillow
```

2. Optional sanity check:

```bash
.venv/bin/python -c "import torch, transformers, PIL; print('OK')"
```

## Run the App

Use the Python environment explicitly so inference always works:

```bash
CIVICLY_PYTHON=.venv/bin/python3 go run main.go
```

Then open:

- `http://localhost:8080`

## Model Inference Notes

- Backend endpoint `POST /api/predict` writes the uploaded image to a temp file and calls:
  - `predict.py --image <path> --top-k 3`
- The backend auto-resolves Python in this order:
  - `CIVICLY_PYTHON`
  - `.venv/bin/python3`
  - `venv/bin/python3`
  - `python3`

## Retraining the Model

`train.py` fine-tunes `microsoft/resnet-50` and writes outputs into `waste_classifier_model/`.

Before training, update `DATASET_PATH` in `train.py` to your dataset location.

Run:

```bash
.venv/bin/python train.py
```

## API Overview

- `GET /api/health`
- `POST /api/predict`
- `GET /api/predictions/{id}`
- `GET /api/listings`
- `POST /api/listings`
- `POST /api/listings/{id}/buy`

## Troubleshooting

- `model runtime unavailable`:
  - Python exists but required packages are missing.
  - Install `torch`, `transformers`, `Pillow` in the selected Python env.
- `Analyze failed: model inference failed`:
  - Check backend logs for the exact `inference error` line.
- If Go build cache permission issues appear:
  - Run with `GOCACHE=/tmp/go-build`.
