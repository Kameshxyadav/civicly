# civicLy

civicLy is a lightweight waste analysis and recycling workflow app:
- Upload an image and run waste classification with a fine-tuned `microsoft/resnet-50`.
- View top predictions in a result page with a pie chart and routing map.
- Create and buy marketplace listings for recyclable materials.
- User accounts with session-based authentication.
- Seller notifications when a listing is purchased.
- Admin dashboard to manage users and listings.

## Tech Stack

- Backend: Go (`net/http`, SQLite via `modernc.org/sqlite`)
- Frontend: HTML/CSS/vanilla JS
- ML inference: Python (`torch`, `transformers`, `Pillow`)
- Geocoding: OpenStreetMap Nominatim (reverse geocode for human-readable locations)

## Project Structure

- `main.go`: API + static file server + SQLite persistence + auth + notifications + admin
- `predict.py`: Runs model inference on uploaded images
- `train.py`: Training script for the classifier
- `waste_classifier_model/`: Saved model/config artifacts
- `index.html`: Home page with image upload
- `login.html`: Login page
- `register.html`: Registration page
- `sell.html`: Create a listing (requires login)
- `buy.html`: Marketplace — browse and purchase listings
- `result.html`: Waste analysis results with chart and map
- `admin.html`: Admin dashboard (users + all listings)
- `app.js`: Frontend behavior (auth, notifications, marketplace, admin)
- `style.css`: Styles

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

## User Authentication

- Register a new account at `/register.html` or click Login in the topbar.
- Sessions are stored as HttpOnly cookies (7-day expiry).
- Selling requires login — unauthenticated users are redirected to the login page.
- After login, users are redirected back to the page they came from.

### Default Admin Account

An admin user is seeded on first run:

- **Username:** `admin`
- **Password:** `admin123`

Login as admin and visit `/admin.html` to see all users and all listings.

## Notifications

- When a listing is purchased, the seller receives a notification.
- The notification bell in the topbar shows an unread count badge.
- Click the bell to view notifications — they are automatically marked as read.

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

### Public

- `GET  /api/health`
- `POST /api/predict`
- `GET  /api/predictions/{id}`
- `GET  /api/listings`

### Auth

- `POST /api/register` — create account + session
- `POST /api/login` — authenticate + session
- `POST /api/logout` — destroy session
- `GET  /api/me` — current user + unread notification count

### Listings (auth required for write)

- `POST /api/listings` — create listing (login required)
- `POST /api/listings/{id}/buy` — purchase a listing
- `POST /api/listings/{id}/sold` — mark own listing as sold

### Notifications (auth required)

- `GET  /api/notifications` — current user's notifications
- `POST /api/notifications/read` — mark all as read

### Admin (admin role required)

- `GET  /api/admin/users` — all users with passwords
- `GET  /api/admin/listings` — all listings (open + sold)

## Troubleshooting

- `model runtime unavailable`:
  - Python exists but required packages are missing.
  - Install `torch`, `transformers`, `Pillow` in the selected Python env.
- `Analyze failed: model inference failed`:
  - Check backend logs for the exact `inference error` line.
- If Go build cache permission issues appear:
  - Run with `GOCACHE=/tmp/go-build`.
