# 🌷 Flower Classifier

A full-stack image classifier that identifies flowers from a photo. Upload an image in the
browser and a fine-tuned **MobileNetV2** model tells you which of 5 flowers it is, with
the probability for every class.

It recognizes: **daisy · dandelion · roses · sunflowers · tulips**

| Layer | Tech |
|---|---|
| Model | TensorFlow 2.21 / Keras 3.13, MobileNetV2 transfer learning + fine-tuning |
| Backend | Python, Flask, Flask-CORS, Pillow, NumPy |
| Frontend | Next.js (App Router), React, TypeScript, Tailwind CSS |

---

## How it works

```
 Browser (Next.js, :3000)                     Flask API (:5000)
 ┌────────────────────────┐   POST /predict   ┌──────────────────────────────────┐
 │ drag & drop / browse    │  multipart form  │ validate type & size (≤ 5 MB)     │
 │ client-side validation  │ ───────────────► │ PIL → RGB → 224×224 (bilinear)    │
 │ preview + "Identify"    │                  │ float32 array, shape (1,224,224,3)│
 │                         │ ◄─────────────── │ model.predict → 5 softmax probs   │
 │ result card + prob bars │       JSON       │ sort, apply confidence threshold  │
 └────────────────────────┘                   └──────────────────────────────────┘
```

Uploaded images are processed in memory and are never written to disk.

---

## The model: MobileNetV2 transfer learning

### Why MobileNetV2?

[MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al., Google, 2018) is a
convolutional neural network designed to be **small and fast**, so it can run on phones.
It is built from *inverted residual blocks*. Each block:

1. **expands** the channels with a cheap 1×1 convolution,
2. filters each channel separately with a **depthwise** 3×3 convolution, which costs far
   less compute than a normal convolution, and
3. **projects** back down to a few channels with a *linear bottleneck*, using no
   activation so information isn't lost in the narrow layer.

That gives good ImageNet accuracy with only about 2.3 M parameters. VGG16 has about 138 M.

### Transfer learning

Training a CNN from scratch needs millions of images. Instead, this project starts from
MobileNetV2 **pre-trained on ImageNet** (1.4 M images, 1,000 classes). Its convolutional
layers already detect edges, textures, petals and shapes. Only a new classification head
has to be learned for the 5 flower classes.

Training ran in two stages:

1. **Feature extraction:** freeze the whole MobileNetV2 base and train only the new head.
2. **Fine-tuning:** unfreeze the **top 31 of the base's 154 layers** and keep training
   with a low learning rate, so the high-level features adapt to flowers. The lower,
   more general layers stay frozen.

### Architecture

Inspected from `backend/flowers.keras`:

| # | Layer | Output shape | Params | Purpose |
|---|---|---|---|---|
| 1 | Input | (224, 224, 3) | 0 | RGB image, pixels 0–255 |
| 2 | Data augmentation (`Sequential`) | (224, 224, 3) | 0 | `RandomFlip("horizontal")`, `RandomRotation(0.1)`, `RandomZoom(0.1)`. **Active only during training.** |
| 3 | `Rescaling(1/127.5, offset=-1)` | (224, 224, 3) | 0 | Maps pixels from [0, 255] to [-1, 1], the range MobileNetV2 expects |
| 4 | MobileNetV2 base (ImageNet weights, no top) | (7, 7, 1280) | 2,257,984 | Feature extractor; top 31 layers fine-tuned |
| 5 | `GlobalAveragePooling2D` | (1280) | 0 | Averages each 7×7 feature map into one number |
| 6 | `Dropout(0.2)` | (1280) | 0 | Regularization against overfitting (training only) |
| 7 | `Dense(5, softmax)` | (5) | 6,405 | One probability per flower class |

**Total:** 2,264,389 parameters (8.6 MB). 1,532,805 trainable, 731,584 frozen.

### Inference contract

Because augmentation and rescaling are **inside** the model, the API sends it raw pixels:

- **Input:** `float32` tensor of shape `(1, 224, 224, 3)` with values **0–255**.
  Do **not** divide by 255 or call `preprocess_input`, or the image gets scaled twice.
- **Resize:** bilinear, the same method `image_dataset_from_directory` used to load the
  training data.
- **Output:** 5 softmax probabilities in the order stored in
  [`backend/classes.json`](backend/classes.json):
  `["daisy", "dandelion", "roses", "sunflowers", "tulips"]`.
- **"Not sure" threshold:** if the top probability is below `CONFIDENCE_THRESHOLD`
  (default `0.6`), the API returns `"sure": false` and the UI says the image might not be
  one of the 5 known flowers.

> **Limitation:** a softmax classifier always picks one of its 5 classes, even for a photo
> of a cat. The confidence threshold is a simple guard against that, not a real
> "unknown" detector.

---

## Project structure

```
backend/                 Flask API (port 5000)
  app.py                 loads the model once; /predict and /health
  classes.json           class names in model output order
  flowers.keras          trained MobileNetV2 model (Keras 3 format)
  requirements.txt       pinned TensorFlow / Keras versions
frontend/                Next.js app (port 3000)
  app/page.tsx           upload UI, preview, results
  app/layout.tsx         page shell and metadata
  .env.example           NEXT_PUBLIC_API_URL
README.md
```

---

## Getting started

### Prerequisites

- **Python 3.10–3.12** (TensorFlow 2.21 doesn't support every newer Python release)
- **Node.js 20+**

### 1. Backend (Flask)

```bash
cd backend

# Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

# Install dependencies (TensorFlow is large, so this takes a few minutes)
pip install -r requirements.txt

# Start the API on http://localhost:5000
python app.py
```

Wait for `Model loaded.` before sending requests.

`tensorflow==2.21.0` and `keras==3.13.2` are pinned to the exact versions the model was
trained and saved with, so `flowers.keras` loads exactly as it was saved.

To change the confidence threshold (default `0.6`):

```bash
# Windows PowerShell
$env:CONFIDENCE_THRESHOLD = "0.75"; python app.py
# macOS / Linux
CONFIDENCE_THRESHOLD=0.75 python app.py
```

> **macOS:** AirPlay Receiver also uses port 5000. If you see "address already in use",
> turn it off in System Settings → General → AirDrop & Handoff.

### 2. Test the API with curl

```bash
curl http://localhost:5000/health
# {"classes":["daisy","dandelion","roses","sunflowers","tulips"],"status":"ok"}

curl -X POST -F "image=@path/to/flower.jpg" http://localhost:5000/predict
```

On Windows PowerShell, type `curl.exe` instead of `curl`.

### 3. Frontend (Next.js)

In a second terminal, with the backend still running:

```bash
cd frontend
npm install

# Optional: use a different backend URL (default http://localhost:5000)
cp .env.example .env.local        # Windows: copy .env.example .env.local

npm run dev
```

Open **http://localhost:3000**, drop in a flower photo, and click **Identify flower**.

---

## API reference

### `GET /health`

```json
{ "status": "ok", "classes": ["daisy", "dandelion", "roses", "sunflowers", "tulips"] }
```

### `POST /predict`

`multipart/form-data` with the image in a field named **`image`** (JPG, JPEG, PNG or WebP, ≤ 5 MB).

**200 OK**

```json
{
  "prediction": "sunflowers",
  "confidence": 0.9731,
  "sure": true,
  "probabilities": [
    { "label": "sunflowers", "probability": 0.9731 },
    { "label": "tulips",     "probability": 0.0142 },
    { "label": "daisy",      "probability": 0.0071 },
    { "label": "dandelion",  "probability": 0.0039 },
    { "label": "roses",      "probability": 0.0017 }
  ]
}
```

| Field | Meaning |
|---|---|
| `prediction` | Class with the highest probability |
| `confidence` | That probability, rounded to 4 decimals |
| `sure` | `confidence >= CONFIDENCE_THRESHOLD` |
| `probabilities` | All 5 classes, sorted from most to least likely |

**Errors** are returned as `{"error": "..."}`:

| Status | When |
|---|---|
| `400` | No file, unsupported type, or a file that can't be decoded as an image |
| `413` | File larger than 5 MB |

CORS allows only `http://localhost:3000`. To serve the frontend from another origin,
update the `CORS(...)` line in `backend/app.py`.
