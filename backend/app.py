"""Flask API that serves the flower classifier (flowers.keras)."""

import io
import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError

# Keras 3 picks its backend from this env var; set it before importing keras.
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "flowers.keras"
CLASSES_PATH = BASE_DIR / "classes.json"

IMG_SIZE = (224, 224)  # the input size MobileNetV2 was trained with
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
ALLOWED_MIMETYPES = {"image/jpeg", "image/png", "image/webp"}
# PIL's name for each format we accept (checked after opening, so a renamed .txt fails).
ALLOWED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}

# Below this confidence the API reports "sure": false. Override with e.g.
#   CONFIDENCE_THRESHOLD=0.75 python app.py
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.6"))

# --- Model loading ----------------------------------------------------------
# Loading a model is slow (seconds), so we do it ONCE when the server starts,
# not on every request. load_model rebuilds the full architecture (MobileNetV2
# base + your classification head) and restores the trained weights from the
# .keras file. compile=False skips restoring the optimizer/loss, which are only
# needed for training, not for making predictions.
print(f"Loading model from {MODEL_PATH} ...")
model = keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded.")

# The class order MUST match the order the model was trained with: output[i]
# is the probability of CLASS_NAMES[i].
with open(CLASSES_PATH, encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

app = Flask(__name__)
# Flask rejects bodies larger than this with a 413 before our code runs.
# Allow a little extra for multipart headers; the exact 5 MB check is below.
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 64 * 1024
CORS(app, origins=["http://localhost:3000"])


def error(message, status):
    return jsonify({"error": message}), status


def preprocess(image_bytes):
    """Turn raw uploaded bytes into the tensor the model expects: (1, 224, 224, 3)."""
    img = Image.open(io.BytesIO(image_bytes))
    if img.format not in ALLOWED_PIL_FORMATS:
        raise ValueError("unsupported format")

    # convert("RGB") drops a PNG's alpha (transparency) channel and turns
    # grayscale/palette images into 3-channel color, since the model expects
    # exactly 3 channels (R, G, B).
    img = img.convert("RGB")

    # Resize with BILINEAR to match image_dataset_from_directory, which
    # used bilinear interpolation when it loaded the training images.
    img = img.resize(IMG_SIZE, Image.Resampling.BILINEAR)

    # Pixels stay in the 0-255 range as float32. We do NOT divide by 255 or
    # call preprocess_input: the model has its own Rescaling layer inside it,
    # so scaling here would scale twice and wreck the predictions.
    arr = np.asarray(img, dtype=np.float32)  # shape (224, 224, 3)

    # Models predict on batches, so add a batch dimension: (1, 224, 224, 3).
    return np.expand_dims(arr, axis=0)


@app.get("/health")
def health():
    return jsonify({"status": "ok", "classes": CLASS_NAMES})


@app.post("/predict")
def predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return error("No image uploaded. Send the file in a form field named 'image'.", 400)

    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS and file.mimetype not in ALLOWED_MIMETYPES:
        return error("Unsupported file type. Please upload a JPG, PNG or WebP image.", 400)

    # Read into memory only; uploads are never written to disk.
    image_bytes = file.read()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        return error("Image is too large. The maximum size is 5 MB.", 413)

    try:
        batch = preprocess(image_bytes)
    except (UnidentifiedImageError, ValueError, OSError):
        return error("Could not read the image. Please upload a valid JPG, PNG or WebP file.", 400)

    # Augmentation layers (random flip/rotation, etc.) are automatically
    # inactive here because predict() runs the model in inference mode.
    probs = model.predict(batch, verbose=0)[0]  # shape (5,), sums to ~1

    ranked = sorted(
        ({"label": label, "probability": round(float(p), 4)} for label, p in zip(CLASS_NAMES, probs)),
        key=lambda item: item["probability"],
        reverse=True,
    )
    top = ranked[0]

    return jsonify(
        {
            "prediction": top["label"],
            "confidence": top["probability"],
            "sure": top["probability"] >= CONFIDENCE_THRESHOLD,
            "probabilities": ranked,
        }
    )


@app.errorhandler(413)
def too_large(_e):
    return error("Image is too large. The maximum size is 5 MB.", 413)


if __name__ == "__main__":
    # 127.0.0.1 = only reachable from this computer (safe for local dev).
    app.run(host="127.0.0.1", port=5000)
