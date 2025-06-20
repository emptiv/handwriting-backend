from PIL import Image, ImageOps
import numpy as np
import io
import base64
import keras
from fastapi import FastAPI, Request
import json
import os

app = FastAPI()

# Manually load all models and decoders
MODELS_DIR = "models"
DECODERS_DIR = "decoders"

models = {}
decoders = {}

# Preload available models and decoders
for filename in os.listdir(MODELS_DIR):
    if filename.endswith(".keras"):
        lesson_id = filename.replace(".keras", "")
        try:
            models[lesson_id] = keras.models.load_model(os.path.join(MODELS_DIR, filename))
            decoder_path = os.path.join(DECODERS_DIR, f"{lesson_id}.json")
            if os.path.exists(decoder_path):
                with open(decoder_path, "r") as f:
                    decoders[lesson_id] = json.load(f)
            else:
                print(f"⚠️ No decoder found for {lesson_id}")
        except Exception as e:
            print(f"❌ Failed to load model for {lesson_id}: {e}")

@app.post("/predict")
async def predict(req: Request):
    data = await req.json()
    lesson = data.get("lesson")
    image_data = data.get("image")

    if not lesson or lesson not in models:
        return {"error": f"Lesson '{lesson}' not found or model not loaded."}
    if not image_data:
        return {"error": "No image data provided."}

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")
        img = ImageOps.invert(img)

        min_dim = min(img.size)
        img = ImageOps.fit(img, (min_dim, min_dim), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        img = img.resize((50, 50), resample=Image.Resampling.BILINEAR)

        # Preprocess for model
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # shape: (1, 50, 50, 1)

        # Prediction
        model = models[lesson]
        decoder = decoders.get(lesson, {})
        pred = model.predict(arr)
        label = int(np.argmax(pred))
        decoded_label = decoder.get(str(label), f"Label {label}")

        print(f"Lesson: {lesson} | Prediction: {decoded_label}")
        return {"prediction": decoded_label}

    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}
