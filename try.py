from PIL import Image, ImageOps
import numpy as np
import io
import base64
import keras
from fastapi import FastAPI, Request

app = FastAPI()
try:
    model = keras.models.load_model("baybayin_model_v2.keras")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


decoder = {0: 'e_i', 1: 'o_u', 2: 'a'}

@app.post("/predict")
async def predict(req: Request):
    if model is None:
        return {"error": "Model not loaded."}
    data = await req.json()
    image_data = data.get("image")

    try:
        # Decode base64 PNG
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert("L")  # grayscale

        # Invert like tkinter (white BG → black BG)
        img = ImageOps.invert(img)

        # Resize to model input size
        img = img.resize((50, 50), resample=Image.Resampling.BILINEAR)

        # Normalize and reshape
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # shape: (1, 50, 50, 1)

        pred = model.predict(arr)
        label = int(np.argmax(pred))
        print("Predicted label index:", label)

        print("Prediction:", decoder[label])
    except Exception as e:
        print("Error during prediction:", e)
        return {"error": str(e)}
