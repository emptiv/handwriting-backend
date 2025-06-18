from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import io
import base64
import keras
from fastapi import FastAPI, Request

app = FastAPI()
try:
    model = keras.models.load_model("model_cc3_2.keras")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


decoder = {0: 'sa', 1: 'da/ra', 2: 'ta'}

@app.post("/predict")
async def predict(req: Request):
    if model is None:
        return {"error": "Model not loaded."}
    
    data = await req.json()
    image_data = data.get("image")

    try:
        # Decode base64 PNG
        img_bytes = base64.b64decode(image_data)
        with open("received.png", "wb") as f:
            f.write(img_bytes)

        img = Image.open(io.BytesIO(img_bytes)).convert("L")

        # Invert like tkinter (white BG → black BG)
        img = ImageOps.invert(img)

        min_dim = min(img.size)
        img = ImageOps.fit(img, (min_dim, min_dim), method=Image.Resampling.BILINEAR, centering=(0.5, 0.5))
        img = img.resize((50, 50), resample=Image.Resampling.BILINEAR)

        # Normalize and reshape
        arr = np.array(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=(0, -1))  # shape: (1, 50, 50, 1)

        pred = model.predict(arr)
        print("Prediction raw:", pred.tolist())
        label = int(np.argmax(pred))
        print("Predicted label index:", label)
        print("Prediction:", decoder[label])

        # ✅ THIS IS THE MISSING LINE
        return {"prediction": decoder[label]}
        
    except Exception as e:
        print("Error during prediction:", e)
        return {"error": str(e)}
        
