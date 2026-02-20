from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import create_model

import time
import logging
from prometheus_fastapi_instrumentator import Instrumentator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Cats vs Dogs Classification API")

# Instrumentator
Instrumentator().instrument(app).expose(app)

# Metrics storage
metrics = {
    "request_count": 0,
    "total_latency": 0
}

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Update metrics
    metrics["request_count"] += 1
    metrics["total_latency"] += process_time
    
    # Log request
    logger.info(f"Path: {request.url.path} | Method: {request.method} | Status: {response.status_code} | Latency: {process_time:.4f}s")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.get("/", include_in_schema=False)
async def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/metrics")
def get_metrics():
    avg_latency = metrics["total_latency"] / metrics["request_count"] if metrics["request_count"] > 0 else 0
    return {
        "request_count": metrics["request_count"],
        "average_latency": avg_latency
    }


MODEL_PATH = "model.h5"
model = None

@app.on_event("startup")
def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    else:
        print("Model file not found. Starting with initialized model (untrained).")
        model = create_model()

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        processed_image = preprocess_image(contents)
        
        if model is None:
             raise HTTPException(status_code=500, detail="Model not loaded.")

        prediction = model.predict(processed_image)[0][0]
        label = "Dog" if prediction > 0.5 else "Cat"
        confidence = float(prediction) if label == "Dog" else 1.0 - float(prediction)
        
        return {
            "filename": file.filename,
            "label": label,
            "probability": float(prediction),
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
