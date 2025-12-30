import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from torchvision import transforms
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(title="Custom ResNet50 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to specific domains later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load labels
with open("class_names.json") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# Load ONNX model once at startup
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

# Image preprocessing (MUST match training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )
])

@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    tensor = preprocess(image).unsqueeze(0).numpy()

    outputs = session.run(None, {"input": tensor})[0][0]
    probs = np.exp(outputs) / np.sum(np.exp(outputs))

    best_idx = int(np.argmax(probs))
    best_confidence = float(probs[best_idx])

    return {
        "label": class_names[str(best_idx)],
        "confidence": round(best_confidence, 4)
    }
