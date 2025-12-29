import io
import json
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import onnxruntime as ort
from torchvision import transforms

app = FastAPI(title="Custom ResNet50 API")

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

topk = min(5, NUM_CLASSES)
indices = probs.argsort()[-topk:][::-1]

results = [
{
"label": class_names[str(i)],
"confidence": round(float(probs[i]), 4)
}
for i in indices
]

return {"predictions": results}
