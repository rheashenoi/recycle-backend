import io
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from torchvision import transforms

# Load labels
with open("class_names.json") as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# Load ONNX model (global = cached between invocations)
session = ort.InferenceSession(
    "model.onnx",
    providers=["CPUExecutionProvider"]
)

# Preprocessing (must match training)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def handler(request):
    if request.method != "POST":
        return {
            "statusCode": 405,
            "body": "Method Not Allowed"
        }

    file = request.files.get("file")
    if not file:
        return {
            "statusCode": 400,
            "body": "Missing image file"
        }

    image = Image.open(io.BytesIO(file)).convert("RGB")
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

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"predictions": results})
    }

