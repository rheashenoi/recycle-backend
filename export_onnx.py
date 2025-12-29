import torch
from torchvision import models

NUM_CLASSES = 5 # change
MODEL_PATH = "model.pth"
ONNX_PATH = "model.onnx"

model = models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.eval()

x = torch.randn(1, 3, 224, 224)

torch.onnx.export(
model,
x,
ONNX_PATH,
input_names=["input"],
output_names=["output"],
opset_version=18
)

print("Exported model.onnx")

