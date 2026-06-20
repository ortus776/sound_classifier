import torch
from model_def import MiniVGGClassifier
model = MiniVGGClassifier()
model.eval()
dummy_input = torch.randn(1, 2, 64, 345)
torch.onnx.export(model, dummy_input, "model.onnx")