import onnxruntime as ort
import torch
import time
import numpy as np

from dataset import get_datasets
from utils import get_model

onnx_model_path = "best_unetpp_128.onnx"
pth_model_path = "best_unetpp_128.pth"

# ONNX session
session = ort.InferenceSession(onnx_model_path)

# PyTorch model
model = get_model("unetpp")
model.load_state_dict(torch.load(pth_model_path, map_location="cpu"))
model.eval()
model.to("cpu")

dataset = get_datasets(img_size=128)

onnx_latencies = []
pth_latencies = []

# ---- warmup ----
for i in range(5):
    img, _ = dataset[i]
    img_np = img.unsqueeze(0).numpy()
    img_torch = img.unsqueeze(0)

    session.run(None, {"input": img_np})
    with torch.no_grad():
        model(img_torch)

# ---- benchmark ----
for i in range(50):  # limit for stability
    img, _ = dataset[i]

    img_np = img.unsqueeze(0).numpy()
    img_torch = img.unsqueeze(0)

    # ONNX
    start = time.time()
    onnx_output = session.run(None, {"input": img_np})[0]
    onnx_latencies.append(time.time() - start)

    # PyTorch
    with torch.no_grad():
        start = time.time()
        pth_output = model(img_torch)
        pth_latencies.append(time.time() - start)

# ---- results ----
print(f"Average ONNX latency: {np.mean(onnx_latencies)*1000:.2f} ms")
print(f"Average PyTorch latency: {np.mean(pth_latencies)*1000:.2f} ms")
diff = np.mean(np.abs(
    onnx_output - pth_output.detach().numpy()
))
print("Mean abs diff:", diff)