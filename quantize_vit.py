#!/usr/bin/env python3
"""
quantize_vit.py
----------------------------------------
Quantizes a pretrained Vision Transformer (ViT) model using dynamic quantization.
This reduces model size and speeds up inference on CPU (and even GPU in some cases).
"""

import torch
import torch.nn as nn
from torchvision import models
from dataset_loader import create_dataloaders
import time
from sklearn.metrics import accuracy_score

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "vit_model.pth"
DATASET_ROOT = "/home/ali/.cache/kagglehub/datasets/andrewmvd/face-mask-detection/versions/1"
NUM_CLASSES = 3
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU for quantization.")

# =====================================================
# LOAD DATASET (for accuracy check)
# =====================================================
_, test_loader = create_dataloaders(DATASET_ROOT, verbose=False)

# =====================================================
# LOAD TRAINED ViT MODEL
# =====================================================
vit = models.vit_b_16(pretrained=False)
vit.heads = nn.Linear(vit.heads.head.in_features, NUM_CLASSES)
vit.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
vit.eval()

print(f"\n✅ Loaded pretrained ViT weights from: {MODEL_PATH}")

# =====================================================
# DYNAMIC QUANTIZATION
# =====================================================
print("\n⚙️ Applying Dynamic Quantization ...")

quantized_model = torch.quantization.quantize_dynamic(
    vit,
    {nn.Linear},          # quantize Linear layers
    dtype=torch.qint8     # use int8 weights
)

# =====================================================
# SAVE QUANTIZED MODEL
# =====================================================
quant_path = "vit_model_quantized.pth"
torch.save(quantized_model.state_dict(), quant_path)
print(f"💾 Quantized model saved to: {quant_path}")
# =====================================================
# EVALUATE ACCURACY (Post-Quantization on CPU)
# =====================================================
quantized_model.eval()
quantized_model.to("cpu")

y_true, y_pred = [], []
start_eval = time.time()

with torch.no_grad():
    for imgs, labels in test_loader:
        outputs = quantized_model(imgs)  # all on CPU
        preds = torch.argmax(outputs, dim=1)
        y_true.extend(labels.numpy())
        y_pred.extend(preds.numpy())

acc = accuracy_score(y_true, y_pred)
end_eval = time.time()
print(f"\n🎯 Quantized ViT Test Accuracy: {acc*100:.2f}%")
print(f"⏱️ Evaluation Time: {(end_eval - start_eval):.2f} sec")

# =====================================================
# BENCHMARK INFERENCE SPEED (CPU)
# =====================================================
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = quantized_model(dummy)
fps = 100 / (time.time() - start)
print(f"⚡ Quantized ViT Inference Speed (CPU): {fps:.2f} FPS")

# =====================================================
# COMPARE MODEL SIZES
# =====================================================
import os

original_size = os.path.getsize(MODEL_PATH) / (1024 ** 2)
quantized_size = os.path.getsize(quant_path) / (1024 ** 2)
reduction = 100 * (1 - quantized_size / original_size)

print(f"\n💡 Model Size Reduction: {original_size:.2f} → {quantized_size:.2f} MB ({reduction:.1f}% smaller)")
print("✅ Quantization complete!\n")
