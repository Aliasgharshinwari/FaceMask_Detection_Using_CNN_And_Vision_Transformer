#!/usr/bin/env python3
"""
webcam_mask_quant_demo.py
-----------------------------------
Runs real-time webcam inference using the *quantized* Vision Transformer.
CPU-only, optimized for dynamic quantized models.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
from PIL import Image
import numpy as np
import time

# =====================================================
# CONFIGURATION
# =====================================================
MODEL_PATH = "vit_model_quantized.pth"
IMG_SIZE = 224
LABELS = ["With Mask", "Without Mask", "Mask Incorrect"]
FONT = cv2.FONT_HERSHEY_SIMPLEX

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")
if device.type == "cuda":
    print("🚀 GPU:", torch.cuda.get_device_name(0))


# =====================================================
# LOAD QUANTIZED MODEL
# =====================================================
print("⚙️ Loading quantized Vision Transformer (CPU only)...")

vit = models.vit_b_16(weights=None)
vit.heads = nn.Linear(vit.heads.head.in_features, len(LABELS))
vit.eval()

# Apply same dynamic quantization as during saving
quant_model = torch.quantization.quantize_dynamic(
    vit, {nn.Linear}, dtype=torch.qint8
)
quant_model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
quant_model.to("cpu")
quant_model.eval()

print("✅ Quantized ViT model loaded successfully.")

# =====================================================
# TRANSFORM PIPELINE
# =====================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict_face(model, face_pil):
    img = transform(face_pil).unsqueeze(0)
    with torch.no_grad():
        out = model(img)
        # Quantized ViT sometimes returns tuple (tensor, None)
        if isinstance(out, tuple):
            out = out[0]
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()

# =====================================================
# WEBCAM LOOP (fixed bounding box + real-time FPS)
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not accessible.")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

fps = 0.0
frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    frame_start = time.time()

    # Get frame dimensions
    h, w, _ = frame.shape

    # Define square region for analysis (face crop)
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    face = frame[y1:y1 + size, x1:x1 + size]

    # Convert cropped face to PIL
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

    # Run prediction
    pred, conf = predict_face(quant_model, face_pil)
    label = f"{LABELS[pred]} ({conf*100:.1f}%)"

    # Calculate instantaneous FPS
    frame_time = time.time() - frame_start
    fps = (fps * 0.9) + (1.0 / frame_time * 0.1) if frame_time > 0 else fps

    # Choose color based on prediction
    color = (0, 255, 0) if pred == 0 else (0, 0, 255) if pred == 1 else (0, 255, 255)

    # Resize the cropped region back to overlay dimensions
    face_resized = cv2.resize(face, (size, size))
    frame[y1:y1 + size, x1:x1 + size] = face_resized

    # Draw bounding box properly aligned to the cropped region
    cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), color, 2)

    # Draw labels and FPS
    cv2.putText(frame, label, (x1 + 10, y1 - 10), FONT, 0.8, color, 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (30, 40), FONT, 0.7, (255, 255, 255), 2)

    # Display
    cv2.imshow("Quantized ViT Mask Detector (CPU)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

total_time = time.time() - start_time
print(f"\n📊 Average FPS: {frame_count / total_time:.2f}")
print("👋 Exiting...")
