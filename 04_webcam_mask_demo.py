#!/usr/bin/env python3
"""
webcam_mask_demo.py
-------------------------------------
Real-time webcam demo for trained CNN or ViT mask-detection models.
 - Automatically detects faces using OpenCV
 - Classifies each face using your trained model
 - GPU accelerated (CUDA if available)
Press 'q' to exit.
"""

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import argparse

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 224
CLASS_NAMES = ["With Mask", "Without Mask", "Mask Worn Incorrectly"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")
if device.type == "cuda":
    print("🚀 GPU:", torch.cuda.get_device_name(0))


# ============================================================
# MODEL DEFINITIONS (must match training)
# ============================================================
class CNNModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.fc(self.features(x))


# ============================================================
# LOAD MODEL (auto-detect CNN or ViT)
# ============================================================
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model_name = os.path.basename(model_path).lower()
    if "vit" in model_name:
        print("🧠 Detected Vision Transformer model.")
        model = models.vit_b_16(pretrained=False)
        model.heads = nn.Linear(model.heads.head.in_features, 3)
    else:
        print("🧱 Detected CNN model.")
        model = CNNModel(num_classes=3)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"✅ Loaded model: {model_path}")
    return model


# ============================================================
# TRANSFORM (same as training)
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# ============================================================
# PREDICTION HELPER
# ============================================================
def predict_face(model, face_img):
    img_tensor = transform(face_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img_tensor)
        probs = torch.softmax(out, dim=1)
        conf, pred = torch.max(probs, dim=1)
    return pred.item(), conf.item()


# ============================================================
# DRAW LABEL
# ============================================================
def draw_label(frame, x, y, w, h, label, conf):
    color = (0, 255, 0) if label == 0 else ((0, 0, 255) if label == 1 else (0, 255, 255))
    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, f"{CLASS_NAMES[label]} ({conf*100:.1f}%)",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# ============================================================
# MAIN FUNCTION
# ============================================================
def run_webcam(model_path):
    model = load_model(model_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                         "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not access webcam.")
        return

    print("🎥 Webcam started. Press 'q' to quit.")

    fps = 0.0
    frame_count = 0
    start_time = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        t0 = cv2.getTickCount()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            face_pil = Image.fromarray(cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB))
            pred, conf = predict_face(model, face_pil)
            draw_label(frame, x, y, w, h, pred, conf)

        # --- FPS calculation ---
        t1 = cv2.getTickCount()
        frame_time = (t1 - t0) / cv2.getTickFrequency()
        fps = (fps * 0.9) + (1.0 / frame_time * 0.1) if frame_time > 0 else fps

        # Display FPS in corner
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Real-Time Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    total_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    avg_fps = frame_count / total_time
    print(f"\n📊 Average FPS: {avg_fps:.2f}")

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed.")


    cap.release()
    cv2.destroyAllWindows()
    print("✅ Webcam closed.")


run_webcam("vit_model.pth")

#python webcam_mask_demo.py --model vit_model.pth
#python webcam_mask_demo.py --model cnn_model.pth
