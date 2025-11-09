import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

from dataset_loader import create_dataloaders

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_ROOT = "../dataset"
NUM_CLASSES = 3
EPOCHS = 25
LR = 2e-5
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# LOAD DATASET
# =====================================================
train_loader, test_loader = create_dataloaders(DATASET_ROOT, verbose=True)

# =====================================================
# LOAD PRETRAINED ViT
# =====================================================
vit = models.vit_b_16(pretrained=True)
vit.heads = nn.Linear(vit.heads.head.in_features, NUM_CLASSES)
vit = vit.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

# -----------------------------------------------------
# 🧮 Count trainable parameters
# -----------------------------------------------------
num_params = sum(p.numel() for p in vit.parameters() if p.requires_grad)
print(f"\n🧮 Total Trainable Parameters: {num_params:,}\n")

# =====================================================
# TRAIN LOOP
# =====================================================
start_time = time.time()

for epoch in range(EPOCHS):
    vit.train()
    total_loss = 0.0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            outputs = vit(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss:.4f}")

# =====================================================
# EVALUATE
# =====================================================
vit.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = torch.argmax(vit(imgs), dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 ViT Test Accuracy: {acc*100:.2f}%")

# =====================================================
# SAVE MODEL
# =====================================================
torch.save(vit.state_dict(), "vit_model.pth")
print("✅ Saved model as vit_model.pth")

# =====================================================
# BENCHMARK INFERENCE SPEED
# =====================================================
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = vit(dummy)
torch.cuda.synchronize()
fps = 100 / (time.time() - start)
print(f"⚡ ViT Inference Speed: {fps:.2f} FPS")

# =====================================================
# TOTAL TRAINING TIME
# =====================================================
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"\n⏱️ Total Training Time: {int(mins)} min {int(secs)} sec\n")
