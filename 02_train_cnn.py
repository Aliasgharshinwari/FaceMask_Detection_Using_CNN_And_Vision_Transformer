import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time

from dataset_loader import create_dataloaders

# =====================================================
# CONFIGURATION
# =====================================================
DATASET_ROOT = "../dataset"
IMG_SIZE = 224
NUM_CLASSES = 3
EPOCHS = 5
LR = 1e-3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# =====================================================
# LOAD DATASET
# =====================================================
train_loader, test_loader = create_dataloaders(DATASET_ROOT, verbose=True)

# =====================================================
# DEFINE CNN
# =====================================================
class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
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

# =====================================================
# INITIALIZE MODEL
# =====================================================
model = CNNModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler()

# -----------------------------------------------------
# 🧮 Count total trainable parameters
# -----------------------------------------------------
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n🧮 Total Trainable Parameters: {num_params:,}\n")

# =====================================================
# TRAIN LOOP
# =====================================================
start_time = time.time()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

# =====================================================
# EVALUATION
# =====================================================
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = torch.argmax(model(imgs), dim=1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 CNN Test Accuracy: {acc*100:.2f}%")

# =====================================================
# SAVE MODEL
# =====================================================
torch.save(model.state_dict(), "cnn_model.pth")
print("✅ Saved model as cnn_model.pth")

# =====================================================
# BENCHMARK INFERENCE SPEED
# =====================================================
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(100):
        _ = model(dummy)
torch.cuda.synchronize()
fps = 100 / (time.time() - start)
print(f"⚡ CNN Inference Speed: {fps:.2f} FPS")

# =====================================================
# TOTAL TRAINING TIME
# =====================================================
end_time = time.time()
elapsed = end_time - start_time
mins, secs = divmod(elapsed, 60)
print(f"\n⏱️ Total Training Time: {int(mins)} min {int(secs)} sec")
