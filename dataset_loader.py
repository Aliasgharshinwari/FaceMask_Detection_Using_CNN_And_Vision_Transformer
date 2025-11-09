# dataset_loader.py

import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
PIN_MEMORY = True

CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}
PRIORITY = ["without_mask", "mask_weared_incorrect", "with_mask"]

class FaceMaskDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None, verbose=False):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.samples = []
        self.verbose = verbose

        img_labels = defaultdict(set)
        ann_files = [f for f in os.listdir(self.ann_dir) if f.endswith(".xml")]

        for ann in ann_files:
            ann_path = os.path.join(self.ann_dir, ann)
            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
                filename = root.find("filename").text
                for obj in root.findall("object"):
                    label = obj.find("name").text
                    if label in CLASS_MAP:
                        img_labels[filename].add(label)
            except Exception:
                continue

        for img_name, labels in img_labels.items():
            for p in PRIORITY:
                if p in labels:
                    label = CLASS_MAP[p]
                    break
            else:
                continue

            img_path = os.path.join(self.img_dir, img_name)
            if os.path.exists(img_path):
                self.samples.append((img_path, label))

        if verbose:
            print(f"Loaded {len(self.samples)} samples.")
            c = Counter([l for _, l in self.samples])
            print("Class distribution:", dict(c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


def create_dataloaders(dataset_root, batch_size=BATCH_SIZE, verbose=False):
    img_dir = os.path.join(dataset_root, "images")
    ann_dir = os.path.join(dataset_root, "annotations")

    # 🔹 Training transforms (data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # 🔹 Test transforms (no augmentation)
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Build full dataset once
    full_dataset = FaceMaskDataset(img_dir, ann_dir, transform=None, verbose=verbose)

    # Split into train/test
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_indices, test_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, test_size])

    # Apply separate transforms to subsets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    # Wrap transforms
    train_dataset.dataset.transform = train_transforms
    test_dataset.dataset.transform = test_transforms

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    return train_loader, test_loader
