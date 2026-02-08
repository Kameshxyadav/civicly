import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms, datasets
from transformers import ResNetForImageClassification, AutoImageProcessor
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import json
import matplotlib.pyplot as plt
from collections import Counter

# =================== CONFIG ===================
DATASET_PATH = "/home/m0ta_b1lla/hackathon-civicly/waste-classifier/database"       # <-- your dataset path
BATCH_SIZE = 32                  # 32 is safe for 8GB VRAM
EPOCHS = 25
LR = 3e-5
NUM_WORKERS = 4
MODEL_NAME = "microsoft/resnet-50"
SAVE_DIR = Path(__file__).resolve().parent / "waste_classifier_model"
# ===============================================

SAVE_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---------- Image Processor & Transforms ----------
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std),
])

# ---------- Load Dataset ----------
full_dataset = datasets.ImageFolder(DATASET_PATH)

def is_valid_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False

valid_samples = [(path, label) for path, label in full_dataset.samples if is_valid_image(path)]
bad_count = len(full_dataset.samples) - len(valid_samples)
if bad_count:
    print(f"\nSkipping {bad_count} unreadable/corrupted image file(s).")
    full_dataset.samples = valid_samples
    full_dataset.imgs = valid_samples
    full_dataset.targets = [label for _, label in valid_samples]

class_names = full_dataset.classes
num_classes = len(class_names)

print(f"\nClasses ({num_classes}): {class_names}")
print(f"Total images: {len(full_dataset)}")

# Print per-class count
label_counts = Counter([label for _, label in full_dataset.samples])
for idx, name in enumerate(class_names):
    print(f"  {name}: {label_counts[idx]} images")

# ---------- Split Dataset ----------
total = len(full_dataset)
train_size = int(0.8 * total)
val_size = int(0.1 * total)
test_size = total - train_size - val_size

train_set, val_set, test_set = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)
print(f"\nSplit: {train_size} train / {val_size} val / {test_size} test")

# Class balancing for training (helps weak classes like plastic/medical/paper).
train_labels = [full_dataset.targets[i] for i in train_set.indices]
train_class_counts = Counter(train_labels)
sample_weights = torch.DoubleTensor([1.0 / train_class_counts[label] for label in train_labels])
train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

class_weight_values = [len(train_labels) / (num_classes * train_class_counts[i]) for i in range(num_classes)]
class_weights = torch.tensor(class_weight_values, dtype=torch.float32, device=device)

# Wrapper to apply transforms
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if img.mode != "RGB":
            # Normalize uncommon palette/alpha modes before augmentation.
            if img.mode == "P" and "transparency" in img.info:
                img = img.convert("RGBA")
            img = img.convert("RGB")
        return self.transform(img), label
    def __len__(self):
        return len(self.subset)

train_loader = DataLoader(TransformSubset(train_set, train_transform),
                          batch_size=BATCH_SIZE, sampler=train_sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(TransformSubset(val_set, val_transform),
                        batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
test_loader = DataLoader(TransformSubset(test_set, val_transform),
                         batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=True)

# ---------- Load Pretrained ResNet-50 ----------
print(f"\nLoading {MODEL_NAME}...")
model = ResNetForImageClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_classes,
    ignore_mismatched_sizes=True,
)
model = model.to(device)

# Freeze early layers and fine-tune the two deepest encoder stages + classifier.
trainable_scopes = ("classifier", "resnet.encoder.stages.2", "resnet.encoder.stages.3")
for name, param in model.named_parameters():
    param.requires_grad = any(scope in name for scope in trainable_scopes)

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"Trainable: {trainable:,} / {total_params:,} parameters")

# ---------- Training Setup ----------
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
amp_enabled = device.type == "cuda"
scaler = torch.amp.GradScaler(enabled=amp_enabled)  # Mixed precision for faster training

best_val_acc = 0.0
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

# ---------- Training Loop ----------
print("\n" + "="*60)
print("TRAINING STARTED")
print("="*60)

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    train_loss, train_correct, train_total = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images).logits
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += images.size(0)

    scheduler.step()

    # --- Validate ---
    model.eval()
    val_loss, val_correct, val_total = 0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(images).logits
                loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += images.size(0)

    t_loss = train_loss / train_total
    v_loss = val_loss / val_total
    t_acc = train_correct / train_total
    v_acc = val_correct / val_total

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["train_acc"].append(t_acc)
    history["val_acc"].append(v_acc)

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | "
          f"Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), SAVE_DIR / "best_model.pth")
        print(f"  ✓ Best model saved (val_acc={v_acc:.4f})")

# ---------- Test ----------
print("\n" + "="*60)
print("TESTING")
print("="*60)

model.load_state_dict(torch.load(SAVE_DIR / "best_model.pth"))
model.eval()

test_correct, test_total = 0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).logits
        preds = outputs.argmax(1)
        test_correct += (preds == labels).sum().item()
        test_total += images.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

print(f"✅ Test Accuracy: {test_correct/test_total:.4f}")

# Per-class accuracy
print("\nPer-class results:")
for i, name in enumerate(class_names):
    cls_correct = sum(1 for p, l in zip(all_preds, all_labels) if l == i and p == i)
    cls_total = sum(1 for l in all_labels if l == i)
    if cls_total > 0:
        print(f"  {name}: {cls_correct}/{cls_total} ({cls_correct/cls_total*100:.1f}%)")

# ---------- Save Config & Plot ----------
config = {"class_names": class_names, "model_name": MODEL_NAME, "num_classes": num_classes}
with open(SAVE_DIR / "config.json", "w") as f:
    json.dump(config, f, indent=2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"], label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"], label="Val")
ax2.set_title("Accuracy"); ax2.legend()
plt.savefig(SAVE_DIR / "training_plot.png", dpi=150)
plt.close(fig)

print(f"\n✅ Everything saved to {SAVE_DIR}/")
