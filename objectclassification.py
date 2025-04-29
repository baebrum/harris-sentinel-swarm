# vit_training.py (cleaned)

import os, re, time, logging
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

from vit_pytorch import ViT

# ========== CONFIG ==========
DATASET_ROOT = "/Users/jacobanderson/Documents/Spring 2025/CompE696/compe-696/Man-Woman"
MODEL_SAVE_PATH = "vit_target_recognition.pth"
CSV_LOG_PATH = "vit_test_predictions.csv"
SAVE_PLOTS = True
OUTPUT_DIR = "./outputs"
FILTER_DATASET = True
FILTER_PATTERN = r"(men|women)"

BATCH_SIZE, EPOCHS = 16, 3
IMG_SIZE = 224
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

# ========== HELPERS ==========
def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

def prepare_dataloaders():
    transform = get_transforms()
    dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)

    if FILTER_DATASET:
        pattern = re.compile(FILTER_PATTERN)
        selected_classes = sorted([cls for cls in dataset.classes if pattern.search(cls)])
        if not selected_classes:
            raise ValueError("No classes matched the regex pattern.")

        selected_class_idxs = {cls: idx for cls, idx in dataset.class_to_idx.items() if cls in selected_classes}
        filtered_samples = [s for s in dataset.samples if s[1] in selected_class_idxs.values()]

        dataset.samples = filtered_samples
        dataset.targets = [label for _, label in filtered_samples]

        old_to_new_idx = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted(selected_class_idxs.values()))}
        dataset.samples = [(path, old_to_new_idx[label]) for path, label in dataset.samples]
        dataset.targets = [old_to_new_idx[label] for label in dataset.targets]

        idx_to_class = {v: k for k, v in selected_class_idxs.items()}
        new_classes = [idx_to_class[old_idx] for old_idx in sorted(selected_class_idxs.values())]
        dataset.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}
        dataset.classes = new_classes

        log.info(f"Filtered classes: {dataset.classes}")
    else:
        log.info(f"Using full dataset: {dataset.classes}")

    train_len = int(TRAIN_SPLIT * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader, dataset.classes

def build_model(num_classes):
    model = ViT(
        image_size=IMG_SIZE,
        patch_size=16,
        num_classes=num_classes,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ).to(DEVICE)
    return model

def train(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += outputs.argmax(1).eq(labels).sum().item()
        total += labels.size(0)
    return total_loss, 100 * correct / total

def test(model, loader, class_names, csv_path):
    model.eval()
    correct, total = 0, 0
    rows = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            probs = F.softmax(model(images), dim=1)
            top_probs, top_idxs = probs.max(1)
            pred, conf = top_idxs.item(), top_probs.item()
            correct += (pred == labels.item())
            rows.append([idx, class_names[labels.item()], class_names[pred], f"{conf:.4f}"])
            total += 1
    pd.DataFrame(rows, columns=["image_index", "true_label", "predicted_label", "confidence"]).to_csv(csv_path, index=False)
    return 100 * correct / total

def plot_confusion_matrix(csv_path):
    df = pd.read_csv(csv_path)
    if "true_label" not in df.columns or "predicted_label" not in df.columns:
        log.warning("CSV missing required columns.")
        return
    true_labels = df["true_label"]
    pred_labels = df["predicted_label"]
    labels = sorted(true_labels.unique())
    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ViT Confusion Matrix")
    plt.tight_layout()

    if SAVE_PLOTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(OUTPUT_DIR, "vit_confusion_matrix.png")
        plt.savefig(save_path)
        log.info(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

# ========== MAIN ==========
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_loader, test_loader, class_names = prepare_dataloaders()

    model = build_model(len(class_names))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    for epoch in range(EPOCHS):
        start = time.time()
        loss, acc = train(model, train_loader, criterion, optimizer)
        log.info(f"[Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.2f}%")
        log.info(f"Epoch {epoch+1} took {time.time() - start:.2f} seconds")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log.info(f"Model saved at {MODEL_SAVE_PATH}")

    accuracy = test(model, test_loader, class_names, CSV_LOG_PATH)
    log.info(f"Test Accuracy: {accuracy:.2f}%")

    plot_confusion_matrix(CSV_LOG_PATH)

if __name__ == "__main__":
    main()