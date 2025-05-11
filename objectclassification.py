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

# CONFIG
DATASET_ROOT = "./UTKFace_5000_Split"
BASE_MODEL_PATH = "vit_target_recognition"
BASE_CSV_PATH = "vit_test_predictions"
BASE_OUTPUT_DIR = "./outputs"
FILTER_DATASET = True
FILTER_PATTERN = r"(man|woman)"
BATCH_SIZE, EPOCHS = 128, 8
IMG_SIZE = 224
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

# ViT PRESETS
# Each preset defines: dim, depth, heads, mlp_dim
PRESETS = {
    "standard":    {"dim": 512, "depth": 6, "heads": 8, "mlp_dim": 1024},
    "small":       {"dim": 128, "depth": 4, "heads": 4, "mlp_dim": 256},
    "tiny":        {"dim": 64,  "depth": 2, "heads": 2, "mlp_dim": 128},
    "super_tiny":  {"dim": 32,  "depth": 1, "heads": 2, "mlp_dim": 64},
}

# HELPERS
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
        selected_idxs = {cls: idx for cls, idx in dataset.class_to_idx.items() if cls in selected_classes}
        filtered = [(path, lbl) for path, lbl in dataset.samples if lbl in selected_idxs.values()]

        dataset.samples = [(p, list(selected_idxs.values()).index(lbl)) for p, lbl in filtered]
        dataset.targets = [lbl for _, lbl in dataset.samples]
        new_classes = list(selected_classes)
        dataset.classes = new_classes
        dataset.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}

        log.info(f"Filtered classes: {dataset.classes}")
    else:
        log.info(f"Using full dataset: {dataset.classes}")

    train_len = int(TRAIN_SPLIT * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=32, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_loader, test_loader, dataset.classes

def build_model(num_classes, cfg):
    model = ViT(
        image_size=IMG_SIZE,
        patch_size=16,
        num_classes=num_classes,
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["heads"],
        mlp_dim=cfg["mlp_dim"],
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
    pd.DataFrame(rows, columns=["image_index","true_label","predicted_label","confidence"]) \
      .to_csv(csv_path, index=False)
    return 100 * correct / total

def plot_confusion_matrix(csv_path, output_dir):
    df = pd.read_csv(csv_path)
    if "true_label" not in df or "predicted_label" not in df:
        log.warning("CSV missing required columns.")
        return
    labels = sorted(df["true_label"].unique())
    cm = confusion_matrix(df["true_label"], df["predicted_label"], labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("ViT Confusion Matrix")
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "vit_confusion_matrix.png")
    plt.savefig(save_path)
    log.info(f"Confusion matrix saved to {save_path}")

# MAIN
def main():
    for preset_name, cfg in PRESETS.items():
        log.info(f"=== Training ViT [{preset_name}] ===")
        # prepare paths
        model_path   = f"{BASE_MODEL_PATH}_{preset_name}.pth"
        csv_path     = f"{BASE_CSV_PATH}_{preset_name}.csv"
        output_dir   = os.path.join(BASE_OUTPUT_DIR, preset_name)
        os.makedirs(output_dir, exist_ok=True)

        # data
        train_loader, test_loader, class_names = prepare_dataloaders()

        # model, loss, optimizer
        model     = build_model(len(class_names), cfg)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # training loop
        for epoch in range(EPOCHS):
            start = time.time()
            loss, acc = train(model, train_loader, criterion, optimizer)
            log.info(f"[{preset_name}][Epoch {epoch+1}/{EPOCHS}] Loss: {loss:.4f}, Train Acc: {acc:.2f}%")
            log.info(f"[{preset_name}] Epoch {epoch+1} took {time.time() - start:.2f}s")

        # save model
        torch.save(model.state_dict(), model_path)
        log.info(f"[{preset_name}] Model saved at {model_path}")

        # evaluate & plot
        test_acc = test(model, test_loader, class_names, csv_path)
        log.info(f"[{preset_name}] Test Accuracy: {test_acc:.2f}%")
        plot_confusion_matrix(csv_path, output_dir)

if __name__ == "__main__":
    main()