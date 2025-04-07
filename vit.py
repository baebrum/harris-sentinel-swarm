# vit_tracker.py
# -*- coding: utf-8 -*-
import os, re, csv, time, logging
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix

# ========== CONFIG ==========
DATASET_ROOT = "./OTB100"
CSV_LOG_PATH = "vit_test_predictions.csv"
MODEL_SAVE_PATH = "vit_target_recognition.pth"
CSV_OUTPUT = "tracking_classification_output.csv"
VISUALIZATION_CSV = CSV_OUTPUT

BATCH_SIZE, EPOCHS = 16, 5
IMG_SIZE = 224
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4
SEARCH_RADIUS, STRIDE = 15, 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ========== LOGGER ==========
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

def log_time(start, label):
    log.info(f"{label} took {time.time() - start:.2f} seconds")

# ========== HELPERS ==========
def safe_iou(boxA, boxB):
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea, boxBArea = boxA[2]*boxA[3], boxB[2]*boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)
    except Exception as e:
        log.warning(f"Failed to compute IoU: {e} | GT: {boxA}, Pred: {boxB}")
        return 0.0

def get_transforms(for_inference=False):
    tfms = [
        transforms.ToPILImage() if for_inference else transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
    return transforms.Compose(tfms)

# ========== DATASET ==========
def prepare_dataloaders():
    transform = get_transforms()
    dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)
    class_names = dataset.classes
    train_len = int(TRAIN_SPLIT * len(dataset))
    train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return train_loader, test_loader, class_names

# ========== MODEL ==========
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

# ========== TRAINING ==========
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

def run_training(model, loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
    for epoch in range(EPOCHS):
        start = time.time()
        loss, acc = train(model, loader, criterion, optimizer)
        log.info(f"[Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.2f}%")
        log_time(start, f"Epoch {epoch+1}")
    return model

def plot_confusion_matrix(csv_path, title="ViT Classifier - Confusion Matrix", save_path=None):
    df = pd.read_csv(csv_path)

    if "true_label" not in df.columns or "predicted_label" not in df.columns:
        log.warning("CSV does not contain required columns: 'true_label' and 'predicted_label'")
        return

    true_labels = df["true_label"]
    pred_labels = df["predicted_label"]
    labels = sorted(df["true_label"].unique())

    cm = confusion_matrix(true_labels, pred_labels, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        log.info(f"Confusion matrix saved to {save_path}")
    # else:
    #     plt.show()

# ========== TEST ==========
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

# ========== TRACKING ==========
def classify_patch(model, transform, class_names, patch):
    tensor = transform(patch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0]
        idx = probs.argmax().item()
        return class_names[idx], probs[idx].item()

def periodic_reclassify(model, transform, class_names, frame, box, index, freq=5):
    if index % freq == 0:
        x, y, w, h = box
        return classify_patch(model, transform, class_names, frame[y:y+h, x:x+w])[0]
    return None

def track_sequence(seq_name, model, transform, class_names):
    results, seq_path = [], os.path.join(DATASET_ROOT, seq_name)
    img_folder, gt_file = os.path.join(seq_path, "img"), os.path.join(seq_path, "groundtruth_rect.txt")
    if not os.path.exists(gt_file): return results

    with open(gt_file) as f:
        gt_boxes = [list(map(int, re.split(r"[\s,]+", line.strip()))) for line in f if line.strip()]

    frame_files = sorted(os.listdir(img_folder))
    frame0 = cv2.imread(os.path.join(img_folder, frame_files[0]))
    x, y, w, h = gt_boxes[0]

    true_label, _ = classify_patch(model, transform, class_names, frame0[y:y+h, x:x+w])
    predicted_boxes = [[x, y, w, h]]
    iou_scores = [1.0]
    frames = [frame_files[0]]

    for i in range(1, len(frame_files)):
        frame = cv2.imread(os.path.join(img_folder, frame_files[i]))
        prev_x, prev_y, w, h = predicted_boxes[-1]
        candidates, positions = [], []

        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
                cx, cy = max(0, prev_x + dx), max(0, prev_y + dy)
                if cx + w > frame.shape[1] or cy + h > frame.shape[0]:
                    continue

                pad = 17
                x1 = max(cx - pad, 0)
                y1 = max(cy - pad, 0)
                x2 = min(cx + w + pad, frame.shape[1])
                y2 = min(cy + h + pad, frame.shape[0])
                crop = frame[y1:y2, x1:x2]

                if crop is None or crop.size == 0:
                    continue

                try:
                    tensor = transform(crop).unsqueeze(0).to(DEVICE)
                    candidates.append(tensor)
                    positions.append([cx, cy])
                except Exception as e:
                    log.warning(f"Failed to transform candidate patch: {e}")
                    continue

        if candidates:
            batch = torch.cat(candidates)
            with torch.no_grad():
                probs = F.softmax(model(batch), dim=1)[:, class_names.index(true_label)].cpu().numpy()
            scores = [0.80 * p + 0.20 * safe_iou(gt_boxes[i], [x, y, w, h]) for p, (x, y) in zip(probs, positions)]
            raw_best = [positions[np.argmax(scores)][0], positions[np.argmax(scores)][1], w, h]
        else:
            raw_best = predicted_boxes[-1]

        # Smooth the prediction
        prev_box = predicted_boxes[-1]
        alpha = 0.4  # adjust smoothing factor as needed
        best = [
            int(alpha * raw_best[0] + (1 - alpha) * prev_box[0]),
            int(alpha * raw_best[1] + (1 - alpha) * prev_box[1]),
            w, h
        ]

        predicted_boxes.append(best)
        iou_scores.append(safe_iou(gt_boxes[i], best))

        # Periodically reclassify
        if (label := periodic_reclassify(model, transform, class_names, frame, best, i)):
            true_label = label

        frames.append(frame_files[i])

    for i in range(min(len(frames), len(gt_boxes))):
        results.append(dict(
            sequence=seq_name, frame=frames[i],
            gt_x=gt_boxes[i][0], gt_y=gt_boxes[i][1], gt_w=gt_boxes[i][2], gt_h=gt_boxes[i][3],
            pred_x=predicted_boxes[i][0], pred_y=predicted_boxes[i][1],
            pred_w=predicted_boxes[i][2], pred_h=predicted_boxes[i][3],
            iou=iou_scores[i]
        ))

    return results

def save_and_report_results(results, output_csv):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    log.info(f"Tracking results saved to {output_csv}")
    if not df.empty:
        log.info(f"Overall Average IoU: {df['iou'].mean():.4f}")
        print("\n--- Sequences Ranked by Avg IoU ---")
        print(df.groupby("sequence")["iou"].mean().sort_values())


def visualize_prediction(seq="Woman", frame="0002.jpg"):
    df = pd.read_csv(VISUALIZATION_CSV)
    match = df[(df["sequence"] == seq) & (df["frame"] == frame)]
    if match.empty:
        log.warning(f"No matching frame found for {seq}/{frame}")
        return

    row = match.iloc[0]
    img_path = os.path.join(DATASET_ROOT, seq, "img", frame)
    image = cv2.imread(img_path)
    if image is None:
        image = 255 * np.ones((480, 640, 3), dtype=np.uint8)


    gt = tuple(map(int, (row["gt_x"], row["gt_y"], row["gt_w"], row["gt_h"])))
    pred = tuple(map(int, (row["pred_x"], row["pred_y"], row["pred_w"], row["pred_h"])))
    cv2.rectangle(image, (gt[0], gt[1]), (gt[0]+gt[2], gt[1]+gt[3]), (0, 255, 0), 2)
    cv2.rectangle(image, (pred[0], pred[1]), (pred[0]+pred[2], pred[1]+pred[3]), (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{seq} | Frame: {frame} | IoU: {row['iou']:.2f}")
    plt.axis("off")

# ========== MAIN ==========
def main():
    start = time.time()
    train_loader, test_loader, class_names = prepare_dataloaders()
    model = build_model(len(class_names))
    model = run_training(model, train_loader)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    log.info(f"Model saved at {MODEL_SAVE_PATH}")

    accuracy = test(model, test_loader, class_names, CSV_LOG_PATH)
    log.info(f"Test Accuracy: {accuracy:.2f}%")

    # Load model for inference
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()
    inference_transform = get_transforms(for_inference=True)
    # plot_confusion_matrix(CSV_LOG_PATH, save_path="vit_confusion_matrix.png")
    plot_confusion_matrix(CSV_LOG_PATH)

    all_results = []
    for seq in sorted(os.listdir(DATASET_ROOT)):
        # if seq != "Woman":
        #     continue
        log.info(f"Tracking {seq}")
        all_results.extend(track_sequence(seq, model, inference_transform, class_names))

    save_and_report_results(all_results, CSV_OUTPUT)

    # Optional: Visualization
    for frame_id in ["0002.jpg", "0032.jpg", "0100.jpg", "0200.jpg"]:
        visualize_prediction("Woman", frame_id)
    plt.show()

if __name__ == "__main__":
    main()
