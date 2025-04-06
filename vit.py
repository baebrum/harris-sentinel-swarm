# -*- coding: utf-8 -*-
import os
import re
import csv
import cv2
import time
import torch
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torch.nn.functional as F
from vit_pytorch import ViT
from sklearn.metrics import confusion_matrix

# ==== CONFIG ====
DATASET_ROOT = "/mnt/beegfs/dgx/acarranza/docs/harris-sentinel-swarm/OTB100"
CSV_LOG_PATH = "vit_test_predictions.csv"
MODEL_SAVE_PATH = "vit_target_recognition.pth"
CSV_OUTPUT = "tracking_classification_output.csv"
VISUALIZATION_CSV = CSV_OUTPUT

BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4

SEARCH_RADIUS = 40
STRIDE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==== LOGGER ====
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
log = logging.getLogger()

def log_time(start, label):
    log.info(f"{label} took {time.time() - start:.2f} seconds")

def safe_iou(boxA, boxB):
    try:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
        yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea)
    except Exception as e:
        log.warning(f"Failed to compute IoU: {e} | GT: {boxA}, Pred: {boxB}")
        return 0.0

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==== DATASET ====
start_time = time.time()
full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)
class_names = full_dataset.classes
train_len = int(TRAIN_SPLIT * len(full_dataset))
test_len = len(full_dataset) - train_len
train_set, test_set = random_split(full_dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
log_time(start_time, "Data loading")

# ==== MODEL ====
start_time = time.time()
model = ViT(
    image_size=IMG_SIZE,
    patch_size=16,
    num_classes=len(class_names),
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LEARNING_RATE)
log_time(start_time, "Model initialization")

# ==== TRAIN ====
def train(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss, 100 * correct / total

start_time = time.time()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    loss, acc = train(model, train_loader, criterion, optimizer, DEVICE)
    log.info(f"[Epoch {epoch+1}] Loss: {loss:.4f}, Train Acc: {acc:.2f}%")
    log_time(epoch_start, f"Epoch {epoch+1}")
log_time(start_time, "Total training")

# ==== TEST ====
def test(model, loader, class_names, device, csv_path):
    model.eval()
    correct, total = 0, 0
    rows = []
    with torch.no_grad():
        for idx, (images, labels) in enumerate(loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.max(1)
            predicted = top_idxs.item()
            confidence = top_probs.item()
            correct += (predicted == labels.item())
            total += 1
            rows.append([idx, class_names[labels.item()], class_names[predicted], f"{confidence:.4f}"])

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_index", "true_label", "predicted_label", "confidence"])
        writer.writerows(rows)

    return 100 * correct / total

start_time = time.time()
accuracy = test(model, test_loader, class_names, DEVICE, CSV_LOG_PATH)
log.info(f"🎯 Test Accuracy: {accuracy:.2f}%")
log_time(start_time, "Model testing")

# ==== SAVE MODEL ====
start_time = time.time()
torch.save(model.state_dict(), MODEL_SAVE_PATH)
log.info(f"Model saved to {MODEL_SAVE_PATH}")
log_time(start_time, "Model saving")

# ==== TRACKING FUNCTION ====
def classify_patch(model, transform, class_names, img_patch):
    input_tensor = transform(img_patch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return class_names[pred_idx], probs[pred_idx].item()

def track_sequence(seq_name, model, transform, class_names):
    results = []
    seq_path = os.path.join(DATASET_ROOT, seq_name)
    gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
    img_folder = os.path.join(seq_path, "img")
    if not os.path.exists(gt_path) or not os.path.isdir(img_folder):
        return results

    with open(gt_path, "r") as f:
        gt_boxes = [list(map(int, re.split(r"[\s,]+", line.strip()))) for line in f if line.strip()]

    frame_files = sorted(os.listdir(img_folder))
    predicted_boxes, iou_scores, processed_frames = [], [], []

    first_frame = cv2.imread(os.path.join(img_folder, frame_files[0]))
    if first_frame is None:
        log.warning(f"Skipping sequence '{seq_name}' due to missing first frame.")
        return results

    x, y, w, h = gt_boxes[0]
    crop = first_frame[y:y+h, x:x+w]
    true_label, _ = classify_patch(model, transform, class_names, crop)
    predicted_boxes.append([x, y, w, h])
    iou_scores.append(1.0)
    processed_frames.append(frame_files[0])

    for i in range(1, len(frame_files)):
        frame = cv2.imread(os.path.join(img_folder, frame_files[i]))
        if frame is None:
            log.warning(f"Skipping frame {frame_files[i]} (could not load)")
            continue

        prev_x, prev_y, prev_w, prev_h = predicted_boxes[-1]
        candidates, positions = [], []

        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
                cx, cy = max(0, prev_x + dx), max(0, prev_y + dy)
                if cx + prev_w > frame.shape[1] or cy + prev_h > frame.shape[0]:
                    continue
                crop = frame[cy:cy + prev_h, cx:cx + prev_w]
                try:
                    tensor = transform(crop).unsqueeze(0)
                    candidates.append(tensor)
                    positions.append([cx, cy])
                except:
                    continue

        if candidates:
            batch_tensor = torch.cat(candidates).to(DEVICE)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                class_idx = class_names.index(true_label)
                confidences = probs[:, class_idx]
            max_idx = torch.argmax(confidences).item()
            best_box = [positions[max_idx][0], positions[max_idx][1], prev_w, prev_h]
        else:
            best_box = predicted_boxes[-1]

        predicted_boxes.append(best_box)
        iou_scores.append(safe_iou(gt_boxes[i], best_box))
        processed_frames.append(frame_files[i])

    for i in range(min(len(processed_frames), len(gt_boxes))):
        results.append({
            "sequence": seq_name,
            "frame": processed_frames[i],
            "gt_x": gt_boxes[i][0], "gt_y": gt_boxes[i][1], "gt_w": gt_boxes[i][2], "gt_h": gt_boxes[i][3],
            "pred_x": predicted_boxes[i][0], "pred_y": predicted_boxes[i][1], "pred_w": predicted_boxes[i][2], "pred_h": predicted_boxes[i][3],
            "iou": iou_scores[i]
        })
    return results

# ==== RUN TRACKING ====
start_time = time.time()
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
model.eval()

inference_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

all_results = []
for seq in sorted(os.listdir(DATASET_ROOT)):
    if seq != "Woman":
        continue
    seq_time = time.time()
    all_results.extend(track_sequence(seq, model, inference_transform, class_names))
    log_time(seq_time, f"Sequence '{seq}'")

log_time(start_time, "All tracking and classification")

# ==== SAVE & REPORT RESULTS ====
def save_and_report_results(results, output_csv):
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    log.info(f"Tracking results saved to {output_csv}")

    if 'iou' in df.columns and not df['iou'].empty:
        avg_iou = df['iou'].mean()
        log.info(f"Overall Average IoU: {avg_iou:.4f}")

        avg_iou_per_seq = df.groupby("sequence")["iou"].mean().reset_index().sort_values(by="iou", ascending=False)
        avg_iou_per_seq.columns = ["sequence", "average_iou"]
        print("\n=== Average IoU per Sequence ===")
        print(avg_iou_per_seq.to_string(index=False))

        avg_iou_per_seq.to_csv("average_iou_per_sequence.csv", index=False)
    else:
        log.warning("No valid IoU values were recorded.")

save_and_report_results(all_results, CSV_OUTPUT)

# ==== VISUALIZATION FUNCTION ====
def visualize_prediction(folder_path=DATASET_ROOT, sequence_name="Woman", frame_name="0002.jpg", csv_path=VISUALIZATION_CSV):
    df = pd.read_csv(csv_path)
    match = df[(df["sequence"] == sequence_name) & (df["frame"] == frame_name)]
    if match.empty:
        log.warning("No match found for the specified sequence and frame.")
        return

    row = match.iloc[0]
    img_path = os.path.join(folder_path, sequence_name, "img", frame_name)
    image = cv2.imread(img_path)

    if image is None:
        log.warning(f"Could not load image at: {img_path}. Using placeholder.")
        image = 255 * np.ones((480, 640, 3), dtype=np.uint8)

    gt_box = (int(row["gt_x"]), int(row["gt_y"]), int(row["gt_w"]), int(row["gt_h"]))
    pred_box = (int(row["pred_x"]), int(row["pred_y"]), int(row["pred_w"]), int(row["pred_h"]))

    cv2.rectangle(image, (gt_box[0], gt_box[1]), (gt_box[0]+gt_box[2], gt_box[1]+gt_box[3]), (0, 255, 0), 2)
    cv2.rectangle(image, (pred_box[0], pred_box[1]), (pred_box[0]+pred_box[2], pred_box[1]+pred_box[3]), (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{sequence_name} | Frame: {frame_name} | IoU: {row['iou']:.2f}")
    plt.axis('off')
    # plt.show()

# Optional visualization call
visualize_prediction(sequence_name="Woman", frame_name="0002.jpg")
visualize_prediction(sequence_name="Woman", frame_name="0032.jpg")
visualize_prediction(sequence_name="Woman", frame_name="0100.jpg")
visualize_prediction(sequence_name="Woman", frame_name="0200.jpg")
plt.show()