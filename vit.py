# -*- coding: utf-8 -*-
import os
import re
import csv
import cv2
import time
import torch
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

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
IMG_SIZE = 224
TRAIN_SPLIT = 0.9
LEARNING_RATE = 3e-4

# Tracking Parameters
SEARCH_RADIUS = 40
STRIDE = 5

# Device config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ==== UTILS ====
def log_time(start, label):
    print(f"🕒 {label} took {time.time() - start:.2f} seconds")

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
        print(f"Failed to compute IoU: {e} | GT: {boxA}, Pred: {boxB}")
        return 0.0

# ==== TRANSFORMS ====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==== LOAD DATA ====
start_time = time.time()
full_dataset = datasets.ImageFolder(DATASET_ROOT, transform=transform)
class_names = full_dataset.classes
train_len = int(TRAIN_SPLIT * len(full_dataset))
test_len = len(full_dataset) - train_len
train_set, test_set = random_split(full_dataset, [train_len, test_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
log_time(start_time, "Data loading")

# ==== VIT MODEL ====
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

# ==== TRAINING ====
start_time = time.time()
for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}, Train Acc: {100 * correct / total:.2f}%")
    log_time(epoch_start, f"Epoch {epoch+1}")
log_time(start_time, "Total training")

# ==== TESTING ====
start_time = time.time()
model.eval()
correct = 0
total = 0

with open(CSV_LOG_PATH, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["image_index", "true_label", "predicted_label", "confidence"])

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_idxs = probs.max(1)
            predicted = top_idxs.item()
            confidence = top_probs.item()
            correct += (predicted == labels.item())
            total += 1
            csv_writer.writerow([
                idx,
                class_names[labels.item()],
                class_names[predicted],
                f"{confidence:.4f}"
            ])
print(f"🎯 Test Accuracy: {100 * correct / total:.2f}%")
log_time(start_time, "Model testing")

# ==== CONFUSION MATRIX ====
# start_time = time.time()
# df = pd.read_csv(CSV_LOG_PATH)
# true_labels = df["true_label"]
# pred_labels = df["predicted_label"]
# labels = sorted(df["true_label"].unique())
# cm = confusion_matrix(true_labels, pred_labels, labels=labels)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("ViT Classifier - Confusion Matrix")
# plt.tight_layout()
# # plt.show()
# log_time(start_time, "Confusion matrix generation")

# ==== SAVE MODEL ====
start_time = time.time()
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")
log_time(start_time, "Model saving")

# ==== TRACKING / CLASSIFICATION ====
start_time = time.time()
dataset = datasets.ImageFolder(DATASET_ROOT)
class_names = dataset.classes
state_dict = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
model.load_state_dict(state_dict, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def classify_patch(img_patch):
    input_tensor = transform(img_patch).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        return class_names[pred_idx], probs[pred_idx].item()

all_results = []
for seq_name in sorted(os.listdir(DATASET_ROOT)):
    if seq_name != "Woman":
        continue
    seq_time = time.time()
    seq_path = os.path.join(DATASET_ROOT, seq_name)
    gt_path = os.path.join(seq_path, "groundtruth_rect.txt")
    img_folder = os.path.join(seq_path, "img")
    if not os.path.exists(gt_path) or not os.path.isdir(img_folder):
        continue

    with open(gt_path, "r") as f:
        gt_boxes = [list(map(int, re.split(r"[\s,]+", line.strip()))) for line in f if line.strip()]

    frame_files = sorted(os.listdir(img_folder))
    predicted_boxes = []
    processed_frames = []
    iou_scores = []

    first_frame = cv2.imread(os.path.join(img_folder, frame_files[0]))
    if first_frame is None:
        print(f"⚠️ Skipping sequence '{seq_name}' due to missing first frame.")
        continue

    x, y, w, h = gt_boxes[0]
    crop = first_frame[y:y+h, x:x+w]
    true_label, _ = classify_patch(crop)
    predicted_boxes.append([x, y, w, h])
    iou_scores.append(1.0)
    processed_frames.append(frame_files[0])

    for i in range(1, len(frame_files)):
        frame = cv2.imread(os.path.join(img_folder, frame_files[i]))
        if frame is None:
            print(f"Skipping frame {frame_files[i]} (could not load)")
            continue

        prev_x, prev_y, prev_w, prev_h = predicted_boxes[-1]
        candidates = []
        positions = []

        for dx in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
            for dy in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1, STRIDE):
                cx = max(0, prev_x + dx)
                cy = max(0, prev_y + dy)
                if cx + prev_w > frame.shape[1] or cy + prev_h > frame.shape[0]:
                    continue
                crop = frame[cy:cy + prev_h, cx:cx + prev_w]
                try:
                    tensor = transform(crop).unsqueeze(0)
                    candidates.append(tensor)
                    positions.append([cx, cy])
                except:
                    continue

        if not candidates:
            best_box = predicted_boxes[-1]
        else:
            batch_tensor = torch.cat(candidates).to(DEVICE)
            with torch.no_grad():
                outputs = model(batch_tensor)
                probs = F.softmax(outputs, dim=1)
                class_idx = class_names.index(true_label)
                confidences = probs[:, class_idx]
            max_idx = torch.argmax(confidences).item()
            best_box = [positions[max_idx][0], positions[max_idx][1], prev_w, prev_h]

        predicted_boxes.append(best_box)
        iou = safe_iou(gt_boxes[i], best_box)
        iou_scores.append(iou)
        processed_frames.append(frame_files[i])

    for i in range(min(len(processed_frames), len(gt_boxes))):
        all_results.append({
            "sequence": seq_name,
            "frame": processed_frames[i],
            "gt_x": gt_boxes[i][0],
            "gt_y": gt_boxes[i][1],
            "gt_w": gt_boxes[i][2],
            "gt_h": gt_boxes[i][3],
            "pred_x": predicted_boxes[i][0],
            "pred_y": predicted_boxes[i][1],
            "pred_w": predicted_boxes[i][2],
            "pred_h": predicted_boxes[i][3],
            "iou": iou_scores[i]
        })
    log_time(seq_time, f"Sequence '{seq_name}'")

log_time(start_time, "All tracking and classification")

# ==== SAVE RESULTS ====
start_time = time.time()
results_df = pd.DataFrame(all_results)
results_df.to_csv(CSV_OUTPUT, index=False)
print(f"Tracking complete for all sequences. Results saved to {CSV_OUTPUT}")

# Calculate and display overall IoU
if 'iou' in results_df.columns and not results_df['iou'].empty:
    print(f"Overall Average IoU: {results_df['iou'].mean():.4f}")

    # ==== CALCULATE AVERAGE IOU PER SEQUENCE ====
    avg_iou_per_sequence = results_df.groupby("sequence")["iou"].mean().reset_index()
    avg_iou_per_sequence.columns = ["sequence", "average_iou"]
    avg_iou_per_sequence = avg_iou_per_sequence.sort_values(by="average_iou", ascending=False)

    print("\n=== Average IoU per Sequence ===")
    print(avg_iou_per_sequence.to_string(index=False))


    # Optional: Save to CSV
    avg_iou_per_sequence.to_csv("average_iou_per_sequence.csv", index=False)
else:
    print("No valid IoU values were recorded.")

log_time(start_time, "Saving results")


# ==== VISUALIZATION FUNCTION ====
def visualize_prediction(folder_path=DATASET_ROOT, sequence_name="Woman", frame_name="0002.jpg", csv_path=VISUALIZATION_CSV):
    df = pd.read_csv(csv_path)
    match = df[(df["sequence"] == sequence_name) & (df["frame"] == frame_name)]
    if match.empty:
        print("No match found for the specified sequence and frame.")
        return

    row = match.iloc[0]
    img_path = os.path.join(folder_path, sequence_name, "img", frame_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Could not load image at: {img_path}. Using placeholder.")
        image = 255 * np.ones((480, 640, 3), dtype=np.uint8)

    gt_x, gt_y, gt_w, gt_h = int(row["gt_x"]), int(row["gt_y"]), int(row["gt_w"]), int(row["gt_h"])
    pred_x, pred_y, pred_w, pred_h = int(row["pred_x"]), int(row["pred_y"]), int(row["pred_w"]), int(row["pred_h"])

    cv2.rectangle(image, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h), (0, 255, 0), 2)
    cv2.rectangle(image, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"{sequence_name} | Frame: {frame_name} | IoU: {row['iou']:.2f}")
    plt.axis('off')
    # plt.show()

# Optional: call to visualize
visualize_prediction(sequence_name="Woman", frame_name="0002.jpg")
plt.show()