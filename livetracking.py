import cv2
import torch
from torchvision import transforms
from vit_pytorch import ViT
import numpy as np

# ================= CONFIG =================
VIT_PATH = "/Users/jacobanderson/Documents/Spring 2025/CompE696/compe-696/vit_target_recognition.pth"
YOLO_MODEL = "yolov5s.pt"  # Or yolov5m.pt / yolov5l.pt
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["men, women"]  # Update your ViT labels
# ==========================================

# Load ViT model
vit = ViT(
    image_size=IMG_SIZE,
    patch_size=16,
    num_classes=9,  # MATCH how many classes you trained with!
    dim=512,
    depth=6,
    heads=8,
    mlp_dim=1024,
    dropout=0.1,
    emb_dropout=0.1
)
vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
vit.to(DEVICE)
vit.eval()

# Load YOLOv5 model
yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
yolo.to(DEVICE)

# Preprocess function
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Running live YOLO+ViT tracking. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = yolo(frame)  # <- NOT .predict()

    detections = results.xyxy[0]  # tensor [N, 6]: x1, y1, x2, y2, conf, class

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)

        cropped = frame[y1:y2, x1:x2]

        # Skip very tiny boxes
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue

        # Classify with ViT
        with torch.no_grad():
            input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)
            output = vit(input_tensor)
            class_idx = torch.argmax(output, dim=1).item()
            if class_idx >= len(LABELS):
                class_label = "Unknown"
            else:
                class_label = LABELS[class_idx]

        # Draw bounding box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_label}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("YOLO + ViT Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
