import cv2
import torch
from torchvision import transforms
from vit_pytorch import ViT
import numpy as np
import time

# ================= CONFIG =================
VIT_PATH = "vit_target_recognition.pth"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["man", "woman"]  # Your ViT labels
# ==========================================

# Load ViT model
vit = ViT(
    image_size=IMG_SIZE,
    patch_size=16,
    num_classes=len(LABELS),
    dim=128,
    depth=4,
    heads=4,
    mlp_dim=256,
    dropout=0.1,
    emb_dropout=0.1
)
vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
vit.to(DEVICE)
vit.eval()

# Preprocessing transform
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)
prev_time = time.time()
print("[INFO] Running live face-based ViT classification. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        cropped = frame[y:y+h, x:x+w]
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            continue

        # Classify with ViT
        with torch.no_grad():
            input_tensor = transform(cropped).unsqueeze(0).to(DEVICE)
            output = vit(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze()
            man_prob = probs[0].item()
            woman_prob = probs[1].item()

        label = "man" if man_prob > woman_prob else "woman"
        confidence = max(man_prob, woman_prob)

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # DEBUG: Show crop
        #cv2.imshow("ViT Input Crop", cropped)

    # FPS counter
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("ViT Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
