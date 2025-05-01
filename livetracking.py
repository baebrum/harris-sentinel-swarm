import cv2
import torch
from torchvision import transforms
from vit_pytorch import ViT
import numpy as np
import time
import argparse

torch.backends.quantized.engine = 'qnnpack'

# ================= ARGPARSE CONFIG =================
parser = argparse.ArgumentParser(description="ViT Face Classification (webcam)")

parser.add_argument(
    "--config",
    choices=["standard", "small", "tiny", "super_tiny"],
    default="standard",
    help="Which ViT configuration to use"
)
args = parser.parse_args()

# ================ PRESETS DEFINITION ================
# Each preset defines: checkpoint filename, dim, depth, heads, mlp_dim
PRESETS = {
    "standard": {
        "vit_path": "vit_target_recognition_standard.pth",
        "dim": 512, "depth": 6, "heads": 8, "mlp_dim": 1024
    },
    "small": {
        "vit_path": "vit_target_recognition_small.pth",
        "dim": 128, "depth": 4, "heads": 4, "mlp_dim": 256
    },
    "tiny": {
        "vit_path": "vit_target_recognition_tiny.pth",
        "dim": 64,  "depth": 2, "heads": 2, "mlp_dim": 128
    },
    "super_tiny": {
        "vit_path": "vit_target_recognition_supertiny.pth",
        "dim": 32,  "depth": 1, "heads": 2, "mlp_dim": 64
    }
}

cfg = PRESETS[args.config]

# ================= CONFIG =================
VIT_PATH    = cfg["vit_path"]
IMG_SIZE    = 224
PATCH_SIZE  = 16
DIM         = cfg["dim"]
DEPTH       = cfg["depth"]
HEADS       = cfg["heads"]
MLP_DIM     = cfg["mlp_dim"]
DROPOUT     = 0.1
EMB_DROPOUT = 0.1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["man", "woman"]  # Your ViT labels
# ==========================================

print(f"[INFO] Loading '{args.config}' ViT — {DIM}-dim, {DEPTH} blocks, {HEADS} heads, MLP {MLP_DIM}")

# Load ViT model
vit = ViT(
    image_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    num_classes=len(LABELS),
    dim=DIM,
    depth=DEPTH,
    heads=HEADS,
    mlp_dim=MLP_DIM,
    dropout=DROPOUT,
    emb_dropout=EMB_DROPOUT
)
vit.load_state_dict(torch.load(VIT_PATH, map_location=DEVICE))
vit.to(DEVICE)
vit.eval()

vit = torch.quantization.quantize_dynamic(vit, {torch.nn.Linear}, dtype=torch.qint8)

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
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 24)

print("Resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

prev_time = time.time()
frame_count = 0
confidence_sum = 0.0
classified_frames = 0
start_time = time.time()
print("[INFO] Running live face-based ViT classification. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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

        label = "male" if man_prob > woman_prob else "female"
        confidence = max(man_prob, woman_prob)
        confidence_sum += confidence
        classified_frames += 1

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} ({confidence*100:.1f}%)",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        # DEBUG: Show crop
        #cv2.imshow("ViT Input Crop", cropped)

    # FPS counter
    curr_time = time.time()
    fps = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(
        frame,
        f"FPS: {fps:.2f}",
        (frame.shape[1] - 180, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 255),
        2
    )

    cv2.imshow("ViT Face Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time()
avg_fps = frame_count / (end_time - start_time)
avg_conf = (confidence_sum / classified_frames * 100) if classified_frames > 0 else 0.0
print(f"Average FPS: {avg_fps:.2f}")
print(f"Average Confidence: {avg_conf:.2f}%")

cap.release()
cv2.destroyAllWindows()
