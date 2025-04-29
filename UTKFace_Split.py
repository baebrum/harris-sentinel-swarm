import os
import shutil
import random
from tqdm import tqdm  # progress bar

# === CONFIGURATION ===
INPUT_FOLDER = "/Users/jacobanderson/Documents/Spring 2025/CompE696/compe-696/UTKFace/UTKFace"   # Change this to your actual UTKFace image folder
OUTPUT_FOLDER = "UTKFace_5000_Split"
N_SAMPLES = 2500

# === Create output directories ===
os.makedirs(os.path.join(OUTPUT_FOLDER, "man"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, "woman"), exist_ok=True)

# === Filter valid files and split by gender ===
male_images = []
female_images = []
skipped = 0

print("Scanning files...")
for img in os.listdir(INPUT_FOLDER):
    try:
        parts = img.split('_')
        if len(parts) < 2:
            raise ValueError("Invalid filename format")
        gender = parts[1]
        if gender == '0':
            male_images.append(img)
        elif gender == '1':
            female_images.append(img)
    except Exception as e:
        skipped += 1
        print(f"Skipping '{img}' — Reason: {e}")

print(f"Found {len(male_images)} male and {len(female_images)} female images")
print(f"Skipped {skipped} malformed files")

# === Random sampling ===
if len(male_images) < N_SAMPLES or len(female_images) < N_SAMPLES:
    raise ValueError("Not enough valid images to sample from.")

selected_males = random.sample(male_images, N_SAMPLES)
selected_females = random.sample(female_images, N_SAMPLES)

# === Copy files to new directories with clean filenames ===
print("\nCopying male images...")
for idx, img in enumerate(tqdm(selected_males)):
    shutil.copy(os.path.join(INPUT_FOLDER, img), os.path.join(OUTPUT_FOLDER, "man", f"{idx:04d}.jpg"))

print("\nCopying female images...")
for idx, img in enumerate(tqdm(selected_females)):
    shutil.copy(os.path.join(INPUT_FOLDER, img), os.path.join(OUTPUT_FOLDER, "woman", f"{idx:04d}.jpg"))

print("\n✅ Dataset creation complete!")
print(f"Saved to: {OUTPUT_FOLDER}")
