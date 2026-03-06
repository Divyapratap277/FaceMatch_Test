# build_db.py
import os
import cv2
import torch
import numpy as np
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from tqdm import tqdm
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv

load_dotenv()

# Cloudinary setup
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# MongoDB setup
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["facematch"]
collection = db["faces"]

# ---- Prompt ----
confirm_drop = input("🗑️ Drop existing data and rebuild everything? (y/n): ")
if confirm_drop.lower() == "y":
    collection.drop()
    print("🗑️ Cleared existing collection\n")
    index_bargad = True
else:
    print("✅ Keeping existing data, appending LFW only...\n")
    index_bargad = False

confirm = input("⚠️ Continue with indexing? (y/n): ")
if confirm.lower() != "y":
    print("Cancelled.")
    exit()

# Model setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using: {DEVICE}\n")
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

total = 0

with torch.no_grad():

    # ========== BARGAD EMPLOYEES ==========
    if index_bargad:
        DATASET_DIR = "dataset/bargad"
        print("👥 Indexing Bargad employees...\n")

        for person in tqdm(os.listdir(DATASET_DIR)):
            person_dir = os.path.join(DATASET_DIR, person)
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                img_path = os.path.join(person_dir, img_file)

                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    h, w = img.shape[:2]
                    if max(h, w) > 800:
                        scale = 800 / max(h, w)
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    face = mtcnn(img_rgb)
                    if face is None:
                        print(f"  ❌ No face: {img_file}")
                        continue

                    face = face.unsqueeze(0).to(DEVICE)
                    emb = model(face).cpu().numpy()[0].astype("float32")
                    emb = emb / np.linalg.norm(emb)

                    upload = cloudinary.uploader.upload(
                        img_path,
                        folder=f"facematch/bargad/{person}",
                        public_id=os.path.splitext(img_file)[0],
                        overwrite=True
                    )
                    image_url = upload["secure_url"]

                    collection.insert_one({
                        "label": person,
                        "source": "bargad",
                        "image_url": image_url,
                        "embedding": emb.tolist()
                    })

                    total += 1
                    print(f"  ✅ [{total}] {person}/{img_file} → Cloudinary ✓")

                except Exception as e:
                    print(f"  ❌ Error on {img_file}: {e}")
                    continue

        print(f"\n✅ Bargad done\n")

    # ========== LFW DATASET ==========
    LFW_DIR = "dataset/lfw-deepfunneled"
    MAX_LFW_IMAGES = 5000
    lfw_count = 0

    print("🌍 Indexing LFW dataset (max 5000 images, 2+ photos per person)...\n")

    for person in tqdm(os.listdir(LFW_DIR)):
        if lfw_count >= MAX_LFW_IMAGES:
            break

        person_dir = os.path.join(LFW_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = [f for f in os.listdir(person_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(imgs) < 2:
            continue  # skip single-photo people

        for img_file in imgs:
            if lfw_count >= MAX_LFW_IMAGES:
                break

            img_path = os.path.join(person_dir, img_file)

            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                face = mtcnn(img_rgb)
                if face is None:
                    print(f"  ❌ No face: {img_file}")
                    continue

                face = face.unsqueeze(0).to(DEVICE)
                emb = model(face).cpu().numpy()[0].astype("float32")
                emb = emb / np.linalg.norm(emb)

                upload = cloudinary.uploader.upload(
                    img_path,
                    folder=f"facematch/lfw/{person}",
                    public_id=os.path.splitext(img_file)[0],
                    overwrite=True
                )
                image_url = upload["secure_url"]

                collection.insert_one({
                    "label": person,
                    "source": "lfw",
                    "image_url": image_url,
                    "embedding": emb.tolist()
                })

                lfw_count += 1
                total += 1
                print(f"  ✅ [{lfw_count}/5000] {person}/{img_file} → Cloudinary ✓")

            except Exception as e:
                print(f"  ❌ Error on {img_file}: {e}")
                continue

    print(f"\n✅ LFW done — {lfw_count} images indexed")

print(f"\n🎉 Total — {total} faces indexed into MongoDB + Cloudinary")
