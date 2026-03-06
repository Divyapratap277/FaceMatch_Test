# add_photo.py
import os
import cv2
import torch
import numpy as np
import cloudinary
import cloudinary.uploader
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

client = MongoClient(os.getenv("MONGODB_URI"))
collection = client["facematch"]["faces"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# ---- EDIT THESE ----
IMG_PATH = "dataset/bargad/user/AB_1.jfif.jpeg"  # path to your photo
LABEL    = "user"                                # person name/id
SOURCE   = "bargad"                               # bargad or lfw
# --------------------

img = cv2.imread(IMG_PATH)
if img is None:
    print("❌ Image not found")
    exit()

h, w = img.shape[:2]
if max(h, w) > 800:
    scale = 800 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

with torch.no_grad():
    face = mtcnn(img_rgb)
    if face is None:
        print("❌ No face detected in image")
        exit()

    face = face.unsqueeze(0).to(DEVICE)
    emb = model(face).cpu().numpy()[0].astype("float32")
    emb = emb / np.linalg.norm(emb)

upload = cloudinary.uploader.upload(
    IMG_PATH,
    folder=f"facematch/{SOURCE}/{LABEL}",
    public_id=os.path.splitext(os.path.basename(IMG_PATH))[0],
    overwrite=True
)
image_url = upload["secure_url"]

collection.insert_one({
    "label": LABEL,
    "source": SOURCE,
    "image_url": image_url,
    "embedding": emb.tolist()
})

print(f"✅ Added {LABEL} → {image_url}")
