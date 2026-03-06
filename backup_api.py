# api.py
import os
import cv2
import torch
import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from facenet_pytorch import MTCNN, InceptionResnetV1
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MIN_CONFIDENCE = 0.72  # adjust after checking debug output

app.mount("/dataset/bargad", StaticFiles(directory="dataset/bargad"), name="bargad")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
index = faiss.read_index("face_index.faiss")
labels = np.load("labels.npy", allow_pickle=True)

# safely load sources — fallback to all "bargad" if file missing
if os.path.exists("sources.npy"):
    sources = np.load("sources.npy", allow_pickle=True)
else:
    sources = np.array(["bargad"] * len(labels))

DATASET_DIRS = {
    "bargad": "dataset/bargad",
    "lfw": "lfw-deepfunneled/lfw-deepfunneled",
}

@app.post("/match")
async def match_face(file: UploadFile = File(...), top_k: int = 20):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Could not read image."}

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = mtcnn(img)

        if face is None:
            return {"error": "No face detected in the uploaded image."}

        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face).cpu().numpy().astype("float32")

        faiss.normalize_L2(emb)
        scores, idxs = index.search(emb, top_k)

        # DEBUG — check terminal to fine-tune MIN_CONFIDENCE
        print("\n🔍 Top 5 raw matches:")
        for d in range(5):
            print(f"  Rank {d+1}: {labels[idxs[0][d]]} ({sources[idxs[0][d]]}) → {scores[0][d]:.3f}")

        seen_labels = set()
        results = []

        for i in range(top_k):
            idx = idxs[0][i]
            label = labels[idx]
            source = sources[idx]
            score = round(float(scores[0][i]), 3)

            if score < MIN_CONFIDENCE:  # ← filter irrelevant matches
                continue

            unique_key = f"{source}/{label}"
            if unique_key in seen_labels:
                continue
            seen_labels.add(unique_key)

            person_dir = os.path.join(DATASET_DIRS.get(source, "dataset/bargad"), label)
            person_images = []
            if os.path.isdir(person_dir):
                imgs = [
                    f for f in os.listdir(person_dir)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
                ]
                person_images = [f"/dataset/{source}/{label}/{img}" for img in imgs]

            results.append({
                "label": label,
                "source": source,
                "confidence": score,
                "images": person_images,
                "image_url": person_images[0] if person_images else None
            })

        if not results:
            return {"error": "No confident match found in the dataset."}

        return {"matches": results}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
