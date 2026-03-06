# api.py
import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import uvicorn

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["facematch"]
collection = db["faces"]



# Models
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

MIN_CONFIDENCE_BARGAD = 0.72  # strict for employees
MIN_CONFIDENCE_LFW = 0.55     # relaxed for celebrities
TOP_K = 50

@app.post("/match")
async def match_face(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Could not read image."}

        # Resize large images
        h, w = img.shape[:2]
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            face = mtcnn(img)
        except Exception:
            return {"error": "Face detection failed. Try another photo."}

        if face is None:
            return {"error": "No face detected in the uploaded image."}

        face = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(face).cpu().numpy()[0].astype("float32")
        emb = emb / np.linalg.norm(emb)

        # MongoDB Atlas Vector Search
        all_docs = list(collection.find({}, {"label": 1, "source": 1, "image_url": 1, "embedding": 1}))
        print(f"📦 Loaded {len(all_docs)} docs from MongoDB")  # ADD THIS
        raw_results = []
        for doc in all_docs:
            db_emb = np.array(doc["embedding"], dtype="float32")
            db_emb = db_emb / np.linalg.norm(db_emb)
            score = float(np.dot(emb, db_emb))
            raw_results.append({
                "label": doc["label"],
                "source": doc["source"],
                "image_url": doc["image_url"],
                "score": round(score, 3)
            })

        raw_results.sort(key=lambda x: x["score"], reverse=True)
        raw_results = raw_results[:TOP_K]

        # Debug
        print("\n🔍 Top 5 raw matches:")
        for r in raw_results[:5]:
            print(f"  {r['label']} ({r['source']}) → {r['score']:.3f}")

        # Group all images per person
        seen = {}
        for r in raw_results:
            key = f"{r['source']}/{r['label']}"
            score = round(float(r["score"]), 3)

            # Different threshold per source
            threshold = MIN_CONFIDENCE_BARGAD if r["source"] == "bargad" else MIN_CONFIDENCE_LFW
            if score < threshold:
                continue

            if key not in seen:
                seen[key] = {
                    "label": r["label"],
                    "source": r["source"],
                    "confidence": score,
                    "images": [r["image_url"]],
                    "image_url": r["image_url"]
                }
            else:
                if r["image_url"] not in seen[key]["images"]:
                    seen[key]["images"].append(r["image_url"])
                if score > seen[key]["confidence"]:
                    seen[key]["confidence"] = score
                    seen[key]["image_url"] = r["image_url"]

        results = list(seen.values())

        if not results:
            return {"error": "No confident match found in the dataset."}

        return {"matches": results}

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8001, reload=False)
