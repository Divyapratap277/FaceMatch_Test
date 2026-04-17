import os
import cv2
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import uvicorn
from datetime import datetime
from typing import Optional

load_dotenv()

app = FastAPI()

# CORS: browsers require Access-Control-Allow-Origin on cross-origin fetch.
# Explicit list + regex covers facematch / any *.bargad.ai HTTPS (Railway + Vercel + local).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://facematch.bargad.ai",
        "https://www.facematch.bargad.ai",
        "https://face-match-test-xgua.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
    ],
    allow_origin_regex=r"^https://([a-z0-9-]+\.)*bargad\.ai$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["facematch"]
collection = db["faces"]
auth_logs = db["auth_logs"]   # NEW: stores geo + liveness logs

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

MIN_CONFIDENCE_BARGAD = 0.72
MIN_CONFIDENCE_LFW = 0.55
TOP_K = 50


# ── MODIFIED /match: accepts optional geo fields ──
@app.post("/match")
async def match_face(
    file: UploadFile = File(...),
    geo_lat: Optional[str] = Form(None),
    geo_long: Optional[str] = Form(None),
    geo_timestamp: Optional[str] = Form(None),
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Could not read image."}

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

        all_docs = list(collection.find({}, {"label": 1, "source": 1, "image_url": 1, "embedding": 1}))
        print(f"📦 Loaded {len(all_docs)} docs from MongoDB")

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

        print("\n🔍 Top 5 raw matches:")
        for r in raw_results[:5]:
            print(f"  {r['label']} ({r['source']}) → {r['score']:.3f}")

        seen = {}
        for r in raw_results:
            key = f"{r['source']}/{r['label']}"
            score = round(float(r["score"]), 3)
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

        # ── NEW: Log geo data ──
        if geo_lat and geo_long:
            auth_logs.insert_one({
                "timestamp": geo_timestamp or datetime.utcnow().isoformat(),
                "geo_lat": geo_lat,
                "geo_long": geo_long,
                "top_match": results[0]["label"] if results else "no_match",
                "match_count": len(results),
                "logged_at": datetime.utcnow()
            })
            print(f"📍 Geo logged: {geo_lat}, {geo_long}")

        if not results:
            return {"error": "No confident match found in the dataset."}

        return {"matches": results}

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── NEW: Passive liveness endpoint ──
@app.post("/liveness")
async def check_liveness(file: UploadFile = File(...)):
    temp_path = f"temp_live_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"live": False, "score": 0.0, "reason": "Cannot read image"}

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Laplacian variance — printed/replayed photos are blurry
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Frequency analysis — real faces have natural high-freq texture
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft_shift) + 1)
        freq_score = float(np.mean(magnitude))

        # Combine into a liveness score (0 to 1)
        lap_norm = min(lap_var / 500.0, 1.0)
        freq_norm = min(freq_score / 100.0, 1.0)
        score = round((lap_norm * 0.6 + freq_norm * 0.4), 3)

        is_live = score > 0.35

        print(f"🧪 Liveness — Laplacian: {lap_var:.1f}, Freq: {freq_score:.1f}, Score: {score}")

        return {
            "live": is_live,
            "score": score,
            "reason": "OK" if is_live else "Possible spoof: low texture/sharpness detected"
        }

    except Exception as e:
        return {"live": False, "score": 0.0, "reason": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
