import os
import cv2
import torch
import numpy as np
import uuid
import secrets
from fastapi import FastAPI, File, UploadFile, Form, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from facenet_pytorch import MTCNN, InceptionResnetV1
from dotenv import load_dotenv
import uvicorn
from datetime import datetime, timedelta
from typing import Optional, List, Tuple, Any, Dict

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
liveness_sessions = db["liveness_sessions"]

# Must match frontend gesture IDs (10 distinct challenges in the pool).
ALL_GESTURE_IDS = [
    "turn_left",
    "turn_right",
    "nod",
    "look_up",
    "smile",
    "surprised",
    "mouth_open",
    "blink",
    "tilt_head",
    "angry",
]

SESSION_TTL_MINUTES = 15
SESSION_ISSUE_MAX_ATTEMPTS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mtcnn = MTCNN(image_size=160, margin=20, device=DEVICE)
model = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

MIN_CONFIDENCE_BARGAD = 0.72
MIN_CONFIDENCE_LFW = 0.55
TOP_K = 50


@app.on_event("startup")
def _ensure_liveness_indexes():
    try:
        liveness_sessions.create_index(
            [("device_id", 1), ("sequence_key", 1)],
            unique=True,
            name="uniq_device_sequence",
        )
        liveness_sessions.create_index(
            "session_id",
            unique=True,
            name="uniq_session_id",
        )
    except Exception as e:
        print(f"Liveness index warning: {e}")


def _passive_liveness_from_bgr(img_bgr: np.ndarray) -> Tuple[bool, float, str]:
    """Texture / sharpness heuristics for print / screen replay (single frame)."""
    if img_bgr is None or img_bgr.size == 0:
        return False, 0.0, "Empty image"
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    dft = np.fft.fft2(gray)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft_shift) + 1)
    freq_score = float(np.mean(magnitude))
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mean = float(np.mean(np.sqrt(sobel_x ** 2 + sobel_y ** 2)))

    lap_norm = min(lap_var / 420.0, 1.0)
    freq_norm = min(freq_score / 95.0, 1.0)
    grad_norm = min(grad_mean / 35.0, 1.0)
    score = round(lap_norm * 0.45 + freq_norm * 0.35 + grad_norm * 0.2, 3)
    # Stricter gate for match path than standalone /liveness demo.
    is_live = score > 0.42
    reason = "OK" if is_live else "Possible spoof: low texture / sharpness / edge energy"
    return is_live, score, reason


def _pick_random_gestures_four() -> List[str]:
    ids = list(ALL_GESTURE_IDS)
    secrets.SystemRandom().shuffle(ids)
    return ids[:4]


@app.post("/liveness/session")
async def create_liveness_session(payload: Dict[str, Any] = Body(...)):
    device_id = (payload or {}).get("device_id")
    if not device_id or not isinstance(device_id, str) or len(device_id) > 128:
        raise HTTPException(status_code=400, detail="Invalid device_id")

    for _ in range(SESSION_ISSUE_MAX_ATTEMPTS):
        gestures = _pick_random_gestures_four()
        sequence_key = "|".join(gestures)
        session_id = str(uuid.uuid4())
        expires_at = datetime.utcnow() + timedelta(minutes=SESSION_TTL_MINUTES)
        doc = {
            "session_id": session_id,
            "device_id": device_id,
            "gestures": gestures,
            "sequence_key": sequence_key,
            "status": "issued",
            "expires_at": expires_at,
            "created_at": datetime.utcnow(),
        }
        try:
            liveness_sessions.insert_one(doc)
            return {"session_id": session_id, "gestures": gestures}
        except DuplicateKeyError:
            continue

    raise HTTPException(
        status_code=409,
        detail="Could not issue a new gesture sequence for this device (all combinations may be exhausted).",
    )


@app.post("/liveness/session/complete")
async def complete_liveness_session(payload: Dict[str, Any] = Body(...)):
    session_id = (payload or {}).get("session_id")
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")

    now = datetime.utcnow()
    res = liveness_sessions.update_one(
        {
            "session_id": session_id,
            "status": "issued",
            "expires_at": {"$gt": now},
        },
        {"$set": {"status": "completed", "completed_at": now}},
    )
    if res.matched_count == 0:
        raise HTTPException(
            status_code=400,
            detail="Invalid or expired session, or already completed.",
        )
    return {"ok": True}


@app.post("/liveness/temporal")
async def liveness_temporal(
    session_id: str = Form(...),
    device_id: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Short burst of frames: static / periodic replay heuristics."""
    if len(files) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 frames")

    sess = liveness_sessions.find_one(
        {"session_id": session_id, "device_id": device_id, "status": "issued"}
    )
    if not sess or sess.get("expires_at") < datetime.utcnow():
        raise HTTPException(status_code=400, detail="Invalid or expired session")

    lum: List[float] = []
    lap_vars: List[float] = []
    for uf in files:
        raw = await uf.read()
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lum.append(float(np.mean(gray)))
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

    if len(lum) < 4:
        raise HTTPException(status_code=400, detail="Could not decode frames")

    lum = np.array(lum, dtype=np.float64)
    diffs = np.diff(lum)
    motion = float(np.std(diffs))
    lap_std = float(np.std(lap_vars)) if len(lap_vars) >= 2 else 0.0

    # Low motion across burst suggests frozen replay; strong periodicity in diffs suggests refresh rate.
    replay_risk = 0.0
    if motion < 0.35:
        replay_risk += 0.45
    if lap_std < 1.5:
        replay_risk += 0.25

    if len(diffs) >= 6:
        z = diffs - np.mean(diffs)
        spec = np.abs(np.fft.rfft(z))
        spec = spec / (np.max(spec) + 1e-6)
        freqs = np.fft.rfftfreq(len(z), d=1.0)
        # Rough band for 24–30 Hz if ~10 fps sampling → map to index band
        mask = (freqs > 0.18) & (freqs < 0.45)
        if np.any(mask) and float(np.max(spec[mask])) > 0.55:
            replay_risk += 0.35

    replay_risk = min(1.0, round(replay_risk, 3))
    ok = replay_risk < 0.72
    return {"ok": ok, "replay_risk": replay_risk, "motion_std": round(motion, 4)}


# ── MODIFIED /match: accepts optional geo fields ──
@app.post("/match")
async def match_face(
    file: UploadFile = File(...),
    geo_lat: Optional[str] = Form(None),
    geo_long: Optional[str] = Form(None),
    geo_timestamp: Optional[str] = Form(None),
    liveness_session_id: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        img = cv2.imread(temp_path)
        if img is None:
            return {"error": "Could not read image."}

        if not liveness_session_id or not device_id:
            return {"error": "Liveness verification required. Complete the camera liveness steps, then try again."}

        now = datetime.utcnow()
        sess = liveness_sessions.find_one(
            {
                "session_id": liveness_session_id,
                "device_id": device_id,
                "status": "completed",
                "expires_at": {"$gt": now},
            }
        )
        if not sess:
            return {"error": "Invalid or expired liveness session. Please start the camera flow again."}

        live_ok, live_score, live_reason = _passive_liveness_from_bgr(img)
        if not live_ok:
            return {
                "error": f"Liveness check failed on selfie ({live_reason}). Score={live_score:.3f}",
            }

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

        liveness_sessions.update_one(
            {"session_id": liveness_session_id},
            {"$set": {"status": "consumed", "consumed_at": datetime.utcnow()}},
        )
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

        is_live, score, reason = _passive_liveness_from_bgr(img)
        print(f"🧪 Liveness — Score: {score}, live={is_live}")

        return {"live": is_live, "score": score, "reason": reason}

    except Exception as e:
        return {"live": False, "score": 0.0, "reason": str(e)}

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
