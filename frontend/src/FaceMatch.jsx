import React, { useState, useRef, useEffect, useCallback } from "react";
import * as faceapi from "face-api.js";
import "./FaceMatch.css";
import bargadLogo from "./bargad-logo.png";
import bargadBranding from "./bargad-branding (1).svg?url";
import { MapContainer, TileLayer, Marker } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
/** Match API can be slow (Render cold start, large DB scan). */
const MATCH_REQUEST_TIMEOUT_MS = 30_000;

const DEVICE_KEY = "facematch_device_id";
const SUSTAINED_FRAMES = 4;

/** Gesture ids are issued by the server; labels here must stay in sync with `ALL_GESTURE_IDS` in api.py. */
const CHALLENGE_TEXT = {
  turn_left: "↩️ Turn your head LEFT",
  turn_right: "↪️ Turn your head RIGHT",
  nod: "↕️ NOD your head down",
  look_up: "⬆️ LOOK slightly up",
  smile: "😊 Please SMILE",
  surprised: "😲 Look SURPRISED",
  mouth_open: "😮 OPEN your mouth wide",
  blink: "😑 BLINK both eyes once",
  tilt_head: "↔️ TILT head toward one shoulder (ear line)",
  angry: "😠 Make an ANGRY / frowning face",
};

function getOrCreateDeviceId() {
  try {
    let id = localStorage.getItem(DEVICE_KEY);
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem(DEVICE_KEY, id);
    }
    return id;
  } catch {
    return `anon_${Date.now()}_${Math.random().toString(36).slice(2, 11)}`;
  }
}

function landmarkEAR(pts, startIdx) {
  const p = (i) => pts[startIdx + i];
  const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const v1 = dist(p(1), p(5));
  const v2 = dist(p(2), p(4));
  const h = dist(p(0), p(3));
  return (v1 + v2) / (2 * h + 1e-6);
}

function eyeTiltAngleRad(pts) {
  return Math.atan2(pts[45].y - pts[36].y, pts[45].x - pts[36].x);
}

/** Smallest absolute difference between two orientations (radians). */
function headTiltDeltaRad(current, baseline) {
  let d = current - baseline;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return Math.abs(d);
}

async function captureBurstFromVideo(video, count, delayMs) {
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const ctx = c.getContext("2d");
  const out = [];
  for (let i = 0; i < count; i++) {
    ctx.drawImage(video, 0, 0);
    const blob = await new Promise((res) => c.toBlob((b) => res(b), "image/jpeg", 0.52));
    if (blob) out.push(blob);
    await new Promise((r) => setTimeout(r, delayMs));
  }
  return out;
}

function UserAvatarIcon() {
  return (
    <svg
      className="fm-profile-avatar-svg"
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden
    >
      <path
        d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v1c0 .55.45 1 1 1h14c.55 0 1-.45 1-1v-1c0-2.66-5.33-4-8-4z"
        fill="currentColor"
      />
    </svg>
  );
}

export default function FaceMatch({ userEmail, onLogout }) {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [hoverCardIndex, setHoverCardIndex] = useState(null);
  const [selectedImg, setSelectedImg] = useState(null);
  const [progress, setProgress] = useState(0);
  const [showCamera, setShowCamera] = useState(false);
  const [stream, setStream] = useState(null);

  // Liveness + Geo states
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [challengeIndex, setChallengeIndex] = useState(0);
  const [completedChallenges, setCompletedChallenges] = useState([]);
  const [livenessLive, setLivenessLive] = useState(false);
  const [challengeMsg, setChallengeMsg] = useState("");
  const [geoData, setGeoData] = useState(null);
  const [geoError, setGeoError] = useState(null);
  const [geoAddress, setGeoAddress] = useState(null);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);
  const profileMenuRef = useRef(null);
  const [sessionChallenges, setSessionChallenges] = useState([]);
  const [livenessSessionLoading, setLivenessSessionLoading] = useState(false);
  const [canMatch, setCanMatch] = useState(false);

  // Refs
  const videoRef = useRef();
  const canvasRef = useRef();
  const overlayCanvasRef = useRef();
  const inputRef = useRef();
  const intervalRef = useRef(null);
  const meshIntervalRef = useRef(null);
  const challengeIndexRef = useRef(0);
  const completedRef = useRef([]);
  const isDetectingRef = useRef(false);
  const isMeshDetectingRef = useRef(false);
  const baselineRef = useRef(null);
  const noseYHistoryRef = useRef([]);
  const noseCenterXHistoryRef = useRef([]);
  const faceWidthHistoryRef = useRef([]);
  const calEyeAngRef = useRef([]);
  const calLeftEarRef = useRef([]);
  const calRightEarRef = useRef([]);
  const faceLostCountRef = useRef(0);
  const challengeCooldownRef = useRef(false); // prevent double-firing
  const sessionChallengesRef = useRef([]);
  const livenessSessionIdRef = useRef(null);
  const livenessCompletedRef = useRef(false);
  const sustainCountRef = useRef(0);
  const blinkSawLowRef = useRef(false);
  const postVerifyRunningRef = useRef(false);

  // Load face-api models once
  useEffect(() => {
    const load = async () => {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri("/models"),
          faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
          faceapi.nets.faceExpressionNet.loadFromUri("/models"),
        ]);
        setModelsLoaded(true);
        console.log("✅ face-api models loaded");
      } catch (e) {
        console.warn("face-api models failed to load:", e);
      }
    };
    load();
  }, []);

  useEffect(() => {
    if (!profileMenuOpen) return;
    const onDocMouseDown = (e) => {
      const el = profileMenuRef.current;
      if (el && !el.contains(e.target)) setProfileMenuOpen(false);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [profileMenuOpen]);

  // High accuracy geo capture
  const captureGeo = useCallback(() => {
    return new Promise((resolve) => {
      if (!navigator.geolocation) return resolve(null);
      navigator.geolocation.getCurrentPosition(
        (pos) => resolve({
          lat: pos.coords.latitude.toFixed(7),
          long: pos.coords.longitude.toFixed(7),
          timestamp: new Date().toISOString(),
        }),
        () => resolve(null),
        { enableHighAccuracy: true, maximumAge: 0, timeout: 12000 }
      );
    });
  }, []);

  // Reverse geocode — BigDataCloud (free, no key)
  const reverseGeocode = useCallback(async (lat, long) => {
    try {
      const res = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?latitude=${lat}&longitude=${long}&localityLanguage=en`
      );
      const data = await res.json();
      const parts = [data.locality, data.principalSubdivision, data.countryName].filter(Boolean);
      return {
        city: data.locality || data.city || "",
        state: data.principalSubdivision || "",
        country: data.countryName || "",
        full: data.localityInfo?.administrative?.map((a) => a.name).filter(Boolean).join(", ") || parts.join(", "),
        short: parts.join(", "),
      };
    } catch {
      return null;
    }
  }, []);

  // ── MESH DRAW (visual only) ──
  const drawFaceMesh = (detection, video) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !video) return;
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!detection) return;

    const pts = detection.landmarks.positions;
    const connections = [
      [0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],
      [8,9],[9,10],[10,11],[11,12],[12,13],[13,14],[14,15],[15,16],
      [17,18],[18,19],[19,20],[20,21],
      [22,23],[23,24],[24,25],[25,26],
      [27,28],[28,29],[29,30],
      [30,31],[31,32],[32,33],[33,34],[34,35],
      [36,37],[37,38],[38,39],[39,40],[40,41],[41,36],
      [42,43],[43,44],[44,45],[45,46],[46,47],[47,42],
      [48,49],[49,50],[50,51],[51,52],[52,53],[53,54],
      [54,55],[55,56],[56,57],[57,58],[58,59],[59,48],
      [60,61],[61,62],[62,63],[63,64],[64,65],[65,66],[66,67],[67,60],
    ];

    ctx.strokeStyle = "rgba(0, 255, 170, 0.65)";
    ctx.lineWidth = 1.5;
    connections.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(pts[a].x, pts[a].y);
      ctx.lineTo(pts[b].x, pts[b].y);
      ctx.stroke();
    });

    pts.forEach((pt) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 2.5, 0, 2 * Math.PI);
      ctx.fillStyle = "rgba(0, 255, 170, 0.9)";
      ctx.fill();
    });

    // Yellow highlight on active challenge region
    const challengeColor = "rgba(255, 215, 0, 0.9)";
    const list = sessionChallengesRef.current;
    const challenge = list[challengeIndexRef.current];
    if (challenge === "smile" || challenge === "surprised" || challenge === "angry" || challenge === "mouth_open") {
      ctx.beginPath();
      pts.slice(48, 68).forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
      ctx.strokeStyle = challengeColor;
      ctx.lineWidth = 2.5;
      ctx.stroke();
    } else if (
      challenge === "turn_left" ||
      challenge === "turn_right" ||
      challenge === "nod" ||
      challenge === "look_up"
    ) {
      [27, 28, 29, 30].forEach((idx) => {
        ctx.beginPath();
        ctx.arc(pts[idx].x, pts[idx].y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = challengeColor;
        ctx.fill();
      });
    } else if (challenge === "tilt_head") {
      ctx.beginPath();
      ctx.moveTo(pts[36].x, pts[36].y);
      ctx.lineTo(pts[45].x, pts[45].y);
      ctx.strokeStyle = challengeColor;
      ctx.lineWidth = 3;
      ctx.stroke();
    } else if (challenge === "blink") {
      [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47].forEach((idx) => {
        ctx.beginPath();
        ctx.arc(pts[idx].x, pts[idx].y, 3.5, 0, 2 * Math.PI);
        ctx.fillStyle = challengeColor;
        ctx.fill();
      });
    }
  };

  // ── MESH LOOP — purely visual, slow 500ms ──
  const runMeshLoop = useCallback(async () => {
    if (isMeshDetectingRef.current) return;
    if (!videoRef.current || videoRef.current.readyState < 2) return;
    isMeshDetectingRef.current = true;
    try {
      const detection = await faceapi
        .detectSingleFace(videoRef.current, new faceapi.TinyFaceDetectorOptions({ inputSize: 224, scoreThreshold: 0.3 }))
        .withFaceLandmarks();
      drawFaceMesh(detection || null, videoRef.current);
    } catch (_) {}
    isMeshDetectingRef.current = false;
  }, []);

  const markChallengeDoneRef = useRef(() => {});
  markChallengeDoneRef.current = (name) => {
    if (challengeCooldownRef.current) return;
    challengeCooldownRef.current = true;
    setTimeout(() => {
      challengeCooldownRef.current = false;
    }, 1000);

    sustainCountRef.current = 0;
    blinkSawLowRef.current = false;

    const newCompleted = [...completedRef.current, name];
    completedRef.current = newCompleted;
    const nextIndex = challengeIndexRef.current + 1;
    challengeIndexRef.current = nextIndex;
    setCompletedChallenges([...newCompleted]);
    setChallengeIndex(nextIndex);

    const total = sessionChallengesRef.current.length;
    if (nextIndex >= total) {
      clearInterval(intervalRef.current);
      clearInterval(meshIntervalRef.current);
      intervalRef.current = null;
      meshIntervalRef.current = null;
      if (!postVerifyRunningRef.current && livenessSessionIdRef.current) {
        postVerifyRunningRef.current = true;
        setChallengeMsg("Verifying motion…");
        (async () => {
          try {
            const deviceId = getOrCreateDeviceId();
            const video = videoRef.current;
            if (!video || video.readyState < 2) {
              setError("Video not ready for verification.");
              postVerifyRunningRef.current = false;
              return;
            }
            const blobs = await captureBurstFromVideo(video, 6, 130);
            const fd = new FormData();
            fd.append("session_id", livenessSessionIdRef.current);
            fd.append("device_id", deviceId);
            blobs.forEach((b) => fd.append("files", b, "frame.jpg"));
            const tr = await fetch(`${API_URL}/liveness/temporal`, { method: "POST", body: fd });
            const tj = await tr.json().catch(() => ({}));
            if (!tr.ok || !tj.ok) {
              setError(
                typeof tj.detail === "string"
                  ? tj.detail
                  : "Motion/replay check failed. Please try again (use live camera, not a screen recording)."
              );
              postVerifyRunningRef.current = false;
              const media = videoRef.current?.srcObject;
              if (media && "getTracks" in media) media.getTracks().forEach((t) => t.stop());
              setStream(null);
              setShowCamera(false);
              setSessionChallenges([]);
              sessionChallengesRef.current = [];
              livenessSessionIdRef.current = null;
              livenessCompletedRef.current = false;
              setCanMatch(false);
              return;
            }
            const cr = await fetch(`${API_URL}/liveness/session/complete`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ session_id: livenessSessionIdRef.current }),
            });
            if (!cr.ok) {
              const ej = await cr.json().catch(() => ({}));
              setError(typeof ej.detail === "string" ? ej.detail : "Could not complete liveness session.");
              postVerifyRunningRef.current = false;
              setCanMatch(false);
              const media2 = videoRef.current?.srcObject;
              if (media2 && "getTracks" in media2) media2.getTracks().forEach((t) => t.stop());
              setStream(null);
              setShowCamera(false);
              return;
            }
            livenessCompletedRef.current = true;
            setCanMatch(true);
            setLivenessLive(true);
            setChallengeMsg("✅ Liveness confirmed! Click Capture.");
          } catch {
            setError("Verification request failed.");
            const media3 = videoRef.current?.srcObject;
            if (media3 && "getTracks" in media3) media3.getTracks().forEach((t) => t.stop());
            setStream(null);
            setShowCamera(false);
          } finally {
            postVerifyRunningRef.current = false;
          }
        })();
      }
    } else {
      const nextId = sessionChallengesRef.current[nextIndex];
      setChallengeMsg(`✅ Done! Now: ${CHALLENGE_TEXT[nextId] || nextId}`);
    }
  };

  const registerSustained = (ok, name, minFrames = SUSTAINED_FRAMES) => {
    if (ok) {
      sustainCountRef.current += 1;
      if (sustainCountRef.current >= minFrames) {
        markChallengeDoneRef.current(name);
        sustainCountRef.current = 0;
      }
    } else {
      sustainCountRef.current = 0;
    }
  };

  // ── LIVENESS LOOP — detection only, no mesh, fast 150ms ──
  const runLivenessLoop = useCallback(async () => {
    if (isDetectingRef.current) return;
    if (!videoRef.current || videoRef.current.readyState < 2) return;
    const seqLen = sessionChallengesRef.current.length;
    if (seqLen === 0 || challengeIndexRef.current >= seqLen) return;

    isDetectingRef.current = true;
    const video = videoRef.current;

    try {
      const detection = await faceapi
        .detectSingleFace(
          video,
          new faceapi.TinyFaceDetectorOptions({
            inputSize: 320,
            scoreThreshold: 0.3,
          })
        )
        .withFaceLandmarks()
        .withFaceExpressions();

      if (detection) {
        faceLostCountRef.current = 0;
        const pts = detection.landmarks.positions;
        const expr = detection.expressions;

        // ── CALIBRATE BASELINE over 15 frames ──
        if (!baselineRef.current) {
          const faceCenterX = (pts[0].x + pts[16].x) / 2;
          noseYHistoryRef.current.push(pts[30].y);
          noseCenterXHistoryRef.current.push(pts[30].x - faceCenterX);
          faceWidthHistoryRef.current.push(Math.abs(pts[16].x - pts[0].x));
          calEyeAngRef.current.push(eyeTiltAngleRad(pts));
          calLeftEarRef.current.push(landmarkEAR(pts, 36));
          calRightEarRef.current.push(landmarkEAR(pts, 42));

          if (noseYHistoryRef.current.length >= 15) {
            const sortedY = [...noseYHistoryRef.current].sort((a, b) => a - b);
            const sortedX = [...noseCenterXHistoryRef.current].sort((a, b) => a - b);
            const sortedFW = [...faceWidthHistoryRef.current].sort((a, b) => a - b);
            const eyeAngles = [...calEyeAngRef.current].sort((a, b) => a - b);
            const les = [...calLeftEarRef.current].sort((a, b) => a - b);
            const res = [...calRightEarRef.current].sort((a, b) => a - b);
            const mid = Math.floor(sortedY.length / 2);
            const eMid = Math.floor(eyeAngles.length / 2);

            baselineRef.current = {
              noseTipY: sortedY[mid],
              noseCenterX: sortedX[mid],
              faceWidth: sortedFW[mid],
              eyeAngle: eyeAngles[eMid] ?? 0,
              leftEAR: les[eMid] ?? 0.25,
              rightEAR: res[eMid] ?? 0.25,
            };
            noseYHistoryRef.current = [];
            noseCenterXHistoryRef.current = [];
            faceWidthHistoryRef.current = [];
            calEyeAngRef.current = [];
            calLeftEarRef.current = [];
            calRightEarRef.current = [];
            console.log("✅ Baseline ready:", baselineRef.current);
          } else {
            console.log(`📊 Calibrating ${noseYHistoryRef.current.length}/15...`);
          }
          isDetectingRef.current = false;
          return;
        }

        const base = baselineRef.current;
        const challenge = sessionChallengesRef.current[challengeIndexRef.current];

        if (challenge === "turn_left") {
          const faceCenter = (pts[0].x + pts[16].x) / 2;
          const noseOffset = pts[30].x - faceCenter;
          const turnDelta = base.noseCenterX - noseOffset;
          const threshold = Math.max(4, base.faceWidth * 0.075);
          registerSustained(turnDelta > threshold, "turn_left");
        } else if (challenge === "turn_right") {
          const faceCenter = (pts[0].x + pts[16].x) / 2;
          const noseOffset = pts[30].x - faceCenter;
          const turnDelta = noseOffset - base.noseCenterX;
          const threshold = Math.max(4, base.faceWidth * 0.075);
          registerSustained(turnDelta > threshold, "turn_right");
        } else if (challenge === "nod") {
          const nodDelta = pts[30].y - base.noseTipY;
          const threshold = Math.max(6, base.faceWidth * 0.055);
          registerSustained(nodDelta > threshold, "nod");
        } else if (challenge === "look_up") {
          const upDelta = base.noseTipY - pts[30].y;
          const threshold = Math.max(6, base.faceWidth * 0.05);
          registerSustained(upDelta > threshold, "look_up");
        } else if (challenge === "smile") {
          registerSustained(expr.happy > 0.42, "smile");
        } else if (challenge === "surprised") {
          registerSustained(expr.surprised > 0.38, "surprised");
        } else if (challenge === "mouth_open") {
          const mouthOpen = Math.abs(pts[62].y - pts[66].y);
          const threshold = Math.max(5, base.faceWidth * 0.055);
          registerSustained(mouthOpen > threshold, "mouth_open");
        } else if (challenge === "blink") {
          const le = landmarkEAR(pts, 36);
          const re = landmarkEAR(pts, 42);
          const meanB = (base.leftEAR + base.rightEAR) / 2;
          const lowTh = meanB * 0.55;
          const hiTh = meanB * 0.88;
          if (!blinkSawLowRef.current) {
            if (le < lowTh && re < lowTh) blinkSawLowRef.current = true;
          } else if (le > hiTh && re > hiTh) {
            markChallengeDoneRef.current("blink");
            blinkSawLowRef.current = false;
          }
        } else if (challenge === "tilt_head") {
          const tilt = headTiltDeltaRad(eyeTiltAngleRad(pts), base.eyeAngle);
          const need = 0.09;
          registerSustained(tilt > need, "tilt_head");
        } else if (challenge === "angry") {
          registerSustained(expr.angry > 0.36, "angry");
        }
      } else {
        faceLostCountRef.current += 1;
        sustainCountRef.current = 0;
        blinkSawLowRef.current = false;
        if (faceLostCountRef.current > 15) {
          baselineRef.current = null;
          noseYHistoryRef.current = [];
          noseCenterXHistoryRef.current = [];
          faceWidthHistoryRef.current = [];
          calEyeAngRef.current = [];
          calLeftEarRef.current = [];
          calRightEarRef.current = [];
          faceLostCountRef.current = 0;
          console.log("🔄 Baseline reset — face lost too long");
        }
      }
    } catch (e) {
      console.warn("Liveness error:", e);
    }

    isDetectingRef.current = false;
  }, []);

  const handleFile = (f) => {
    if (!f || !f.type.startsWith("image/")) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResults([]);
    setError(null);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    handleFile(e.dataTransfer.files[0]);
  };

  const resetLiveness = () => {
    setChallengeIndex(0);
    challengeIndexRef.current = 0;
    setCompletedChallenges([]);
    completedRef.current = [];
    setLivenessLive(false);
    const first = sessionChallengesRef.current[0];
    setChallengeMsg(first ? CHALLENGE_TEXT[first] || first : "");
    baselineRef.current = null;
    noseYHistoryRef.current = [];
    noseCenterXHistoryRef.current = [];
    faceWidthHistoryRef.current = [];
    calEyeAngRef.current = [];
    calLeftEarRef.current = [];
    calRightEarRef.current = [];
    faceLostCountRef.current = 0;
    challengeCooldownRef.current = false;
    sustainCountRef.current = 0;
    blinkSawLowRef.current = false;
    postVerifyRunningRef.current = false;
  };

  const startCamera = async () => {
    if (!modelsLoaded) {
      setError("⏳ Models still loading, wait 3 seconds and try again.");
      return;
    }
    setError(null);
    setLivenessSessionLoading(true);
    try {
      const deviceId = getOrCreateDeviceId();
      const res = await fetch(`${API_URL}/liveness/session`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ device_id: deviceId }),
      });
      let data = {};
      try {
        data = await res.json();
      } catch {
        data = {};
      }
      if (!res.ok) {
        const msg =
          res.status === 409
            ? typeof data.detail === "string"
              ? data.detail
              : "All gesture sequences may be exhausted for this device."
            : typeof data.detail === "string"
              ? data.detail
              : "Could not start liveness session.";
        setError(msg);
        setLivenessSessionLoading(false);
        return;
      }
      const { session_id, gestures } = data;
      if (!session_id || !Array.isArray(gestures) || gestures.length === 0) {
        setError("Invalid server response for liveness session.");
        setLivenessSessionLoading(false);
        return;
      }
      livenessSessionIdRef.current = session_id;
      livenessCompletedRef.current = false;
      setCanMatch(false);
      sessionChallengesRef.current = gestures;
      setSessionChallenges(gestures);
      resetLiveness();

      const s = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(s);
      setShowCamera(true);
      setLivenessSessionLoading(false);

      setTimeout(() => {
        if (videoRef.current) {
          videoRef.current.srcObject = s;
          videoRef.current.onloadedmetadata = () => {
            intervalRef.current = setInterval(runLivenessLoop, 150);
            meshIntervalRef.current = setInterval(runMeshLoop, 500);
          };
        }
      }, 100);
    } catch {
      setLivenessSessionLoading(false);
      setError("Camera access denied. Please allow camera permission.");
      sessionChallengesRef.current = [];
      setSessionChallenges([]);
      livenessSessionIdRef.current = null;
      livenessCompletedRef.current = false;
      setCanMatch(false);
    }
  };

  const stopMediaOnly = () => {
    if (stream) stream.getTracks().forEach((t) => t.stop());
    clearInterval(intervalRef.current);
    clearInterval(meshIntervalRef.current);
    intervalRef.current = null;
    meshIntervalRef.current = null;
    setStream(null);
    setShowCamera(false);
    setLivenessLive(false);
    const canvas = overlayCanvasRef.current;
    if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
  };

  const stopCamera = () => {
    stopMediaOnly();
    setSessionChallenges([]);
    sessionChallengesRef.current = [];
    livenessSessionIdRef.current = null;
    livenessCompletedRef.current = false;
    setCanMatch(false);
    resetLiveness();
  };

  const takeSelfie = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      const f = new File([blob], "selfie.jpg", { type: "image/jpeg" });
      handleFile(f);
      stopMediaOnly();
    }, "image/jpeg");
  };

  const handleMatch = async () => {
    if (!file) return;
    if (!livenessSessionIdRef.current || !livenessCompletedRef.current || !canMatch) {
      setError("Complete the camera liveness check first, then capture a selfie (or use your verified flow).");
      return;
    }
    setLoading(true);
    setError(null);
    setResults([]);
    setProgress(0);
    setGeoError(null);

    const geo = await captureGeo();
    setGeoData(geo);
    setGeoAddress(null);
    if (!geo) {
      setGeoError("⚠ Location unavailable — proceeding without geo.");
    } else {
      reverseGeocode(geo.lat, geo.long).then((addr) => setGeoAddress(addr));
    }

    const interval = setInterval(() => {
      setProgress((p) => {
        if (p >= 90) { clearInterval(interval); return 90; }
        return p + Math.random() * 12;
      });
    }, 300);

    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), MATCH_REQUEST_TIMEOUT_MS);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("top_k", 10);
    formData.append("device_id", getOrCreateDeviceId());
    if (livenessSessionIdRef.current) {
      formData.append("liveness_session_id", livenessSessionIdRef.current);
    }
    if (geo) {
      formData.append("geo_lat", geo.lat);
      formData.append("geo_long", geo.long);
      formData.append("geo_timestamp", geo.timestamp);
    }

    try {
      const res = await fetch(`${API_URL}/match`, {
        method: "POST",
        body: formData,
        signal: controller.signal,
      });
      clearTimeout(timeout);
      const data = await res.json();
      if (data.error) {
        setError(data.error);
        setProgress(0);
      } else {
        setResults(data.matches);
        setProgress(100);
        livenessSessionIdRef.current = null;
        livenessCompletedRef.current = false;
        setCanMatch(false);
        setSessionChallenges([]);
        sessionChallengesRef.current = [];
      }
    } catch (err) {
      clearTimeout(timeout);
      setProgress(0);
      if (err.name === "AbortError") {
        setError("Request timed out. Try again.");
      } else {
        setError("Cannot connect to backend.");
      }
    } finally {
      clearInterval(interval);
      setLoading(false);
    }
  };

  const getColor = (score) => {
    if (score >= 0.90) return "#24aa4d";
    if (score >= 0.75) return "#ffbf01";
    return "#ff0000";
  };

  const getLabel = (score) => {
    if (score >= 0.90) return "High";
    if (score >= 0.75) return "Medium";
    return "Low";
  };

  return (
    <div className="fm-page">
      <header className="fm-header-banner">
        <img src={bargadLogo} alt="Bargad" className="fm-header-logo" />
        <div className="fm-header-profile" ref={profileMenuRef}>
          <button
            type="button"
            className="fm-profile-btn"
            onClick={() => setProfileMenuOpen((o) => !o)}
            aria-expanded={profileMenuOpen}
            aria-haspopup="true"
            aria-label="Account menu"
          >
            <span className="fm-profile-avatar-wrap" aria-hidden>
              <UserAvatarIcon />
            </span>
            <span className="fm-profile-chevron" aria-hidden>
              {profileMenuOpen ? "▲" : "▼"}
            </span>
          </button>
          {profileMenuOpen && (
            <div className="fm-profile-dropdown" role="menu">
              {userEmail ? (
                <div className="fm-profile-email" title={userEmail}>
                  {userEmail}
                </div>
              ) : null}
              <button
                type="button"
                className="fm-profile-logout"
                role="menuitem"
                onClick={() => {
                  setProfileMenuOpen(false);
                  onLogout?.();
                }}
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </header>

      <div className={`fm-container ${showCamera ? "fm-container-wide" : ""}`}>

        <div className="fm-header">
          <div className="fm-logo">⬡</div>
          <div>
            <h1>Face Match</h1>
            <p>
              Complete the camera liveness flow, then capture or choose an image — Find Matches requires a verified
              session.
            </p>
          </div>
        </div>

        <div className={showCamera ? "fm-upload-camera-row" : ""}>

          {/* Left col — dropzone + buttons */}
          <div className="fm-left-col">
            <div
              className={`fm-dropzone ${dragging ? "active" : ""} ${preview ? "has-preview" : ""}`}
              onClick={() => inputRef.current.click()}
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                hidden
                onChange={(e) => handleFile(e.target.files[0])}
              />
              {preview ? (
                <div className="fm-preview">
                  <img src={preview} alt="Preview" />
                  <div className="fm-preview-overlay">Click to change</div>
                </div>
              ) : (
                <div className="fm-drop-content">
                  <div className="fm-drop-icon">📁</div>
                  <p>Drag & drop or <span>click to upload</span></p>
                  <small>JPG, PNG, WEBP supported</small>
                </div>
              )}
            </div>

            <div className="fm-btn-row">
              <button className="fm-btn" onClick={handleMatch} disabled={!file || loading || !canMatch}>
                {loading ? <><span className="fm-spinner" /> Searching Faces...</> : "Find Matches"}
              </button>
              <button className="fm-camera-btn" onClick={showCamera ? stopCamera : startCamera}>
                {showCamera ? "✕ Cancel" : "Take Selfie Instead"}
              </button>
            </div>
          </div>

          {/* Right col — camera + liveness panel */}
          {showCamera && (
            <div className="fm-right-col">
              <div className="fm-camera-outer">
                <div className="fm-camera-wrapper">
                  <video ref={videoRef} autoPlay playsInline className="fm-camera-feed" />
                  <canvas ref={overlayCanvasRef} className="fm-mesh-overlay" />
                  <canvas ref={canvasRef} hidden />
                  <div className="fm-camera-actions">
                    {livenessLive && (
                      <button className="fm-capture-btn" onClick={takeSelfie}>
                        📸 Capture
                      </button>
                    )}
                  </div>
                  <div className="fm-scan-line" />
                  <div className="fm-scan-corners">
                    <span className="corner tl" /><span className="corner tr" />
                    <span className="corner bl" /><span className="corner br" />
                  </div>
                </div>
              </div>

              {/* Liveness panel beside camera */}
              <div className="fm-liveness-panel fm-liveness-beside">
                {!livenessLive ? (
                  <>
                    <div className="fm-liveness-title">
                      {livenessSessionLoading
                        ? "⏳ Issuing challenge…"
                        : modelsLoaded
                          ? "🔍 Liveness Check"
                          : "⏳ Loading models…"}
                    </div>
                    <div className="fm-liveness-steps">
                      {sessionChallenges.length === 0 ? (
                        <div className="fm-liveness-msg">Steps load when the session starts.</div>
                      ) : (
                        sessionChallenges.map((c, i) => (
                          <div
                            key={`${c}-${i}`}
                            className={`fm-liveness-step ${
                              completedChallenges.includes(c) ? "done" :
                              i === challengeIndex ? "active" : "pending"
                            }`}
                          >
                            {completedChallenges.includes(c) ? "✅" : i === challengeIndex ? "▶" : "○"}
                            &nbsp;{CHALLENGE_TEXT[c] || c}
                          </div>
                        ))
                      )}
                    </div>
                    {challengeMsg && <div className="fm-liveness-msg">{challengeMsg}</div>}
                  </>
                ) : (
                  <div className="fm-liveness-success">✅ Liveness Verified!<br />Click 📸 Capture</div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Geo card */}
        {geoData && (
          <div className="fm-geo-card">
            <div className="fm-geo-map">
              <MapContainer
                center={[parseFloat(geoData.lat), parseFloat(geoData.long)]}
                zoom={16}
                style={{ width: "120px", height: "110px" }}
                zoomControl={false}
                dragging={false}
                scrollWheelZoom={false}
                doubleClickZoom={false}
                attributionControl={false}
              >
                <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                <Marker position={[parseFloat(geoData.lat), parseFloat(geoData.long)]} />
              </MapContainer>
            </div>
            <div className="fm-geo-details">
              <div className="fm-geo-city">
                📍 {geoAddress ? geoAddress.short : "Fetching address..."} 🇮🇳
              </div>
              {geoAddress?.full && (
                <div className="fm-geo-full-address">{geoAddress.full}</div>
              )}
              <div className="fm-geo-coords">
                Lat {parseFloat(geoData.lat).toFixed(6)}° &nbsp; Long {parseFloat(geoData.long).toFixed(6)}°
              </div>
              <div className="fm-geo-time">
                {new Date(geoData.timestamp).toLocaleString("en-IN", {
                  weekday: "long", day: "2-digit", month: "2-digit",
                  year: "numeric", hour: "2-digit", minute: "2-digit",
                  timeZoneName: "short",
                })}
              </div>
            </div>
          </div>
        )}
        {geoError && <div className="fm-geo-error">{geoError}</div>}

        {/* Progress Bar */}
        {loading && (
          <div className="fm-progress-wrapper">
            <div className="fm-progress-bar" style={{ width: `${Math.min(progress, 100)}%` }} />
            <span>Searching Faces... {Math.min(Math.round(progress), 100)}%</span>
          </div>
        )}

        {/* Scanning Animation */}
        {loading && preview && (
          <div className="fm-scan-wrapper">
            <img src={preview} alt="Scanning" className="fm-scan-img" />
            <div className="fm-scan-line" />
            <div className="fm-scan-corners">
              <span className="corner tl" /><span className="corner tr" />
              <span className="corner bl" /><span className="corner br" />
            </div>
            <div className="fm-scan-label">Analyzing Face...</div>
          </div>
        )}

        {error && <div className="fm-error">⚠ {error}</div>}

        {/* Results */}
        {results.length > 0 && (
          <div className="fm-results">
            <h2>Top {results.length} Matches</h2>
            <div className="fm-grid">
              {results.map((match, i) => (
                <div className="fm-card" key={i} onClick={() => setSelectedImg(match)}>
                  <div className="fm-rank">#{i + 1}</div>
                  <div className="fm-card-img">
                    {match.images && match.images.length > 0 ? (
                      <>
                        <div className={`fm-multi-imgs ${hoverCardIndex === i ? "fm-img-hidden" : ""}`}>
                          {match.images.map((imgUrl, j) => (
                            <img key={j} src={imgUrl} alt={`match-${j}`} className="fm-multi-img"
                              onClick={(e) => { e.stopPropagation(); setSelectedImg({ ...match, image_url: imgUrl }); }}
                            />
                          ))}
                        </div>
                        <img src={preview} alt="Your upload"
                          className={`fm-img-uploaded ${hoverCardIndex === i ? "fm-img-visible" : ""}`}
                        />
                        <button className="fm-compare-btn"
                          onMouseEnter={() => setHoverCardIndex(i)}
                          onMouseLeave={() => setHoverCardIndex(null)}
                          onClick={(e) => e.stopPropagation()}
                        >Compare</button>
                      </>
                    ) : (
                      <div className="fm-no-img">No Image</div>
                    )}
                  </div>
                  <div className="fm-card-body">
                    <p className="fm-name">{match.label.replace(/_/g, " ")}</p>
                    <div className="fm-bar-bg">
                      <div className="fm-bar-fill"
                        style={{ width: `${Math.min(match.confidence * 100, 100)}%`, background: getColor(match.confidence) }}
                      />
                    </div>
                    <div className="fm-score-row">
                      <span className="fm-badge" style={{ background: getColor(match.confidence) }}>
                        {getLabel(match.confidence)}
                      </span>
                      <span className="fm-score fm-score-highlight" style={{ color: getColor(match.confidence) }}>
                        {(match.confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedImg && (
          <div className="fm-modal" onClick={() => setSelectedImg(null)}>
            <div className="fm-modal-box" onClick={(e) => e.stopPropagation()}>
              <img src={selectedImg.image_url} alt="Match" />
              <p style={{ color: getColor(selectedImg.confidence) }}>
                {(selectedImg.confidence * 100).toFixed(1)}% Match
              </p>
              <span className="fm-modal-label">{selectedImg.label.replace(/_/g, " ")}</span>
              <button className="fm-modal-close" onClick={() => setSelectedImg(null)}>✕ Close</button>
            </div>
          </div>
        )}

      </div>

      <div className="fm-footer-branding">
        <img src={bargadBranding} alt="Bargad" />
      </div>
    </div>
  );
}
