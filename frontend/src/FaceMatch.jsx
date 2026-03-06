import React, { useState, useRef } from "react";
import "./FaceMatch.css";
import mainBg from "./Main background.svg?url";
import masterDemoHeader from "./master-demo-header.svg?url";
import headerMasterDemo from "./header-master-demo.svg?url";
import bargadBranding from "./bargad-branding (1).svg?url";

const API_URL = "http://localhost:8001";

export default function FaceMatch() {
  const [preview, setPreview] = useState(null);
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const [hoverCardIndex, setHoverCardIndex] = useState(null); // Feature: Compare hover
  const [selectedImg, setSelectedImg] = useState(null);       // Feature B: Modal
  const [progress, setProgress] = useState(0);  
  const videoRef = useRef();
const canvasRef = useRef();
const [showCamera, setShowCamera] = useState(false);
const [stream, setStream] = useState(null);
              // Feature C: Progress bar

  const inputRef = useRef();

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

  const startCamera = async () => {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    setStream(s);
    setShowCamera(true);
    setTimeout(() => {
      if (videoRef.current) videoRef.current.srcObject = s;
    }, 100);
  } catch {
    setError("Camera access denied. Please allow camera permission.");
  }
};

const stopCamera = () => {
  if (stream) stream.getTracks().forEach(t => t.stop());
  setStream(null);
  setShowCamera(false);
};

const takeSelfie = () => {
  const canvas = canvasRef.current;
  const video = videoRef.current;
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  canvas.toBlob((blob) => {
    const f = new File([blob], "selfie.jpg", { type: "image/jpeg" });
    handleFile(f);
    stopCamera();
  }, "image/jpeg");
};


  // Feature C: progress bar simulation during fetch
  const handleMatch = async () => {
  if (!file) return;
  setLoading(true);
  setError(null);
  setResults([]);
  setProgress(0);

  const interval = setInterval(() => {
    setProgress((p) => {
      if (p >= 90) { clearInterval(interval); return 90; }
      return p + Math.random() * 12;
    });
  }, 300);

  // Timeout after 30 seconds
  const controller = new AbortController();
  const timeout = setTimeout(() => {
    controller.abort();
  }, 30000);

  const formData = new FormData();
  formData.append("file", file);
  formData.append("top_k", 10);

  try {
    const res = await fetch(`${API_URL}/match`, {
      method: "POST",
      body: formData,
      signal: controller.signal,  // ← abort after 30s
    });

    clearTimeout(timeout);
    const data = await res.json();

    if (data.error) {
      setError(data.error);
      setProgress(0);           // ← reset progress on error
    } else {
      setResults(data.matches);
      setProgress(100);         // ← complete progress
    }

  } catch (err) {
    clearTimeout(timeout);
    setProgress(0);

    if (err.name === "AbortError") {
      setError("Request timed out. Backend is taking too long. Try again.");
    } else {
      setError("Cannot connect to backend. Make sure python api.py is running.");
    }

  } finally {
    clearInterval(interval);
    setLoading(false);          // ← THIS was missing — always stop loading
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
    <div
      className="fm-page"
      style={{
        backgroundImage: `url(${mainBg})`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
        backgroundAttachment: "fixed",
      }}
    >
      {/* Header: master-demo-header as background, header-master-demo on top */}
      <header className="fm-header-banner">
        <img src={masterDemoHeader} alt="" className="fm-header-bg" />
        <img src={headerMasterDemo} alt="Header" className="fm-header-logo" />
      </header>

      <div className="fm-container">

        {/* Header */}
        <div className="fm-header">
          <div className="fm-logo">⬡</div>
          <div>
            <h1>Face Match</h1>
            <p>Upload a face image — get top 10 matches from the dataset</p>
          </div>
        </div>

        {/* Upload Box */}
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

        {/* Match + Take Selfie buttons — horizontal */}
        <div className="fm-btn-row">
          <button
            className="fm-btn"
            onClick={handleMatch}
            disabled={!file || loading}
          >
            {loading ? <><span className="fm-spinner" /> Searching Faces...</> : "Find Matches"}
          </button>
          {!showCamera && (
            <button className="fm-camera-btn" onClick={startCamera}>
              Take Selfie Instead
            </button>
          )}
        </div>

        {/* Feature C: Progress Bar */}
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
              <span className="corner tl" />
              <span className="corner tr" />
              <span className="corner bl" />
              <span className="corner br" />
            </div>
            <div className="fm-scan-label">Analyzing Face...</div>
          </div>
        )}

{/* Camera View */}
{showCamera && (
  <div className="fm-camera-wrapper">
    <video
      ref={videoRef}
      autoPlay
      playsInline
      className="fm-camera-feed"
    />
    <canvas ref={canvasRef} hidden />
    <div className="fm-camera-actions">
      <button className="fm-capture-btn" onClick={takeSelfie}>
        📸 Capture
      </button>
      <button className="fm-cancel-btn" onClick={stopCamera}>
        ✕ Cancel
      </button>
    </div>
    {/* Scanning overlay on camera */}
    <div className="fm-scan-line" />
    <div className="fm-scan-corners">
      <span className="corner tl" />
      <span className="corner tr" />
      <span className="corner bl" />
      <span className="corner br" />
    </div>
  </div>
)}


        {/* Error */}
        {error && <div className="fm-error">⚠ {error}</div>}

        {/* Results — no slider */}
        {results.length > 0 && (
          <div className="fm-results">
            <h2>Top {results.length} Matches</h2>
            <div className="fm-grid">
              {results
                .map((match, i) => (
                  <div
                    className="fm-card"
                    key={i}
                    onClick={() => setSelectedImg(match)}  // Feature B: open modal
                  >
                    <div className="fm-rank">#{i + 1}</div>
                    <div className="fm-card-img">
  {match.images && match.images.length > 0 ? (
    <>
      {/* Multi photos grid — hidden on compare hover */}
      <div className={`fm-multi-imgs ${hoverCardIndex === i ? "fm-img-hidden" : ""}`}>
        {match.images.map((imgUrl, j) => (
          <img
            key={j}
            src={imgUrl}
            alt={`match-${j}`}
            className="fm-multi-img"
            onClick={(e) => {
              e.stopPropagation();
              setSelectedImg({ ...match, image_url: imgUrl });
            }}
          />
        ))}
      </div>

      {/* Uploaded image shown on compare hover */}
      <img
        src={preview}
        alt="Your upload"
        className={`fm-img-uploaded ${hoverCardIndex === i ? "fm-img-visible" : ""}`}
      />

      {/* Compare button */}
      <button
        className="fm-compare-btn"
        onMouseEnter={() => setHoverCardIndex(i)}
        onMouseLeave={() => setHoverCardIndex(null)}
        onClick={(e) => e.stopPropagation()}
      >
         Compare
      </button>
    </>
  ) : (
    <div className="fm-no-img">No Image</div>
  )}
</div>

                    <div className="fm-card-body">
                      <p className="fm-name">{match.label.replace(/_/g, " ")}</p>
                      <div className="fm-bar-bg">
                        <div
                          className="fm-bar-fill"
                          style={{
                            width: `${Math.min(match.confidence * 100, 100)}%`,
                            background: getColor(match.confidence),
                          }}
                        />
                      </div>
                      <div className="fm-score-row">
                        <span
                          className="fm-badge"
                          style={{ background: getColor(match.confidence) }}
                        >
                          {getLabel(match.confidence)}
                        </span>
                        <span
                          className="fm-score fm-score-highlight"
                          style={{ color: getColor(match.confidence) }}
                        >
                          {(match.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Feature B: Enlarge Modal */}
        {selectedImg && (
          <div className="fm-modal" onClick={() => setSelectedImg(null)}>
            <div className="fm-modal-box" onClick={(e) => e.stopPropagation()}>
              <img
                src={selectedImg.image_url}
                alt="Match"
              />
              <p style={{ color: getColor(selectedImg.confidence) }}>
                {(selectedImg.confidence * 100).toFixed(1)}% Match
              </p>
              <span className="fm-modal-label">{selectedImg.label.replace(/_/g, " ")}</span>
              <button className="fm-modal-close" onClick={() => setSelectedImg(null)}>
                ✕ Close
              </button>
            </div>
          </div>
        )}

      </div>

      {/* Bargad branding: bottom right, fixed */}
      <div className="fm-footer-branding">
        <img src={bargadBranding} alt="Bargad" />
      </div>
    </div>
  );
}
