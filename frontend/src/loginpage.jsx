import React, { useState } from "react";
import "./loginpage.css";

/** Allowed logins (client-side only — credentials ship in the bundle). */
const ALLOWED_USERS = [
  { email: "support@bargad.ai", password: "Bargad@2025" },
  {
    email: "srenivas.venkiteswaran@poonawallafincorp.com",
    password: "Srenivas.v@2026",
  },
];

function normalizeEmail(s) {
  return String(s || "").trim().toLowerCase();
}

function validateLogin(emailRaw, passwordRaw) {
  const email = normalizeEmail(emailRaw);
  const password = String(passwordRaw || "");
  return ALLOWED_USERS.some(
    (u) => normalizeEmail(u.email) === email && u.password === password
  );
}

function ShieldSVG({ large }) {
  const size = large ? 64 : 32;
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="#fff"
      width={size}
      height={size}
      style={{ display: "block" }}
    >
      <path
        fillRule="evenodd"
        d="M12.516 2.17a.75.75 0 0 0-1.032 0 11.209 11.209 0 0 1-7.877 3.08.75.75 0 0 0-.722.515A12.74 12.74 0 0 0 2.25 9.75c0 5.942 4.064 10.933 9.563 12.348a.749.749 0 0 0 .374 0c5.499-1.415 9.563-6.406 9.563-12.348 0-1.39-.223-2.73-.635-3.985a.75.75 0 0 0-.722-.516l-.143.001c-2.996 0-5.717-1.17-7.734-3.08Zm3.094 8.016a.75.75 0 1 0-1.22-.872l-3.236 4.53L9.53 12.22a.75.75 0 0 0-1.06 1.06l2.25 2.25a.75.75 0 0 0 1.14-.094l3.75-5.25Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

function EyeIcon({ open }) {
  if (open) {
    return (
      <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.98 8.223A10.477 10.477 0 0 0 1.934 12C3.226 16.338 7.244 19.5 12 19.5c4.756 0 8.773-3.162 10.065-7.777a10.44 10.44 0 0 0-2.044-3.777M6.228 6.228A10.45 10.45 0 0 1 12 4.5c4.756 0 8.773 3.162 10.065 7.777a10.46 10.46 0 0 1-1.67 3.05M6.228 6.228 3 3m3.228 3.228 3.65 3.65m7.894 7.894L21 21m-3.228-3.228-3.65-3.65m0 0a3 3 0 1 0-4.244-4.244m4.242 4.242L9.88 9.88" />
      </svg>
    );
  }
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="1.5">
      <path strokeLinecap="round" strokeLinejoin="round" d="M2.036 12.322a1.012 1.012 0 0 1 0-.639C3.423 7.51 7.36 4.5 12 4.5c4.638 0 8.573 3.007 9.963 7.178.07.207.07.431 0 .639C20.577 16.49 16.64 19.5 12 19.5c-4.638 0-8.573-3.007-9.963-7.178Z" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
    </svg>
  );
}

function FingerprintIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" width="28" height="28">
      <path
        fillRule="evenodd"
        d="M12 3.75a6.715 6.715 0 0 0-3.722 1.118.75.75 0 1 1-.828-1.25 8.25 8.25 0 0 1 12.8 6.883c0 3.014-.574 5.897-1.62 8.543a.75.75 0 0 1-1.395-.551A21.69 21.69 0 0 0 18.75 10.5 6.75 6.75 0 0 0 12 3.75ZM6.157 5.739a.75.75 0 0 1 .21 1.04A6.715 6.715 0 0 0 5.25 10.5c0 1.613-.463 3.12-1.265 4.393a.75.75 0 0 1-1.27-.8A6.715 6.715 0 0 0 3.75 10.5c0-1.68.503-3.246 1.367-4.55a.75.75 0 0 1 1.04-.211ZM12 7.5a3 3 0 0 0-3 3c0 3.1-1.176 5.927-3.105 8.056a.75.75 0 1 1-1.112-1.008A10.459 10.459 0 0 0 7.5 10.5a4.5 4.5 0 1 1 9 0c0 .547-.022 1.09-.067 1.626a.75.75 0 0 1-1.495-.123c.041-.495.062-.996.062-1.503a3 3 0 0 0-3-3Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

/**
 * @param {{ onLogin: (email: string) => void }} props
 */
export default function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    setError("");
    if (!email.trim() || !password) {
      setError("Enter email and password.");
      return;
    }
    if (!validateLogin(email, password)) {
      setError("Invalid email or password.");
      return;
    }
    const canonical = ALLOWED_USERS.find(
      (u) => normalizeEmail(u.email) === normalizeEmail(email)
    );
    onLogin(canonical ? canonical.email : email.trim());
  };

  return (
    <div className="lp-root">
      <div className="lp-frame">
        <div className="lp-corner lp-corner-tl" aria-hidden />
        <div className="lp-corner lp-corner-tr" aria-hidden />
        <div className="lp-corner lp-corner-bl" aria-hidden />
        <div className="lp-corner lp-corner-br" aria-hidden />

        <div className="lp-shields" aria-hidden>
          <ShieldSVG large={false} />
          <ShieldSVG large />
          <ShieldSVG large={false} />
        </div>

        <div className="lp-card">
          <h2 className="lp-title">Login</h2>

          <form onSubmit={handleSubmit}>
            <div className="lp-field">
              <label className="lp-label" htmlFor="lp-email">
                Email
              </label>
              <input
                id="lp-email"
                type="email"
                autoComplete="username"
                className="lp-input"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div className="lp-field">
              <label className="lp-label" htmlFor="lp-password">
                Password
              </label>
              <div className="lp-password-wrap">
                <input
                  id="lp-password"
                  type={showPassword ? "text" : "password"}
                  autoComplete="current-password"
                  className="lp-input"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  style={{ paddingRight: 48 }}
                />
                <button
                  type="button"
                  className="lp-toggle-eye"
                  aria-label={showPassword ? "Hide password" : "Show password"}
                  onClick={() => setShowPassword((v) => !v)}
                >
                  <EyeIcon open={showPassword} />
                </button>
              </div>
            </div>

            <div className="lp-row-options">
              <label className="lp-remember">
                <input type="checkbox" disabled style={{ accentColor: "#24aa4d" }} />
                Remember me
              </label>
              <span style={{ opacity: 0.6 }}>Forgot password?</span>
            </div>

            {error ? <div className="lp-error">{error}</div> : null}

            <button type="submit" className="lp-btn-login">
              Login
            </button>
          </form>

          <p className="lp-help">
            Please contact administrator, in case you are unable to login
          </p>
        </div>

        <div className="lp-fingerprint" aria-hidden>
          <FingerprintIcon />
        </div>
      </div>
    </div>
  );
}
