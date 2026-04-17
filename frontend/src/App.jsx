import { useState, useEffect } from "react";
import FaceMatch from "./FaceMatch";
import LoginPage from "./loginpage";

const AUTH_STORAGE_KEY = "facematch_auth";

function readStoredUser() {
  try {
    const raw = sessionStorage.getItem(AUTH_STORAGE_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (data && typeof data.email === "string") return { email: data.email };
  } catch {
    /* ignore */
  }
  return null;
}

function App() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    setUser(readStoredUser());
  }, []);

  const handleLogin = (email) => {
    const next = { email };
    setUser(next);
    sessionStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
  };

  const handleLogout = () => {
    setUser(null);
    sessionStorage.removeItem(AUTH_STORAGE_KEY);
  };

  if (!user) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return <FaceMatch userEmail={user.email} onLogout={handleLogout} />;
}

export default App;
