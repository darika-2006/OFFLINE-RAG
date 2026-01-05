import { useState } from "react";
import api from "../api";

export default function AdminLogin({ setAdminAuth }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");

  const login = async () => {
    try {
      await api.post("/admin/login", { username, password });
      setAdminAuth(true);
    } catch {
      alert("Invalid credentials");
    }
  };

  return (
    <div className="page-center">
      <div className="card">
        <h2>Admin Login</h2>
        <input
          placeholder="Username"
          onChange={(e) => setUsername(e.target.value)}
        />
        <input
          type="password"
          placeholder="Password"
          onChange={(e) => setPassword(e.target.value)}
        />
        <button onClick={login}>Login</button>
      </div>
    </div>
  );
  
}
