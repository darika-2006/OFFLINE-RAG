import { useState } from "react";
import RoleSelect from "./components/RoleSelect";
import AdminLogin from "./components/AdminLogin";
import AdminUI from "./components/AdminUI";
import EmployeeUI from "./components/EmployeeUI";
import "./App.css";

function App() {
  const [role, setRole] = useState(null);
  const [adminAuth, setAdminAuth] = useState(false);

  if (!role) return <RoleSelect setRole={setRole} />;

  if (role === "admin" && !adminAuth)
    return <AdminLogin setAdminAuth={setAdminAuth} />;

  if (role === "admin") return <AdminUI />;

  return <EmployeeUI />;
}

export default App;
