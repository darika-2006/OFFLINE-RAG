export default function RoleSelect({ setRole }) {
  return (
    <div className="page-center">
      <div className="card">
        <h2>Select Role</h2>

        <div className="role-buttons">
          <button onClick={() => setRole("admin")}>Admin</button>
          <button onClick={() => setRole("employee")}>Employee</button>
        </div>
      </div>
    </div>
  );
}
