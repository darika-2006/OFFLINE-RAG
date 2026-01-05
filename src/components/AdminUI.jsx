import { useState } from "react";
import api from "../api";

export default function AdminUI() {
  const [files, setFiles] = useState([]);
  const [source, setSource] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const uploadFiles = async () => {
    const formData = new FormData();
    for (let f of files) formData.append("files", f);
    formData.append("source", source);

    await api.post("/admin/upload", formData);
    alert("Files uploaded");
  };

  const ask = async () => {
    const res = await api.post("/ask", { question });
    setAnswer(res.data.answer);
  };

  return (
    <div className="container">
      <h2>Admin Dashboard</h2>

      <div className="section">
        <h3>Upload Documents</h3>
        <input
          type="file"
          multiple
          onChange={(e) => setFiles(e.target.files)}
        />
        <input
          placeholder="Source path"
          onChange={(e) => setSource(e.target.value)}
        />
        <button onClick={uploadFiles}>Upload</button>
      </div>

      <div className="section">
        <h3>Ask Question</h3>
        <input
          placeholder="Enter question"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={ask}>Ask</button>
        {answer && <div className="answer-box">{answer}</div>}
      </div>
    </div>
  );
}
