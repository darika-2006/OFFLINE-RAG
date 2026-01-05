import { useState } from "react";
import api from "../api";

export default function EmployeeUI() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");

  const ask = async () => {
    const res = await api.post("/ask", { question });
    setAnswer(res.data.answer);
  };

  return (
    <div className="container">
      <h2>Employee Portal</h2>

      <div className="section">
        <input
          placeholder="Ask a question"
          onChange={(e) => setQuestion(e.target.value)}
        />
        <button onClick={ask}>Ask</button>
        {answer && <div className="answer-box">{answer}</div>}
      </div>
    </div>
  );
}
