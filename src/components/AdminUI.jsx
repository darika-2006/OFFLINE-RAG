import { useState, useRef, useEffect } from "react";
import api from "../api";

export default function AdminUI() {
  // What the user types
  const [inputValue, setInputValue] = useState("");

  // All chat messages (user + bot)
  const [messages, setMessages] = useState([]);

  // Files to upload
  const [files, setFiles] = useState([]);

  // Show/hide upload box
  const [showUploadForm, setShowUploadForm] = useState(false);

  // Auto-scroll to bottom
  const messagesEndRef = useRef(null);

  // Scroll to latest message whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Upload files to server
  const uploadFiles = async () => {
    if (files.length === 0) {
      alert("Please select files!");
      return;
    }

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      await api.post("/admin/upload", formData);
      alert("Files uploaded!");
      setFiles([]);
      setShowUploadForm(false);
    } catch (error) {
      alert("Upload failed!");
    }
  };

  // Send message to chat
  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    // Add user message
    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: "user",
      timestamp: new Date(),
    };
    setMessages([...messages, userMessage]);
    setInputValue("");

    // Get bot response
    try {
      const response = await api.post("/ask", { question: inputValue });
      const botMessage = {
        id: Date.now() + 1,
        text: response.data.answer || "No response",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Failed to get response",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    }
  };

  // ========== RENDER ==========
  return (
    <div style={styles.container}>
      <div className="chat-container">
        {/* Header */}
        <div className="chat-header">
          <h2>Admin Chat</h2>
        </div>

        {/* Upload Form */}
        {showUploadForm && (
          <div style={styles.uploadBox}>
            <h4>Upload Documents</h4>
            <input
              type="file"
              multiple
              onChange={(e) => setFiles(Array.from(e.target.files))}
            />
            {files.length > 0 && <p>{files.length} files selected</p>}
            <button onClick={uploadFiles}>Upload</button>
            <button onClick={() => setShowUploadForm(false)}>Cancel</button>
          </div>
        )}

        {/* Messages */}
        <div className="chat-messages">
          {messages.length === 0 ? (
            <p>No messages yet. Start chatting!</p>
          ) : (
            messages.map((msg) => (
              <div key={msg.id} className={`message ${msg.sender}`}>
                <div className="message-content">{msg.text}</div>
                <div className="message-time">
                  {msg.timestamp.toLocaleTimeString()}
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div
          style={{
            display: "flex",
            gap: "10px",
            padding: "16px",
            background: "white",
            borderTop: "1px solid #e5e7eb",
            alignItems: "center",
          }}
        >
          <input
            type="text"
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && sendMessage()}
            style={{
              flex: 1,
              padding: "12px 16px",
              border: "1px solid #dbeafe",
              borderRadius: "8px",
              fontSize: "14px",
              outline: "none",
              height: "44px",
            }}
          />
          <button
            onClick={sendMessage}
            style={{
              padding: "10px 20px",
              background: "#2563eb",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "14px",
              fontWeight: "500",
              width: "80px",
              height: "44px",
              flexShrink: 0,
            }}
          >
            Send
          </button>
          <button
            onClick={() => setShowUploadForm(!showUploadForm)}
            style={{
              padding: "10px 16px",
              background: "#6b7280",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "13px",
              width: "100px",
              height: "44px",
              flexShrink: 0,
            }}
          >
            📎 Upload
          </button>
        </div>
      </div>
    </div>
  );
}

// Simple styles
const styles = {
  container: {
    height: "100vh",
    padding: "20px",
    background: "#f8fafc",
  },
  uploadBox: {
    padding: "16px",
    background: "#f3f4f6",
    borderRadius: "8px",
    marginBottom: "16px",
    border: "2px solid #2563eb",
  },
};
