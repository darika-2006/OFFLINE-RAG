import { useState, useRef, useEffect } from "react";
import api from "../api";

export default function EmployeeUI() {
  // What the user types
  const [inputValue, setInputValue] = useState("");

  // All chat messages (user + bot)
  const [messages, setMessages] = useState([]);

  // Auto-scroll to bottom
  const messagesEndRef = useRef(null);

  // Scroll to latest message whenever messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

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

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h2>Employee Chat</h2>
      </div>
      
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
      <div className="chat-input-area">
        <input
          type="text"
          placeholder="Type your message..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === "Enter" && sendMessage()}
          className="chat-input"
        />
        <button onClick={sendMessage} className="send-btn">
          Send
        </button>
      </div>
    </div>
  );
}
