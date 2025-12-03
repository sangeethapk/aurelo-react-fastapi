import React, { useState, useRef, useEffect } from "react";
import "./Chatbot.css";

const API_BASE = "http://localhost:8000";

export default function Chatbot({ filename, isOpen, onToggle }) {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hello! 👋 Ask me any questions about this document.",
      sender: "bot",
      timestamp: new Date(),
    },
  ]);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSendMessage = async () => {
    if (!inputValue.trim()) return;
    if (!filename) {
      setMessages((prev) => [
        ...prev,
        {
          id: prev.length + 1,
          text: "❌ Please upload a PDF first to ask questions.",
          sender: "bot",
          timestamp: new Date(),
        },
      ]);
      return;
    }

    // Add user message
    const userMessage = {
      id: messages.length + 1,
      text: inputValue,
      sender: "user",
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setLoading(true);

    try {
      // Call the chat endpoint on backend
      const response = await fetch(`${API_BASE}/chat?filename=${encodeURIComponent(filename)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: inputValue }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const data = await response.json();
      const botMessage = {
        id: messages.length + 2,
        text: data.answer || "Sorry, I couldn't find an answer to that question.",
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage = {
        id: messages.length + 2,
        text: `⚠️ Error: ${error.message}. Please try again.`,
        sender: "bot",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  if (!isOpen) {
    return (
      <button className="chatbot-fab" onClick={onToggle} title="Open Chatbot">
        💬
      </button>
    );
  }

  return (
    <div className="chatbot-container">
      {/* Header */}
      <div className="chatbot-header">
        <h3>📚 Document Q&A</h3>
        <button className="chatbot-close" onClick={onToggle}>
          ✕
        </button>
      </div>

      {/* Messages */}
      <div className="chatbot-messages">
        {messages.map((msg) => (
          <div key={msg.id} className={`message message-${msg.sender}`}>
            <div className="message-content">
              <p>{msg.text}</p>
              <span className="message-time">
                {msg.timestamp.toLocaleTimeString([], {
                  hour: "2-digit",
                  minute: "2-digit",
                })}
              </span>
            </div>
          </div>
        ))}

        {loading && (
          <div className="message message-bot">
            <div className="message-content">
              <div className="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chatbot-input-area">
        <textarea
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Ask a question about the document..."
          disabled={loading}
          rows="3"
          autoFocus
        />
        <button
          onClick={handleSendMessage}
          disabled={loading || !inputValue.trim() || !filename}
          className="send-btn"
        >
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}
