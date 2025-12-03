import React, { useState } from "react";
import { Container, Button, Alert, Spinner, Row, Col, Form } from "react-bootstrap";
import Upload from "./components/Upload";
import SummaryCards from "./components/SummaryCards";
import Chatbot from "./components/Chatbot";
import McqPractice from "./components/McqPractice";
import FillInBlanks from "./components/FillInBlanks";
import ShortAnswer from "./components/ShortAnswer";
import LearningTabs from "./components/LearningTabs";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [filename, setFilename] = useState(null);
  const [readyForSummary, setReadyForSummary] = useState(false);
  const [summaryItems, setSummaryItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [useLLM, setUseLLM] = useState(true);
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [totalChunks, setTotalChunks] = useState(0);
  const [generatingAll, setGeneratingAll] = useState(false); // Track bulk generation

  function handleUploaded(fn, ready = true, chunks = 0) {
    // Clear ALL previous file caches to ensure clean state
    Object.keys(localStorage).forEach(key => {
      if (key.startsWith('aurelo_') && !key.endsWith(`${fn}_cached`)) {
        localStorage.removeItem(key);
      }
    });
    
    setFilename(fn);
    setReadyForSummary(Boolean(ready));
    setSummaryItems([]);
    setError(null);
    setTotalChunks(Number(chunks || 0));
    
    // If file is ready, generate all content in background
    if (ready) {
      generateAllContent(fn);
    }
  }

  // Generate all content (summary, MCQ, FIB, short answer) and cache in localStorage
  async function generateAllContent(fn) {
    try {
      setGeneratingAll(true);
      const url = `${API_BASE}/generate-all?filename=${encodeURIComponent(fn)}`;
      const res = await fetch(url, { method: "POST" });
      
      if (!res.ok) {
        console.warn("Bulk generation failed, will generate on-demand");
        return;
      }

      const data = await res.json();
      
      // Cache results in localStorage
      localStorage.setItem(`aurelo_${fn}_cached`, JSON.stringify({
        summary: data.summary || [],
        mcq: data.mcq || [],
        fill_blanks: data.fill_blanks || [],
        short_answer: data.short_answer || [],
        timestamp: new Date().toISOString(),
      }));
      
      console.log(`✓ Cached all content for ${fn}`);
    } catch (err) {
      console.warn("Failed to generate all content:", err);
    } finally {
      setGeneratingAll(false);
    }
  }

  async function generateSummary() {
    if (!filename) return;
    setLoading(true);
    setError(null);
    setSummaryItems([]);

    try {
      // Decide how many points to request based on file size (total chunks)
      // We'll show between 3 and 12 points. More chunks -> more points.
      const suggested = Math.min(12, Math.max(3, Math.ceil(totalChunks / 5)));
      const payload = { use_llm: useLLM };
      const url = `${API_BASE}/summary?filename=${encodeURIComponent(filename)}&num_points=${suggested}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || `Server returned ${res.status}`);
      }

      const data = await res.json();

      if (!data || !data.summary) throw new Error("No summary returned");

      // Expect data.summary to be an array (backend should return JSON array)
      let items = [];
      if (Array.isArray(data.summary)) {
        items = data.summary.map((s) => String(s).trim()).filter(Boolean);
      } else {
        // fallback if backend returned plain text
        const text = String(data.summary).trim();
        items = text.split(/\n+/).map(s => s.trim()).filter(Boolean);
      }

      // Limit to 12 cards for sanity
      if (items.length > 12) items = items.slice(0, 12);
      setSummaryItems(items);
    } catch (err) {
      console.error(err);
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <Container>
      <div className="app-heading mb-4">
        <h2>Aurelo: Smart Learning Atmosphere</h2>
      </div>

      {!filename ? (
        <Upload onUploaded={handleUploaded} />
      ) : (
        <>
          <div className="mb-4 text-end">
            <Button variant="secondary" size="sm" onClick={() => {
              setFilename(null);
              setReadyForSummary(false);
              setSummaryItems([]);
              setError(null);
            }}>
              Upload Another
            </Button>
          </div>

          {!readyForSummary && (
            <Alert variant="info">File uploaded and processing. Please wait until indexing finishes.</Alert>
          )}

          {readyForSummary && (
            <LearningTabs
              filename={filename}
              useLLM={useLLM}
              setUseLLM={setUseLLM}
              summaryItems={summaryItems}
              setSummaryItems={setSummaryItems}
              generateSummary={generateSummary}
              loading={loading}
              error={error}
            />
          )}
        </>
      )}

      {/* Chatbot Component */}
      <Chatbot 
        filename={filename} 
        isOpen={chatbotOpen} 
        onToggle={() => setChatbotOpen(!chatbotOpen)} 
      />
    </Container>
  );
}
