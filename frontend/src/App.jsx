import React, { useState } from "react";
import { Container, Button, Alert, Spinner, Row, Col, Form } from "react-bootstrap";
import Upload from "./components/Upload";
import SummaryCards from "./components/SummaryCards";
import Chatbot from "./components/Chatbot";

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

  function handleUploaded(fn, ready = true, chunks = 0) {
    setFilename(fn);
    setReadyForSummary(Boolean(ready));
    setSummaryItems([]);
    setError(null);
    setTotalChunks(Number(chunks || 0));
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
      <h1 className="mb-4">AuraLearn — Upload & Summarize</h1>

      {!filename ? (
        <Upload onUploaded={handleUploaded} />
      ) : (
        <>
          <Row className="align-items-center mb-3">
            <Col>
              <h5>File: <strong>{filename}</strong></h5>
            </Col>
            <Col className="text-end">
              <Button variant="secondary" size="sm" onClick={() => {
                setFilename(null);
                setReadyForSummary(false);
                setSummaryItems([]);
                setError(null);
              }}>
                Upload Another
              </Button>
            </Col>
          </Row>

          {!readyForSummary && (
            <Alert variant="info">File uploaded and processing. Please wait until indexing finishes.</Alert>
          )}

          {readyForSummary && !summaryItems.length && (
            <div className="mb-3 d-flex gap-2 align-items-center">
              <Form.Check
                type="checkbox"
                id="use-llm"
                label="Use Gemini LLM"
                checked={useLLM}
                onChange={(e) => setUseLLM(e.target.checked)}
                className="me-3"
              />

              <Button variant="primary" onClick={generateSummary} disabled={loading}>
                {loading ? (<><Spinner animation="border" size="sm" /> Generating...</>) : "Generate Summary"}
              </Button>
            </div>
          )}

          {error && <Alert variant="danger" className="mt-3">{error}</Alert>}

          {summaryItems.length > 0 && (
            <>
              <div className="mb-3">
                <Button variant="outline-secondary" onClick={() => setSummaryItems([])}>Clear Summary</Button>
              </div>
              <SummaryCards summaryItems={summaryItems} />
            </>
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
