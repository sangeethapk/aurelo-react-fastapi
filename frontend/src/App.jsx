import React, { useState } from "react";
import { Container, Button, Alert, Spinner, Row, Col, Form } from "react-bootstrap";
import Upload from "./components/Upload";
import SummaryCards from "./components/SummaryCards";
import Chatbot from "./components/Chatbot";
import McqPractice from "./components/McqPractice";
import FillInBlanks from "./components/FillInBlanks";
import ShortAnswer from "./components/ShortAnswer";

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
      <div className="app-heading mb-4">
        <h2>Aurelo: Smart Learning Atmosphere</h2>
      </div>

      {!filename ? (
        <Upload onUploaded={handleUploaded} />
      ) : (
        <>
          <Row className="align-items-center mb-4">
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

          {readyForSummary && (
            <Row className="g-2" style={{ display: "flex", flexWrap: "nowrap" }}>
              {/* Column 1: Summary */}
                <Col style={{ minWidth: "calc(20% - 6px)", flex: "1" }}>
                <div className="content-panel">
                  <h5 className="mb-3">📄 Summary</h5>
                  
                  {!summaryItems.length && (
                    <div className="mb-3 d-flex gap-2 flex-column">
                      <Form.Check
                        type="checkbox"
                        id="use-llm"
                        label="Use Gemini LLM"
                        checked={useLLM}
                        onChange={(e) => setUseLLM(e.target.checked)}
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
                        <Button variant="outline-secondary" size="sm" onClick={() => setSummaryItems([])}>Clear Summary</Button>
                      </div>
                      <SummaryCards summaryItems={summaryItems} />
                    </>
                  )}
                </div>
              </Col>

              {/* Column 2: MCQ */}
                <Col style={{ minWidth: "calc(20% - 6px)", flex: "1" }}>
                <div className="content-panel">
                  <h5 className="mb-3">✏️ MCQ Practice</h5>
                  <McqPractice filename={filename} />
                </div>
              </Col>

              {/* Column 3: Fill in the Blanks */}
                <Col style={{ minWidth: "calc(20% - 6px)", flex: "1" }}>
                <div className="content-panel">
                  <h5 className="mb-3">🎯 Fill in Blanks</h5>
                  <FillInBlanks filename={filename} />
                </div>
              </Col>

              {/* Column 4: Short Answer */}
                <Col style={{ minWidth: "calc(40% - 6px)", flex: "2" }}>
                <div className="content-panel">
                  <h5 className="mb-3">💭 Short Answer</h5>
                  <ShortAnswer filename={filename} />
                </div>
              </Col>
            </Row>
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
