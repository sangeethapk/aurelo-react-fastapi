import React, { useState } from "react";
import Upload from "./components/Upload";
import { requestSummary } from "./api";

export default function App() {
  const [filename, setFilename] = useState(null);
  const [summary, setSummary] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [useLLM, setUseLLM] = useState(true);

  async function handleGenerateSummary() {
    if (!filename) return;
    setLoadingSummary(true);
    setSummary(null);
    try {
      const res = await requestSummary(filename, useLLM, 8);
      setSummary(res.summary);
    } catch (e) {
      setSummary("Error generating summary: " + (e.message || e));
    } finally {
      setLoadingSummary(false);
    }
  }

  return (
    <div style={{ padding: 20 }}>
      <h1>AuraLearn — Upload and Summarize on Demand</h1>
      {!filename ? (
        <Upload onReady={(fn) => { setFilename(fn); setSummary(null); }} />
      ) : (
        <>
          <p>Using file: <strong>{filename}</strong></p>
          <div style={{ marginBottom: 10 }}>
            <label style={{ marginRight: 10 }}>
              <input type="checkbox" checked={useLLM} onChange={(e) => setUseLLM(e.target.checked)} />
              Use LLM for summary (requires API key on server)
            </label>
            <button onClick={handleGenerateSummary} disabled={loadingSummary}>
              {loadingSummary ? "Generating..." : "Generate Summary"}
            </button>
          </div>

          {summary && (
            <div style={{ whiteSpace: "pre-wrap", border: "1px solid #ddd", padding: 12 }}>
              <h3>Summary</h3>
              <div>{summary}</div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
