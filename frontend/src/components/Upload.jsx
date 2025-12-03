import React, { useState } from "react";
import { Form, Button, Alert, ProgressBar } from "react-bootstrap";

const API_BASE = "http://localhost:8000";

export default function Upload({ onUploaded }) {
  const [file, setFile] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setFile(e.target.files?.[0] || null);
    setStatusMsg("");
  };

  const doUpload = async () => {
    if (!file) {
      setStatusMsg("Select a PDF first");
      return;
    }
    setLoading(true);
    setStatusMsg("Uploading & processing...");

    try {
      const fd = new FormData();
      fd.append("file", file);
      const res = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: fd,
      });

      if (!res.ok) {
        const text = await res.text();
        throw new Error(text || `Upload failed (${res.status})`);
      }

      const data = await res.json();
      if (data.status === "ready") {
        setStatusMsg("File processed and ready for summarization.");
        // pass total_chunks to parent so UI can decide summary size
        onUploaded(data.filename || file.name, true, data.total_chunks || 0);
      } else {
        setStatusMsg("Processing returned unexpected status.");
        onUploaded(data.filename || file.name, false, data.total_chunks || 0);
      }
    } catch (err) {
      console.error(err);
      setStatusMsg("Upload/processing failed: " + (err.message || err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <Form.Group controlId="fileUpload" className="mb-2">
        <Form.Label>Upload a PDF</Form.Label>
        <Form.Control type="file" accept="application/pdf" onChange={handleChange} />
      </Form.Group>

      <div className="d-flex gap-2">
        <Button onClick={doUpload} disabled={!file || loading} variant="primary">
          {loading ? "Uploading..." : "Upload & Index"}
        </Button>
        {statusMsg && <div style={{ alignSelf: "center", color: "#374151" }}>{statusMsg}</div>}
      </div>

      {loading && <ProgressBar animated now={100} className="mt-2" />}
    </div>
  );
}
