import React, { useState } from "react";
import { uploadPDF } from "../api";

export default function Upload({ onReady }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  async function doUpload() {
    if (!file) {
      setStatus("Select a PDF first");
      return;
    }
    setStatus("Uploading & indexing...");
    try {
      const data = await uploadPDF(file);
      // data: { status: "ready", filename, chunks }
      setStatus("File processed and ready for summarization.");
      if (onReady) onReady(data.filename); // notify App that file is ready
    } catch (e) {
      setStatus("Error: " + (e.message || e));
    }
  }

  return (
    <div>
      <input type="file" accept="application/pdf" onChange={(e) => setFile(e.target.files?.[0])} />
      <button onClick={doUpload} style={{ marginLeft: 8 }}>Upload</button>
      <div style={{ marginTop: 8 }}>{status}</div>
    </div>
  );
}
