export const API_BASE = "http://localhost:8000";

export async function uploadPDF(file) {
  const fd = new FormData();
  fd.append("file", file);
  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Upload failed: ${res.status}`);
  }
  return res.json();
}

export async function requestSummary(filename, use_llm = true, top_k = 8) {
  const res = await fetch(`${API_BASE}/summary/${encodeURIComponent(filename)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_llm, top_k }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `Summary failed: ${res.status}`);
  }
  return res.json();
}