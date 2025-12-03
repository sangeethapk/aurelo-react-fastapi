import React from "react";

/**
 * Horizontal scrollable cards layout with arrow indicators.
 * Uses CSS flexbox + overflow-x for a carousel feel with navigation arrows.
 */
export default function SummaryCards({ summaryItems = [] }) {
  if (!summaryItems || summaryItems.length === 0) return null;

  return (
    <div style={{ position: "relative", paddingBottom: 8 }}>
      {/* Arrow indicator for scrolling */}
      {summaryItems.length > 1 && (
        <div style={{
          position: "absolute",
          top: "50%",
          right: -30,
          transform: "translateY(-50%)",
          fontSize: 32,
          color: "#7c3aed",
          fontWeight: "bold",
          opacity: 0.6,
          pointerEvents: "none"
        }}>
          →
        </div>
      )}

      <div style={{ overflowX: "auto", paddingBottom: 8 }}>
        <div style={{
          display: "flex",
          gap: 16,
          alignItems: "stretch",
          padding: "8px 4px",
          minWidth: "min-content"
        }}>
          {summaryItems.map((item, i) => (
            <div key={i} style={{
              minWidth: 360,
              maxWidth: 420,
              flex: "0 0 360px",
              background: "white",
              borderRadius: 12,
              padding: 20,
              boxShadow: "0 6px 18px rgba(15,23,42,0.08)",
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between"
            }}>
              <div style={{
                background: "linear-gradient(90deg,#06b6d4,#7c3aed)",
                color: "white",
                borderRadius: 8,
                padding: "8px 12px",
                fontWeight: 700,
                fontSize: 13,
                marginBottom: 12,
                display: "inline-block",
                width: "fit-content"
              }}>
                {i + 1} of {summaryItems.length}
              </div>
              <div style={{ whiteSpace: "pre-wrap", color: "#0f172a", lineHeight: 1.55, fontSize: 14 }}>
                {item}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
