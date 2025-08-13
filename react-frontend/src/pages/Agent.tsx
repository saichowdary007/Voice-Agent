import React from "react";
import "@deepgram/browser-agent";

declare global {
  namespace JSX {
    interface IntrinsicElements {
      "deepgram-agent": any;
    }
  }
}

type TokenResponse = { access_token: string; expires_in: number };
const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

export default function AgentPage() {
  const agentRef = React.useRef<any>(null);
  const [log, setLog] = React.useState<string[]>([]);
  const [connecting, setConnecting] = React.useState(false);

  const append = (s: string) => setLog((prev) => [...prev, s].slice(-200));

  const start = async () => {
    setConnecting(true);
    try {
      const t = await fetch(`${API_BASE}/api/dg/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ttl_seconds: 30 }),
      });
      const token: TokenResponse = await t.json();

      const s = await fetch(`${API_BASE}/api/dg/settings`);
      const settings = await s.json();

      const el = agentRef.current as HTMLElement & {
        token?: string;
        setAttribute: (k: string, v: string) => void;
      };
      if (!el) return;

      el.addEventListener("structured message", (e: any) => {
        try {
          const data = JSON.parse(e.detail);
          if (data?.type === "ConversationText") {
            append(`${data.role}: ${data.text}`);
          }
        } catch {}
      });

      el.setAttribute("url", "wss://agent.deepgram.com/v1/agent/converse");
      el.setAttribute("auth-scheme", "bearer");
      (el as any).token = token.access_token;
      el.setAttribute("config", JSON.stringify(settings));
    } catch (err: any) {
      append(`error: ${err?.message || String(err)}`);
    } finally {
      setConnecting(false);
    }
  };

  const stop = () => {
    const el = agentRef.current as HTMLElement;
    if (!el) return;
    el.removeAttribute("config");
    append("stopped");
  };

  return (
    <div style={{ padding: 24, display: "grid", gap: 16, gridTemplateColumns: "360px 1fr" }}>
      <div>
        <h2>Deepgram Agent</h2>
        <p>Click Start, grant mic permission, then talk.</p>
        <div style={{ display: "grid", gap: 8 }}>
          <button onClick={start} disabled={connecting} style={{ padding: 10 }}>
            {connecting ? "Starting..." : "Start"}
          </button>
          <button onClick={stop} style={{ padding: 10 }}>Stop</button>
        </div>
        <deepgram-agent id="dg-agent" ref={agentRef as any} height="300" width="300" idle-timeout-ms="15000" output-sample-rate="24000"></deepgram-agent>
      </div>
      <div>
        <h3>Transcript</h3>
        <pre style={{ background: "#111", color: "#0f0", padding: 12, minHeight: 300, borderRadius: 8, overflow: "auto" }}>
{log.join("\n")}
        </pre>
      </div>
    </div>
  );
}


