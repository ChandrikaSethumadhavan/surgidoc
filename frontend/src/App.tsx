import { useCallback, useEffect, useRef, useState } from "react";
import { get, pdfUrl, startSession, type Status } from "./api";
import AskPanel from "./AskPanel";
import Review from "./Review";
import Sources from "./Sources";
import UploadRun from "./UploadRun";

export default function App({ pollMs = 800 }: { pollMs?: number }) {
  const [ready, setReady] = useState(false);
  const [status, setStatus] = useState<Status | null>(null);
  const [msg, setMsg] = useState({ text: "", bad: false });
  const [toast, setToast] = useState("");
  const [hit, setHit] = useState<number | null>(null);
  const last = useRef("");
  const seenNotice = useRef(0);
  const say = useCallback((text: string, bad = false) => setMsg({ text, bad }), []);

  // Re-render only when the status changed, so a surgeon's unsaved typing is never wiped by a poll.
  const refresh = useCallback(async () => {
    const r = await get<Status>("/api/status");
    if (r.status !== 200) return;
    const key = JSON.stringify(r.data);
    if (key === last.current) return;
    last.current = key;
    setStatus(r.data);
    const n = r.data.notice;
    if (n && n.id !== seenNotice.current) {
      if (seenNotice.current !== 0 || n.id > 0) setToast(n.text);
      seenNotice.current = n.id;
    }
  }, []);

  useEffect(() => {
    startSession().then(() => setReady(true), () => say("Could not start a session. Reload the page.", true));
  }, [say]);
  useEffect(() => {
    if (!ready) return;
    void refresh();
    const t = setInterval(() => void refresh(), pollMs);
    return () => clearInterval(t);
  }, [ready, refresh, pollMs]);
  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(""), 8000);
    return () => clearTimeout(t);
  }, [toast]);

  const jump = (n: number) => {
    setHit(n);
    document.getElementById(`src-${n}`)?.scrollIntoView?.({ behavior: "smooth", block: "center" });
  };
  const ctx = { status, say, refresh };

  return (
    <>
      <header>
        <div>
          <h1>SurgiDoc</h1>
          <p>Endoscopy documentation assistant: every statement points to the frame or guideline it came from.</p>
        </div>
      </header>
      <main>
        <AskPanel say={say} />
        <UploadRun {...ctx} />
        <Review {...ctx} onJump={jump} />
        <Sources report={status?.report ?? null} hit={hit} />
        <section className="card">
          <h2>3. Result</h2>
          <div className="row">
            {status?.pdf_ready && <a className="btn primary" href={pdfUrl()}>Download approved PDF</a>}
            <a className="btn" href="/dashboard" target="_blank" rel="noopener">Metrics dashboard</a>
          </div>
          <p className={msg.bad ? "bad" : "ok"}>{msg.text}</p>
        </section>
        {toast && <div className="toast" role="status" aria-live="polite">{toast}</div>}
      </main>
    </>
  );
}
