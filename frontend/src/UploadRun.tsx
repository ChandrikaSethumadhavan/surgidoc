import { useRef, useState } from "react";
import { post, upload, type Ctx } from "./api";

const IDLE = ["idle", "uploaded", "approved", "rejected", "judge_rejected", "blocked", "guardrails_blocked", "no_usable_findings", "failed"];

export default function UploadRun({ status, say, refresh }: Ctx) {
  const [bar, setBar] = useState({ loaded: 0, total: 1 });
  const [operator, setOperator] = useState("");
  const pick = useRef<HTMLInputElement>(null);
  const phase = status?.phase ?? "idle";

  const onPick = async (file: File | undefined) => {
    if (!file) return;
    setBar({ loaded: 0, total: 1 });
    try {
      const r = await upload(file, (loaded, total) => setBar({ loaded, total }));
      say(r.status === 200 ? `Uploaded ${r.data.size} bytes.` : r.data.error ?? "Upload failed.", r.status !== 200);
    } catch {
      say("Upload failed.", true);
    }
    if (pick.current) pick.current.value = "";
    void refresh();
  };
  const run = async () => {
    const r = await post("/api/run", { operator });
    say(r.data.error ?? "Started.", r.status >= 400);
    void refresh();
  };
  const reset = async () => {
    const r = await post("/api/reset");
    say(r.data.error ?? "Ready for a new video.", r.status >= 400);
    void refresh();
  };

  return (
    <section className="card">
      <h2>1. Upload and run</h2>
      <div className="row">
        <input ref={pick} type="file" accept=".mp4,.mov,.avi,.mkv,video/*" aria-label="Video file" disabled={!IDLE.includes(phase)}
               onChange={(e) => void onPick(e.target.files?.[0])} />
      </div>
      <progress value={bar.loaded} max={bar.total} />
      <div className="row">
        <input type="text" placeholder="Your name (operator)" autoComplete="off" value={operator} disabled={phase !== "uploaded"}
               onChange={(e) => setOperator(e.target.value)} />
        <button className="primary" disabled={phase !== "uploaded"} onClick={() => void run()}>Run pipeline</button>
        <button disabled={phase === "running"} onClick={() => void reset()}>New video</button>
      </div>
      <p>
        State: <span className="phase">{phase.replace(/_/g, " ")}</span>{" "}
        <span className="mute">{status ? status.message + (status.violations.length ? ` Rules: ${status.violations.join(", ")}.` : "") : ""}</span>
      </p>
      <ul className="steps">
        {status?.steps.map((s) => <li key={s.id} className={s.done ? "done" : "todo"}>{(s.done ? "✓ " : "") + s.label}</li>)}
      </ul>
    </section>
  );
}
