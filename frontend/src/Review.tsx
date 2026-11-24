import { useState } from "react";
import { post, type Ctx } from "./api";
import { label, Thumb } from "./Sources";

interface Edit { text?: string; removed?: boolean }

export default function Review({ status, say, refresh, onJump }: Ctx & { onJump: (n: number) => void }) {
  const report = status?.report ?? null;
  const phase = status?.phase ?? "idle";
  const editing = phase === "pending_review";
  const [held, setHeld] = useState<{ key: string; edits: Record<number, Edit> }>({ key: "", edits: {} });
  const [reviewer, setReviewer] = useState("");
  const [reason, setReason] = useState("");

  // Unsaved edits belong to one revision of one draft, so a saved edit or a regenerated draft starts clean.
  // Derived from the key rather than reset in an effect: an effect can run after the first keystroke and wipe it.
  const key = `${report?.id}:${report?.revision}`;
  const edits = held.key === key ? held.edits : {};
  const patch = (i: number, e: Edit) =>
    setHeld((prev) => {
      const cur = prev.key === key ? prev.edits : {};
      return { key, edits: { ...cur, [i]: { ...cur[i], ...e } } };
    });
  const unsaved = !!report && report.statements.some((st, i) => {
    const e = edits[i];
    return !!e && (!!e.removed || (e.text !== undefined && e.text !== st.text));
  });
  const done = (r: { status: number; data: { error?: string } }, ok: string) => {
    say(r.data.error ?? ok, r.status >= 400);
    void refresh();
  };

  const save = async () => {
    if (!report) return;
    const statements = report.statements.flatMap((st, i) => (edits[i]?.removed ? [] : [{ i, text: edits[i]?.text ?? st.text }]));
    done(await post("/api/edit", { name: reviewer, revision: report.revision, statements }), "Edits saved. Re-read the report before approving.");
  };
  const approve = async () => {
    if (unsaved) {
      say("You have unsaved edits. Save them first, or reload to discard them.", true);
      return;
    }
    done(await post("/api/approve", { name: reviewer }), "Approved.");
  };
  const reject = async () => {
    const r = await post<{ regenerating?: boolean }>("/api/reject", { name: reviewer, reason });
    done(r, r.data.regenerating ? "Rejected. Writing a new draft from your feedback…" : "Rejected. No regenerations left for this video.");
  };

  const byN = new Map((report?.sources ?? []).map((s) => [s.n, s]));
  const closing = status?.approver ? (phase === "approved" ? `Approved by ${status.approver}.` : phase === "rejected" ? `Rejected by ${status.approver}.` : "") : "";

  return (
    <section className="card">
      <h2>2. Review the draft</h2>
      {report?.statements.map((st, i) => {
        const removed = !!edits[i]?.removed;
        const text = edits[i]?.text ?? st.text;
        return (
          <div key={i} className={removed ? "stmt removed" : "stmt"}>
            {editing ? (
              <textarea className="edit" rows={Math.max(2, Math.ceil(st.text.length / 90))} aria-label={`Statement ${i + 1}`} value={text}
                        onChange={(e) => patch(i, { text: e.target.value })} />
            ) : (
              <p>{st.text}</p>
            )}
            <div className="tools">
              {st.guideline && <span className="kind">Guideline reminder</span>}
              {st.cites.map((n) => <button key={n} type="button" className="badge" title={`Show source ${n}`} onClick={() => onJump(n)}>{n}</button>)}
              {editing && report.statements.length > 1 && (
                <button type="button" className="danger" onClick={() => patch(i, { removed: !removed })}>{removed ? "Restore" : "Remove"}</button>
              )}
            </div>
            <div className="ev">
              {st.cites.map((n) => {
                const src = byN.get(n);
                return src && (
                  <button key={n} type="button" className="chip" onClick={() => onJump(n)}>
                    <Thumb src={src} />
                    <span>{n} · {label(src)}</span>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="row">
        <input type="text" placeholder="Reviewer name" autoComplete="off" value={reviewer} disabled={!editing} onChange={(e) => setReviewer(e.target.value)} />
        <button disabled={!editing} onClick={() => void save()}>Save edits</button>
        <button className="primary" disabled={!editing} onClick={() => void approve()}>Approve</button>
        <input type="text" placeholder="Reason (to reject)" autoComplete="off" value={reason} disabled={!editing} onChange={(e) => setReason(e.target.value)} />
        <button className="danger" disabled={!editing} onClick={() => void reject()}>Reject</button>
      </div>
      <p>{closing}</p>
    </section>
  );
}
