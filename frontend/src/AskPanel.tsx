import { useState } from "react";
import { post, type Answer } from "./api";

const CHIPS: [string, string][] = [
  ["Report contents", "What should an upper GI endoscopy report contain?"],
  ["Consent", "What must patients be told before consent for an endoscopic procedure?"],
  ["Auf Deutsch", "Welche Qualitätsanforderungen gelten für die Dokumentation in der Endoskopie?"],
];

export default function AskPanel({ say }: { say: (text: string, bad?: boolean) => void }) {
  const [name, setName] = useState("");
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [answer, setAnswer] = useState<Answer | null>(null);

  const ask = async () => {
    setAsking(true);
    const r = await post<Answer>("/api/ask", { name, question });
    setAsking(false);
    if (r.status !== 200) {
      say(r.data.error ?? "Could not answer.", true);
      return;
    }
    say("");
    setAnswer(r.data);
  };

  return (
    <section className="card">
      <h2>Ask the guidelines</h2>
      <p className="mute">Answers come only from the EU and German guideline documents in the knowledge base, with the page they came from.</p>
      <div className="row">
        <input type="text" placeholder="Your name" autoComplete="off" value={name} onChange={(e) => setName(e.target.value)} />
      </div>
      <textarea className="edit" rows={2} maxLength={500} aria-label="Question" value={question}
                placeholder="For example: What must patients be told before consent for an endoscopy?" onChange={(e) => setQuestion(e.target.value)} />
      <div className="row">
        <button className="primary" disabled={asking} onClick={() => void ask()}>Ask</button>
        {asking && <span className="mute">Looking it up…</span>}
        {CHIPS.map(([text, q]) => <button key={text} className="chip-btn" onClick={() => setQuestion(q)}>{text}</button>)}
      </div>
      {answer && (
        <div className="answer">
          <p>{answer.answer}</p>
          {answer.citations.map((c) => (
            <div key={c.n} className="src">
              <div className="head">
                <span className="badge">{c.n}</span>
                <strong>{c.title + (c.page ? `, page ${c.page}` : "")}</strong>
                <span className="kind">Guideline reference · {c.file}</span>
              </div>
              <div className="body"><blockquote>“{c.text}”</blockquote></div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
