import { frameUrl, type Report, type Source } from "./api";

export const label = (s: Source) =>
  s.kind === "frame" ? `Video frame at ${s.time}` : s.title + (s.page ? `, page ${s.page}` : `, section ${s.section}`);

export function Thumb({ src }: { src: Source }) {
  return src.kind === "frame" && src.image ? <img src={frameUrl(src.image)} alt={`Video frame at ${src.time}`} loading="lazy" /> : null;
}

export default function Sources({ report, hit }: { report: Report | null; hit: number | null }) {
  const cited = report ? report.sources.filter((s) => s.kind === "knowledge").length : 0;
  let note = "";
  if (report && !cited) {
    note = report.guidelines_retrieved
      ? `No guideline passage supports these statements (${report.guidelines_retrieved} retrieved, none cited). They rest only on the AI's reading of the video frames.`
      : "No guideline passage was found for this video. These statements rest only on the AI's reading of the video frames.";
  }
  return (
    <section className="card">
      <h2>Sources</h2>
      {!report && <p className="mute">Evidence for each statement appears here once a draft is ready.</p>}
      {note && <p className="mute">{note}</p>}
      {report?.sources.map((s) => (
        <div key={s.n} id={`src-${s.n}`} className={s.n === hit ? "src hit" : "src"}>
          <div className="head">
            <span className="badge">{s.n}</span>
            <strong>{label(s)}</strong>
            <span className="kind">
              {s.kind === "frame" ? `Video evidence · AI description by ${s.by}` : "Guideline reference · retrieved from the knowledge base"}
            </span>
          </div>
          <div className="body">
            {s.kind === "frame" ? (
              <>
                <Thumb src={s} />
                <div>What the model said about this frame: {s.text}</div>
              </>
            ) : (
              <>
                <blockquote>“{s.text}”</blockquote>
                <div className="mute">Document: {s.file}</div>
              </>
            )}
          </div>
        </div>
      ))}
    </section>
  );
}
