// Types mirror what surgidoc/webui.py returns. The CSRF token comes from GET /api/session and travels in a header
// (or ?t= for image and download URLs, which the browser fetches without our headers).
export interface Step { id: string; label: string; done: boolean }
export interface Statement { text: string; cites: number[]; guideline: boolean }
export interface FrameSource { n: number; kind: "frame"; time: string; text: string; by: string; image: string | null }
export interface KnowledgeSource { n: number; kind: "knowledge"; title: string; section: number | null; file: string; page: number | null; text: string }
export type Source = FrameSource | KnowledgeSource;
export interface Report { id: string; revision: number; statements: Statement[]; guidelines_retrieved: number; sources: Source[] }
export interface Status {
  phase: string; message: string; steps: Step[]; report: Report | null; violations: string[]; approver: string | null;
  pdf_ready: boolean; upload: { size: number } | null; notice: { id: number; text: string } | null; regenerations_left: number;
}
export interface Citation { n: number; title: string; page: number | null; file: string; text: string }
export interface Answer { answer: string; covered: boolean; citations: Citation[] }
export interface Reply<T> { status: number; data: T }
export type Failure = { error?: string };

let token = "";

export async function startSession(): Promise<void> {
  const r = await fetch("/api/session");
  if (!r.ok) throw new Error("no session");
  token = ((await r.json()) as { token: string }).token;
}

async function call<T>(path: string, init: RequestInit = {}): Promise<Reply<T & Failure>> {
  const r = await fetch(path, { ...init, headers: { "X-CSRF-Token": token, ...(init.headers as Record<string, string> | undefined) } });
  let data = {};
  try { data = JSON.parse(await r.text()); } catch { /* an empty or non-JSON body is treated as no data */ }
  return { status: r.status, data: data as T & Failure };
}

export const get = <T,>(path: string) => call<T>(path);
export const post = <T,>(path: string, body: object = {}) =>
  call<T>(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

const withToken = (url: string) => `${url}${url.includes("?") ? "&" : "?"}t=${encodeURIComponent(token)}`;
export const frameUrl = (image: string) => withToken(image);
export const pdfUrl = () => withToken("/api/report.pdf");

/** XHR, not fetch, because only XHR reports upload progress. */
export function upload(file: File, onProgress: (loaded: number, total: number) => void): Promise<Reply<{ size?: number } & Failure>> {
  return new Promise((resolve, reject) => {
    const ext = (file.name.split(".").pop() ?? "").toLowerCase();
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `/api/upload?ext=${encodeURIComponent(ext)}`);
    xhr.setRequestHeader("X-CSRF-Token", token);
    xhr.setRequestHeader("Content-Type", "application/octet-stream");
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded, e.total); };
    xhr.onload = () => {
      let data = {};
      try { data = JSON.parse(xhr.responseText); } catch { /* no JSON body */ }
      resolve({ status: xhr.status, data });
    };
    xhr.onerror = () => reject(new Error("upload failed"));
    xhr.send(file);
  });
}

/** What every panel needs from the app shell. */
export interface Ctx { status: Status | null; say: (text: string, bad?: boolean) => void; refresh: () => Promise<void> }
