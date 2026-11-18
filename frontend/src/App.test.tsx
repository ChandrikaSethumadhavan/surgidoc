import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";
import type { Status } from "./api";

const draft = (): Status => ({
  phase: "pending_review", message: "Draft ready: a reviewer must approve or reject it.",
  steps: [{ id: "frames", label: "Extract frames", done: true }], violations: [], approver: null, pdf_ready: false,
  upload: null, notice: null, regenerations_left: 3,
  report: {
    id: "r1", revision: 1, guidelines_retrieved: 1,
    statements: [
      { text: "Liver edge visible.", cites: [1], guideline: false },
      { text: "Consent must be documented.", cites: [2], guideline: true },
    ],
    sources: [
      { n: 1, kind: "frame", time: "0:04", text: "liver edge", by: "Gemini vision model", image: "/api/frame?id=F0" },
      { n: 2, kind: "knowledge", title: "S2k guideline", section: 3, file: "s2k.pdf", page: 12, text: "Consent is documented." },
    ],
  },
});

let status: Status;
let calls: { path: string; body: any; token: string | null }[];
let replies: Record<string, unknown>;

beforeEach(() => {
  status = draft();
  calls = [];
  replies = {};
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    if (url === "/api/session") return new Response(JSON.stringify({ token: "tok123" }));
    if (url === "/api/status") return new Response(JSON.stringify(status));
    const headers = (init?.headers ?? {}) as Record<string, string>;
    calls.push({ path: url, body: init?.body ? JSON.parse(String(init.body)) : null, token: headers["X-CSRF-Token"] ?? null });
    return new Response(JSON.stringify(replies[url] ?? { ok: true }));
  }));
});
afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

const statement = (n: number) => screen.findByLabelText(`Statement ${n}`) as Promise<HTMLTextAreaElement>;
const edit = async (value: string) => fireEvent.change(await statement(1), { target: { value } });
const button = (name: string) => screen.getByRole("button", { name });

test("report text is rendered as text, never as HTML", async () => {
  const evil = '<img src="x" onerror="alert(1)"> hi';
  status.phase = "approved";
  status.report!.statements[0]!.text = evil;
  render(<App pollMs={20} />);
  expect(await screen.findByText(evil)).toBeInTheDocument();
  expect(document.querySelector('img[src="x"]')).toBeNull();
});

test("a poll that changes the status but not the draft keeps unsaved edits", async () => {
  render(<App pollMs={20} />);
  await edit("My edit");
  status.steps = [...status.steps, { id: "later", label: "Later step", done: true }];
  expect(await screen.findByText(/Later step/)).toBeInTheDocument();
  expect(await statement(1)).toHaveValue("My edit");
});

test("approve is refused while edits are unsaved, then works after saving", async () => {
  render(<App pollMs={20} />);
  fireEvent.change(screen.getByPlaceholderText("Reviewer name"), { target: { value: "Rita" } });
  await edit("My edit");
  fireEvent.click(button("Approve"));
  expect(await screen.findByText(/unsaved edits/)).toBeInTheDocument();
  expect(calls.some((c) => c.path === "/api/approve")).toBe(false);

  fireEvent.click(button("Save edits"));
  await waitFor(() => expect(calls.find((c) => c.path === "/api/edit")).toBeDefined());
  expect(calls.find((c) => c.path === "/api/edit")!.body).toEqual({
    name: "Rita", revision: 1,
    statements: [{ i: 0, text: "My edit" }, { i: 1, text: "Consent must be documented." }],
  });
  status = {
    ...status, steps: [...status.steps, { id: "saved", label: "Saved step", done: true }],
    report: { ...status.report!, revision: 2, statements: [{ ...status.report!.statements[0]!, text: "My edit" }, status.report!.statements[1]!] },
  };
  expect(await screen.findByText(/Saved step/)).toBeInTheDocument(); // the poll with the new revision has arrived
  expect(await statement(1)).toHaveValue("My edit");
  fireEvent.click(button("Approve"));
  await waitFor(() => expect(calls.some((c) => c.path === "/api/approve")).toBe(true));
});

test("a removed statement is left out of the saved edit", async () => {
  render(<App pollMs={20} />);
  await statement(1);
  fireEvent.click(screen.getAllByRole("button", { name: "Remove" })[0]!);
  fireEvent.click(button("Save edits"));
  await waitFor(() => expect(calls.find((c) => c.path === "/api/edit")).toBeDefined());
  expect(calls.find((c) => c.path === "/api/edit")!.body.statements).toEqual([{ i: 1, text: "Consent must be documented." }]);
});

test("frame and PDF URLs carry the session token", async () => {
  status.phase = "approved";
  status.approver = "Rita";
  status.pdf_ready = true;
  render(<App pollMs={20} />);
  const frames = await screen.findAllByAltText("Video frame at 0:04");
  expect(frames[0]).toHaveAttribute("src", "/api/frame?id=F0&t=tok123");
  expect(screen.getByRole("link", { name: "Download approved PDF" })).toHaveAttribute("href", "/api/report.pdf?t=tok123");
  expect(screen.getByText("Approved by Rita.")).toBeInTheDocument();
});

test("every action sends the token header and the review controls are locked outside review", async () => {
  status.phase = "running";
  render(<App pollMs={20} />);
  await screen.findByText("running");
  for (const name of ["Approve", "Reject", "Save edits", "Run pipeline", "New video"]) expect(button(name)).toBeDisabled();
  status.phase = "uploaded";
  await waitFor(() => expect(button("Run pipeline")).toBeEnabled());
  fireEvent.click(button("Run pipeline"));
  await waitFor(() => expect(calls.find((c) => c.path === "/api/run")).toBeDefined());
  expect(calls.find((c) => c.path === "/api/run")!.token).toBe("tok123");
});

test("guideline answers and their quotes are shown as text", async () => {
  replies["/api/ask"] = { answer: "Tell patients about risks [1].", covered: true, citations: [{ n: 1, title: "S2k", page: 9, file: "s2k.pdf", text: "<b>risks</b>" }] };
  render(<App pollMs={20} />);
  fireEvent.change(screen.getByPlaceholderText("Your name"), { target: { value: "Rita" } });
  fireEvent.change(screen.getByLabelText("Question"), { target: { value: "What about consent?" } });
  fireEvent.click(button("Ask"));
  expect(await screen.findByText("Tell patients about risks [1].")).toBeInTheDocument();
  expect(screen.getByText("“<b>risks</b>”")).toBeInTheDocument();
  expect(calls.find((c) => c.path === "/api/ask")!.body).toEqual({ name: "Rita", question: "What about consent?" });
});

test("guideline reminders and the page of each guideline source are shown", async () => {
  render(<App pollMs={20} />);
  expect(await screen.findByText("Guideline reminder")).toBeInTheDocument();
  expect(screen.getAllByText("S2k guideline, page 12").length).toBeGreaterThan(0);
  expect(screen.getByText("“Consent is documented.”")).toBeInTheDocument();
  expect(screen.getByText(/Video evidence · AI description by Gemini vision model/)).toBeInTheDocument();
});
