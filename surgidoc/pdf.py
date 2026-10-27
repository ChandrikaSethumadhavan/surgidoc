"""FR-10: render a human-approved report to PDF. Refuses anything that is not approved and hash-intact (AC-3).

fpdf2 core Helvetica only covers Latin-1 (German umlauts are fine); other characters become '?'.
ponytail: bundle a Unicode TTF if non-Latin-1 text ever matters.
"""
import base64
import io
import re

from fpdf import FPDF

from surgidoc.approval import NotApproved, content_hash

FOOTER = "Approved by a named clinician. AI-assisted documentation; not a diagnosis."


def _t(text):
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", str(text))
    return text.encode("latin-1", "replace").decode("latin-1")


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, _t(f"{FOOTER}  Page {self.page_no()}/{{nb}}"), align="C")


def citation_numbers(statements):
    """Source id -> 1, 2, 3 ... in order of first citation, so readers never see internal ids like F0 or K1."""
    ids = dict.fromkeys(sid for st in statements for sid in (st["sources"] if isinstance(st, dict) else st.sources))
    return {sid: n for n, sid in enumerate(ids, 1)}


def is_guideline(statement):
    """True when every cited source is a guideline passage (K ids): a reminder, not a finding from the video."""
    srcs = statement["sources"] if isinstance(statement, dict) else statement.sources
    return bool(srcs) and all(str(c).startswith("K") for c in srcs)


def clock(t):
    t = int(t or 0)
    return f"{t // 60:02d}:{t % 60:02d}"


def _verify(report):
    if not isinstance(report, dict) or report.get("status") != "APPROVED":
        raise NotApproved("only an APPROVED report can be rendered")
    name = report.get("approver_name")
    if not isinstance(name, str) or not name.strip() or not report.get("approved_at"):
        raise NotApproved("report has no named approver")
    if content_hash(report["statements"], report["sources"]) != report.get("content_hash"):
        raise NotApproved("report content changed since approval")


def render_pdf(report):
    _verify(report)
    pdf = _Pdf(format="A4")
    pdf.alias_nb_pages()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(True, 20)
    pdf.add_page()
    pdf.set_title("Endoscopy procedure report")
    w = pdf.epw

    def line(text, size=10, style="", gap=1):
        pdf.set_font("Helvetica", style, size)
        pdf.multi_cell(w, size * 0.5 + 1.5, _t(text), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(gap)

    line("Endoscopy Procedure Report", 16, "B", 2)
    line("Status: APPROVED", 11, "B")
    line(f"Approved by: {report['approver_name']}   on {report['approved_at']}", 10)
    pdf.ln(3)
    nums = citation_numbers(report["statements"])
    numbered = list(enumerate(report["statements"], 1))
    video = [(i, s) for i, s in numbered if not is_guideline(s)]
    guide = [(i, s) for i, s in numbered if is_guideline(s)]
    for heading, rows in (("Findings from the video" if guide else "Findings", video),
                          ("Guideline reminders for this report", guide)):
        if rows:
            line(heading, 13, "B", 2)
            for i, s in rows:
                line(f"{i}. {s['text']}  [{', '.join(str(nums[c]) for c in s['sources'] if c in nums)}]", 10, "", 2)
            pdf.ln(3)
    line("Sources", 13, "B", 2)
    for sid, n in nums.items():
        src = report["sources"].get(sid)
        if not src:
            continue
        if src.get("kind") == "knowledge":
            where = f"p. {src['page']}" if src.get("page") else f"section {src['index']}"
            line(f"[{n}] Guideline: {src.get('title') or src['source']}, {where}", 9, "B", 0)
            line(f"\"{src['text'][:400]}\"", 8, "I", 2)
            continue
        line(f"[{n}] Video frame at {clock(src.get('t'))}", 9, "B", 1)
        try:
            pdf.image(io.BytesIO(base64.b64decode(src["image"])), x=pdf.l_margin, w=60)
            pdf.ln(1)
        except Exception:  # no or unreadable picture: the finding text below still stands
            pass
        line(f"Finding: {src['text'][:300]}", 8, "", 2)
    return bytes(pdf.output())
