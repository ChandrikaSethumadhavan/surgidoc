"""FR-10: render a human-approved report to PDF. Refuses anything that is not approved and hash-intact (AC-3).

fpdf2 core Helvetica only covers Latin-1 (German umlauts are fine); other characters become '?'.
ponytail: bundle a Unicode TTF if non-Latin-1 text ever matters.
"""
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
    line(f"Report {report['report_id']}, revision {report['revision']}, content hash {report['content_hash'][:12]}", 8, "", 4)
    line("Findings", 13, "B", 2)
    for i, s in enumerate(report["statements"], 1):
        line(f"{i}. {s['text']}  [{', '.join(s['sources'])}]", 10, "", 2)
    pdf.ln(3)
    line("Sources", 13, "B", 2)
    for sid, src in report["sources"].items():
        where = f"{src['source']} (chunk {src['index']})" if src.get("kind") == "knowledge" else "video frame finding"
        line(f"[{sid}] {where}: {src['text'][:300]}", 8, "", 1)
    return bytes(pdf.output())
