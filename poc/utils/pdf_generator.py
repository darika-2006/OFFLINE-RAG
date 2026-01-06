from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from datetime import datetime
import os


def generate_pdf(answer_text: str) -> str:
    """
    Converts structured text + markdown tables into a clean PDF
    Returns path to generated PDF
    """

    # ---------- output directory ----------
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    filename = f"answer_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    file_path = os.path.join(output_dir, filename)
    file_path = os.path.abspath(file_path)

    # ---------- document ----------
    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="Heading",
        fontSize=14,
        spaceAfter=10,
        leading=16,
        textColor=colors.darkblue,
        bold=True
    ))

    styles.add(ParagraphStyle(
        name="Body",
        fontSize=10,
        leading=14,
        spaceAfter=6
    ))

    content = []

    lines = answer_text.split("\n")
    table_buffer = []
    in_table = False

    # ---------- parse content ----------
    for line in lines:
        line = line.strip()

        # ---- markdown table detection ----
        if line.startswith("|") and "|" in line:
            in_table = True
            table_buffer.append(line)
            continue

        # ---- end of table ----
        if in_table and not line.startswith("|"):
            content.append(_build_table(table_buffer))
            content.append(Spacer(1, 12))
            table_buffer = []
            in_table = False

        # ---- headings ----
        if line.startswith("###"):
            content.append(Paragraph(line.replace("###", "").strip(), styles["Heading"]))
            content.append(Spacer(1, 6))

        # ---- bullet points ----
        elif line.startswith("- "):
            content.append(Paragraph("• " + line[2:], styles["Body"]))

        # ---- normal text ----
        elif line:
            content.append(Paragraph(line, styles["Body"]))

    # flush table if file ends with table
    if table_buffer:
        content.append(_build_table(table_buffer))

    doc.build(content)
    return file_path


def _build_table(table_lines):
    """
    Converts markdown-style table to ReportLab Table
    """
    data = []

    for row in table_lines:
        if "---" in row:
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        data.append(cells)

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
    ]))

    return table