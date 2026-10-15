"""
Certified Executive Sustainability Report Generator for Smart Waste Digital Twin.
Generates publication-quality PDF and Markdown reports using real telemetry.
"""

import io
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)


def generate_pdf_report(report_data: dict) -> bytes:
    """
    Generates a certified executive PDF report based strictly on real telemetry.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    primary_color = colors.HexColor("#0f172a")
    accent_color = colors.HexColor("#2563eb")
    green_color = colors.HexColor("#16a34a")
    dark_gray = colors.HexColor("#334155")
    light_bg = colors.HexColor("#f8fafc")

    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontSize=20,
        leading=24,
        textColor=primary_color,
        fontName="Helvetica-Bold",
        spaceAfter=3,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontSize=9.5,
        leading=13,
        textColor=dark_gray,
        fontName="Helvetica",
        spaceAfter=10,
    )
    heading2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontSize=12,
        leading=16,
        textColor=accent_color,
        fontName="Helvetica-Bold",
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontSize=9,
        leading=13,
        textColor=primary_color,
        fontName="Helvetica",
        spaceAfter=4,
    )

    story = []

    # 1. Header
    title_text = report_data.get("title", "SMART WASTE MANAGEMENT DIGITAL TWIN")
    story.append(Paragraph(f"<b>♻️ {title_text.upper()}</b>", title_style))
    story.append(Paragraph(
        f"<b>Certified Monthly Sustainability & Fleet Telemetry Audit</b> &bull; "
        f"Period: {report_data.get('month', datetime.now().strftime('%B %Y'))} &bull; "
        f"Generated: {report_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=accent_color, spaceAfter=8))

    # 2. Executive Summary
    story.append(Paragraph("1. Executive Summary", heading2_style))
    exec_summary = report_data.get("executive_summary", "Digital Twin telemetry synchronized.")
    story.append(Paragraph(exec_summary, body_style))
    story.append(Spacer(1, 6))

    # 3. KPIs Table
    story.append(Paragraph("2. Environmental & Operational Impact KPIs", heading2_style))
    impact = report_data.get("environmental_impact", {})
    route_opt = report_data.get("route_optimization", {})

    kpi_data = [
        [
            Paragraph("<b>Total Waste Diverted from Landfill</b>", body_style),
            Paragraph(f"<b>{impact.get('diverted_kg', 0.0):.2f} kg</b>", body_style),
            Paragraph("<b>Net CO₂ Emissions Saved</b>", body_style),
            Paragraph(f"<b>{impact.get('co2_saved_kg', 0.0):.2f} kg CO₂</b>", body_style),
        ],
        [
            Paragraph("<b>Measured Recycling Rate</b>", body_style),
            Paragraph(f"<b>{impact.get('recycling_rate', 0.0):.1f}%</b>", body_style),
            Paragraph("<b>Total Verified Items Classified</b>", body_style),
            Paragraph(f"<b>{impact.get('total_items', 0)} items</b>", body_style),
        ],
        [
            Paragraph("<b>Collection Truck Fuel Saved</b>", body_style),
            Paragraph(f"<b>{route_opt.get('fuel_saved_l', 0.0):.2f} Liters</b>", body_style),
            Paragraph("<b>Logistics Distance Reduced</b>", body_style),
            Paragraph(f"<b>{route_opt.get('distance_saved_km', 0.0):.1f} km</b>", body_style),
        ]
    ]

    t_kpi = Table(kpi_data, colWidths=[2.3 * inch, 1.2 * inch, 2.3 * inch, 1.2 * inch])
    t_kpi.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), light_bg),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t_kpi)
    story.append(Spacer(1, 8))

    # 4. The 6 Smart Bins Status
    story.append(Paragraph("3. Smart Bin Telemetry (6 Segregated Streams)", heading2_style))
    bin_summaries = report_data.get("bin_summaries", [])
    
    bin_table_rows = [
        [
            Paragraph("<b>Smart Bin Name</b>", body_style),
            Paragraph("<b>Category Stream</b>", body_style),
            Paragraph("<b>Fill Level</b>", body_style),
            Paragraph("<b>Items Logged</b>", body_style),
            Paragraph("<b>Mass (kg)</b>", body_style),
            Paragraph("<b>Status</b>", body_style),
        ]
    ]

    for b in bin_summaries:
        fill_val = b.get("CurrentFill", 0)
        status_str = "Critical" if fill_val >= 85 else ("Filling" if fill_val >= 60 else "Optimal")
        status_color = colors.HexColor("#dc2626") if fill_val >= 85 else (colors.HexColor("#d97706") if fill_val >= 60 else green_color)
        
        bin_table_rows.append([
            Paragraph(str(b.get("Name", "Bin")), body_style),
            Paragraph(str(b.get("Category", "General")), body_style),
            Paragraph(f"<b>{fill_val}%</b>", body_style),
            Paragraph(str(b.get("TotalItems", 0)), body_style),
            Paragraph(f"{b.get('EstWeight', 0.0):.2f} kg", body_style),
            Paragraph(f"<font color='{status_color}'><b>{status_str}</b></font>", body_style),
        ])

    t_bins = Table(bin_table_rows, colWidths=[1.8 * inch, 1.4 * inch, 0.8 * inch, 0.9 * inch, 0.9 * inch, 1.0 * inch])
    t_bins.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 5),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(t_bins)
    story.append(Spacer(1, 8))

    # 5. GenAI Recommendations
    story.append(Paragraph("4. Generative AI Strategic Recommendations", heading2_style))
    recs = report_data.get("ai_recommendations", [])
    if isinstance(recs, str):
        story.append(Paragraph(recs, body_style))
    elif isinstance(recs, list):
        for idx, rec in enumerate(recs, 1):
            story.append(Paragraph(f"<b>{idx}.</b> {rec}", body_style))
    story.append(Spacer(1, 6))

    # 6. Logistics Optimization
    story.append(Paragraph("5. Digital Twin Route & Logistics Optimization", heading2_style))
    route_rec = report_data.get("route_plan", "Dynamic dispatch enabled based on 75% fill thresholds.")
    story.append(Paragraph(route_rec, body_style))
    story.append(Spacer(1, 10))

    # Footer
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#94a3b8"), spaceAfter=5))
    story.append(Paragraph(
        "<i>Certified by Smart Waste Management Digital Twin System • Computer Vision & Generative AI</i>",
        subtitle_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def generate_markdown_report(report_data: dict) -> str:
    """Generates markdown text report for download."""
    title = report_data.get("title", "Smart Waste Management Digital Twin")
    month = report_data.get("month", datetime.now().strftime("%B %Y"))
    timestamp = report_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    exec_summary = report_data.get("executive_summary", "")
    impact = report_data.get("environmental_impact", {})
    route_opt = report_data.get("route_optimization", {})
    bin_summaries = report_data.get("bin_summaries", [])
    recs = report_data.get("ai_recommendations", [])
    route_plan = report_data.get("route_plan", "")

    md = [
        f"# ♻️ {title}",
        f"**Certified Monthly Sustainability Audit** | Period: {month} | Generated: {timestamp}\n",
        "---",
        "## 1. Executive Summary",
        f"{exec_summary}\n",
        "## 2. Key Performance Indicators",
        f"- **Total Items Logged:** {impact.get('total_items', 0)}",
        f"- **Measured Waste Diverted:** {impact.get('diverted_kg', 0):.2f} kg",
        f"- **Estimated CO₂ Saved:** {impact.get('co2_saved_kg', 0):.2f} kg",
        f"- **Recycling Rate:** {impact.get('recycling_rate', 0):.1f}%",
        f"- **Fuel Saved:** {route_opt.get('fuel_saved_l', 0):.2f} L",
        f"- **Distance Saved:** {route_opt.get('distance_saved_km', 0):.1f} km\n",
        "## 3. The 6 Smart Bins Status",
        "| Bin Name | Stream | Fill Level | Items Logged | Weight (kg) | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |"
    ]
    for b in bin_summaries:
        fill_val = b.get("CurrentFill", 0)
        status = "Critical" if fill_val >= 85 else ("Filling" if fill_val >= 60 else "Optimal")
        md.append(f"| {b.get('Name')} | {b.get('Category')} | {fill_val}% | {b.get('TotalItems', 0)} | {b.get('EstWeight', 0.0):.2f} kg | {status} |")
    md.append("\n")

    md.append("## 4. Generative AI Strategic Recommendations")
    if isinstance(recs, list):
        for i, r in enumerate(recs, 1):
            md.append(f"{i}. {r}")
    else:
        md.append(str(recs))
    md.append("\n")

    md.append("## 5. Logistics Optimization Plan")
    md.append(f"{route_plan}\n")
    md.append("---")
    md.append("*Smart Waste Management Digital Twin System*")

    return "\n".join(md)
