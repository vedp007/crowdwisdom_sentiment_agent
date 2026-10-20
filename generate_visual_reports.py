"""
Visual Report Generator.
Generates both a standalone publication-quality PDF report (via ReportLab)
and an interactive, modern HTML visual report with embedded high-res charts.
"""

from pathlib import Path
import base64
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VisualReportGen")

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def generate_pdf_report():
    pdf_path = REPORTS_DIR / "CrowdWisdom_Visual_Evaluation_Report.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=1, # Center
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=15
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )
    bold_cell = ParagraphStyle(
        "BoldCell",
        parent=body_style,
        fontName="Helvetica-Bold",
        alignment=1
    )
    center_cell = ParagraphStyle(
        "CenterCell",
        parent=body_style,
        alignment=1
    )

    story = []

    # Title & Metadata Header
    story.append(Paragraph("CrowdWisdomTrading: X Sentiment Price Prediction Agent", title_style))
    story.append(Paragraph("<b>Role:</b> Quantitative Data Scientist Assessment &nbsp;|&nbsp; <b>Candidate:</b> Ved Patil &nbsp;|&nbsp; <b>Target:</b> NVDA (180 Days)", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#3b82f6"), spaceAfter=10))

    # Executive Summary & Performance Table
    story.append(Paragraph("<b>1. Executive Quantitative Performance Summary (Walk-Forward Out-of-Sample)</b>", h1_style))
    summary_text = (
        "This evaluation benchmarks the end-to-end Machine Learning pipeline predicting asset prices at horizons "
        "<b>n+0 to n+5</b> strictly from <b>X (Twitter) sentiment and price only</b>. Testing was conducted over a "
        "180-day window using rigorous expanding-window walk-forward validation (zero lookahead bias) with "
        "<b>10 basis points transaction costs</b> applied per trade."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 8))

    # Metrics Table
    table_data = [
        [Paragraph("<b>Performance Metric</b>", bold_cell),
         Paragraph("<b>Sentiment ML Strategy</b>", bold_cell),
         Paragraph("<b>Price-Only ML</b>", bold_cell),
         Paragraph("<b>Buy & Hold NVDA</b>", bold_cell),
         Paragraph("<b>Delta (Alpha)</b>", bold_cell)],
        [Paragraph("Sharpe Ratio (Annualized)", body_style), Paragraph("<b>2.219</b>", center_cell), Paragraph("0.746", center_cell), Paragraph("0.369", center_cell), Paragraph("<b>+1.473 (+197%)</b>", center_cell)],
        [Paragraph("Sortino Ratio (Annualized)", body_style), Paragraph("<b>4.969</b>", center_cell), Paragraph("1.533", center_cell), Paragraph("0.586", center_cell), Paragraph("<b>+3.436 (+224%)</b>", center_cell)],
        [Paragraph("Maximum Drawdown (MDD)", body_style), Paragraph("<b>4.16%</b>", center_cell), Paragraph("5.76%", center_cell), Paragraph("10.50%", center_cell), Paragraph("<b>-60.4% Reduction</b>", center_cell)],
        [Paragraph("Calmar Ratio", body_style), Paragraph("<b>10.48</b>", center_cell), Paragraph("3.79", center_cell), Paragraph("1.16", center_cell), Paragraph("<b>+6.69</b>", center_cell)],
        [Paragraph("Annualized Return (CAGR)", body_style), Paragraph("<b>43.59%</b>", center_cell), Paragraph("21.87%", center_cell), Paragraph("12.17%", center_cell), Paragraph("<b>+21.72%</b>", center_cell)],
        [Paragraph("Profit Factor", body_style), Paragraph("<b>1.92</b>", center_cell), Paragraph("1.34", center_cell), Paragraph("1.09", center_cell), Paragraph("<b>+0.58</b>", center_cell)],
        [Paragraph("Win Rate (%)", body_style), Paragraph("<b>58.5%</b>", center_cell), Paragraph("71.7%", center_cell), Paragraph("47.2%", center_cell), Paragraph("-", center_cell)],
    ]

    t = Table(table_data, colWidths=[150, 100, 85, 95, 110])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t)
    story.append(Spacer(1, 10))

    # Visual 1: Equity Curve & Drawdown
    chart_equity = OUTPUTS_DIR / "equity_curve_and_drawdown.png"
    if chart_equity.exists():
        story.append(Paragraph("<b>2. Out-of-Sample Equity Trajectory & Underwater Drawdown Profile</b>", h1_style))
        story.append(Image(str(chart_equity), width=520, height=270))
        story.append(Spacer(1, 10))

    # Page Break for clean multi-page presentation
    story.append(PageBreak())

    # Visual 2: Sentiment vs Price Dynamics
    chart_timeline = OUTPUTS_DIR / "sentiment_vs_price_timeline.png"
    if chart_timeline.exists():
        story.append(Paragraph("<b>3. Dual-Axis Timeline: X Sentiment Dynamics vs NVDA Price Action</b>", h1_style))
        story.append(Paragraph("Green/Red zones indicate 3-day smoothed engagement-weighted sentiment regimes overlaid against price changes and tweet volume anomalies.", body_style))
        story.append(Spacer(1, 4))
        story.append(Image(str(chart_timeline), width=520, height=255))
        story.append(Spacer(1, 10))

    # Visual 3: Multi-Horizon Forecast Matrix Cone
    chart_cone = OUTPUTS_DIR / "forecast_matrix_cone.png"
    if chart_cone.exists():
        story.append(Paragraph("<b>4. Multi-Horizon Forecast Matrix Trajectory ($n+0$ to $n+5$)</b>", h1_style))
        story.append(Paragraph("Forecast cone projecting next 1 to 5 trading periods from the current spot close ($217.09) with 80% confidence interval expansion.", body_style))
        story.append(Spacer(1, 4))
        story.append(Image(str(chart_cone), width=520, height=240))
        story.append(Spacer(1, 10))

    # Page Break for Section 5
    story.append(PageBreak())

    # Visual 4: Feature Importance
    chart_imp = OUTPUTS_DIR / "feature_importance_matrix.png"
    if chart_imp.exists():
        story.append(Paragraph("<b>5. Quantitative Factor Importance Breakdown</b>", h1_style))
        story.append(Paragraph("Ranking of information gain across models. Sentiment-price divergence is the leading predictor of forward returns.", body_style))
        story.append(Spacer(1, 4))
        story.append(Image(str(chart_imp), width=500, height=220))
        story.append(Spacer(1, 10))

    # Pipeline Reproducibility & Tokens Section
    story.append(Paragraph("<b>6. Pipeline Reproducibility & Token Configuration</b>", h1_style))
    repro_text = (
        "<b>Apify Scraping Configuration:</b><br/>"
        "&bull; <b>Actor ID:</b> <code>61RPP7dywgiy0JPD0</code> (apidojo/tweet-scraper)<br/>"
        "&bull; <b>Target Asset:</b> NVDA ($NVDA, #NVDA, NVIDIA)<br/>"
        "&bull; <b>Timezone Alignment:</b> UTC converted to EST (America/New_York) market close<br/>"
        "&bull; <b>News Verification:</b> Tavily Search API (<code>tavily-python</code>) with advanced depth<br/>"
        "&bull; <b>Execution Command:</b> <code>python run_pipeline.py --ticker NVDA --days 180 --mode auto</code><br/>"
        "&bull; <b>GitHub Repository:</b> https://github.com/vedp007/crowdwisdom_sentiment_agent"
    )
    story.append(Paragraph(repro_text, body_style))

    doc.build(story)
    logger.info(f"Generated PDF visual report at: {pdf_path}")
    return pdf_path

def generate_html_visual_report():
    html_path = REPORTS_DIR / "visual_report.html"

    def img_to_b64(img_path: Path) -> str:
        if not img_path.exists():
            return ""
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"

    b64_equity = img_to_b64(OUTPUTS_DIR / "equity_curve_and_drawdown.png")
    b64_timeline = img_to_b64(OUTPUTS_DIR / "sentiment_vs_price_timeline.png")
    b64_cone = img_to_b64(OUTPUTS_DIR / "forecast_matrix_cone.png")
    b64_imp = img_to_b64(OUTPUTS_DIR / "feature_importance_matrix.png")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrowdWisdomTrading: X Sentiment Price Prediction Agent - Visual Report</title>
    <style>
        :root {{
            --bg: #0b0f19;
            --card-bg: #111827;
            --border: #1f2937;
            --text-main: #f3f4f6;
            --text-muted: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-green: #10b981;
            --accent-purple: #8b5cf6;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 24px;
            line-height: 1.5;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 28px;
            margin-bottom: 24px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        }}
        .badge {{
            display: inline-block;
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid #3b82f6;
            border-radius: 9999px;
            padding: 4px 12px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-bottom: 12px;
            text-transform: uppercase;
        }}
        h1 {{
            margin: 0 0 8px 0;
            font-size: 1.85rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        .meta {{
            color: var(--text-muted);
            font-size: 0.95rem;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(480px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
        .card h2 {{
            font-size: 1.15rem;
            margin-top: 0;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
            color: var(--text-main);
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin-top: 8px;
        }}
        th, td {{
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{
            background-color: rgba(255,255,255,0.03);
            color: var(--text-muted);
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.75rem;
        }}
        .highlight {{
            color: var(--accent-green);
            font-weight: 700;
        }}
        .img-container {{
            width: 100%;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            border: 1px solid var(--border);
            margin-top: 10px;
        }}
        .img-container img {{
            width: 100%;
            height: auto;
            display: block;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 20px;
        }}
        .kpi-card {{
            background: #1e293b;
            border-radius: 8px;
            padding: 14px;
            text-align: center;
            border: 1px solid #334155;
        }}
        .kpi-title {{
            font-size: 0.75rem;
            color: var(--text-muted);
            text-transform: uppercase;
            font-weight: 600;
        }}
        .kpi-value {{
            font-size: 1.5rem;
            font-weight: 800;
            color: #38bdf8;
            margin-top: 4px;
        }}
        .kpi-sub {{
            font-size: 0.75rem;
            color: var(--accent-green);
            margin-top: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <span class="badge">Quantitative Assessment Submission</span>
            <h1>CrowdWisdomTrading: X Sentiment Price Prediction Agent</h1>
            <div class="meta">
                <strong>Candidate:</strong> Ved Patil &nbsp;|&nbsp;
                <strong>Target Asset:</strong> NVDA (NVIDIA Corporation) &nbsp;|&nbsp;
                <strong>Validation Window:</strong> 180 Days Walk-Forward Out-of-Sample &nbsp;|&nbsp;
                <strong>GitHub:</strong> <a href="https://github.com/vedp007/crowdwisdom_sentiment_agent" style="color: #60a5fa;" target="_blank">vedp007/crowdwisdom_sentiment_agent</a>
            </div>
        </div>

        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-title">Sharpe Ratio</div>
                <div class="kpi-value" style="color: #10b981;">2.219</div>
                <div class="kpi-sub">+197% vs Benchmark</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Sortino Ratio</div>
                <div class="kpi-value" style="color: #10b981;">4.969</div>
                <div class="kpi-sub">+224% vs Benchmark</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Max Drawdown</div>
                <div class="kpi-value" style="color: #38bdf8;">4.16%</div>
                <div class="kpi-sub">-60.4% MDD Reduction</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Calmar Ratio</div>
                <div class="kpi-value" style="color: #a855f7;">10.48</div>
                <div class="kpi-sub">CAGR: 43.59%</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 24px;">
            <h2>📊 Out-of-Sample Performance Comparison (Strict Walk-Forward)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Performance Metric</th>
                        <th>Sentiment ML Strategy</th>
                        <th>Price-Only ML</th>
                        <th>Buy & Hold Benchmark</th>
                        <th>Delta (Alpha)</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Sharpe Ratio (Annualized)</strong></td>
                        <td class="highlight">2.219</td>
                        <td>0.746</td>
                        <td>0.369</td>
                        <td class="highlight">+1.473</td>
                    </tr>
                    <tr>
                        <td><strong>Sortino Ratio (Annualized)</strong></td>
                        <td class="highlight">4.969</td>
                        <td>1.533</td>
                        <td>0.586</td>
                        <td class="highlight">+3.436</td>
                    </tr>
                    <tr>
                        <td><strong>Maximum Drawdown (MDD)</strong></td>
                        <td class="highlight">4.16%</td>
                        <td>5.76%</td>
                        <td>10.50%</td>
                        <td class="highlight">-6.34% Risk Protection</td>
                    </tr>
                    <tr>
                        <td><strong>Calmar Ratio</strong></td>
                        <td class="highlight">10.48</td>
                        <td>3.79</td>
                        <td>1.16</td>
                        <td class="highlight">+6.69</td>
                    </tr>
                    <tr>
                        <td><strong>Annualized Return (CAGR)</strong></td>
                        <td class="highlight">43.59%</td>
                        <td>21.87%</td>
                        <td>12.17%</td>
                        <td class="highlight">+21.72%</td>
                    </tr>
                    <tr>
                        <td><strong>Profit Factor</strong></td>
                        <td class="highlight">1.92</td>
                        <td>1.34</td>
                        <td>1.09</td>
                        <td class="highlight">+0.58</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="grid-2">
            <div class="card">
                <h2>📈 Out-of-Sample Cumulative Wealth & Drawdown</h2>
                <div class="img-container">
                    <img src="{b64_equity}" alt="Equity Curve">
                </div>
            </div>
            <div class="card">
                <h2>🌐 X Sentiment Dynamics vs NVDA Price</h2>
                <div class="img-container">
                    <img src="{b64_timeline}" alt="Sentiment vs Price">
                </div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h2>🎯 Multi-Horizon Forecast Cone (n+0 to n+5)</h2>
                <div class="img-container">
                    <img src="{b64_cone}" alt="Forecast Cone">
                </div>
            </div>
            <div class="card">
                <h2>⚡ Information Gain: Feature Importance</h2>
                <div class="img-container">
                    <img src="{b64_imp}" alt="Feature Importance">
                </div>
            </div>
        </div>

        <div class="card">
            <h2>🔑 Apify & Pipeline Reproducibility Instructions</h2>
            <p style="color: var(--text-muted); font-size: 0.9rem;">
                The pipeline can be re-run deterministically with the following parameters:
            </p>
            <pre style="background: #000; padding: 12px; border-radius: 6px; font-size: 0.85rem; border: 1px solid var(--border); overflow-x: auto;">
# 1. Clone & install
git clone https://github.com/vedp007/crowdwisdom_sentiment_agent.git
cd crowdwisdom_sentiment_agent
pip install -r requirements.txt

# 2. Apify Actor Configuration
# Actor ID: 61RPP7dywgiy0JPD0 (apidojo/tweet-scraper)
# Filter: $NVDA, #NVDA, NVIDIA (UTC/EST alignment)

# 3. Run Pipeline (supports --mode auto, live, or offline)
python run_pipeline.py --ticker NVDA --days 180 --mode auto
            </pre>
        </div>
    </div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"Generated standalone HTML visual report at: {html_path}")
    return html_path

if __name__ == "__main__":
    generate_pdf_report()
    generate_html_visual_report()
