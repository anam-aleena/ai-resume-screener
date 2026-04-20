"""
AI Resume Screener — Report Generator
Author: Aleena Anam | github.com/anam-aleena

Generates text reports, visualizations, and ranked HTML reports
for batch screening results.
"""

import os
import json
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


# ─── TERMINAL REPORT ─────────────────────────────────────────────────────────

def print_screening_report(result, scorer=None):
    """Pretty-print a single ScreeningResult to terminal."""
    c = result.candidate
    s = result.scores
    sep = "=" * 62

    print(f"\n{sep}")
    print(f"  AI RESUME SCREENER — CANDIDATE REPORT")
    print(f"  Author: Aleena Anam | github.com/anam-aleena")
    print(sep)
    print(f"  Candidate   : {c.name}")
    print(f"  Email       : {c.email or 'Not found'}")
    print(f"  Phone       : {c.phone or 'Not found'}")
    print(f"  Education   : {c.education}")
    print(f"  Experience  : {f'{c.years_experience} years' if c.years_experience else 'Not stated'}")
    print(sep)
    print(f"  SCORE BREAKDOWN")
    print(f"  {'TF-IDF Similarity':<25} {s.tfidf_similarity:>6.1f} / 100  (weight 25%)")
    print(f"  {'Skill Match':<25} {s.skill_match:>6.1f} / 100  (weight 35%)")
    print(f"  {'Experience Match':<25} {s.experience_match:>6.1f} / 100  (weight 20%)")
    print(f"  {'Education Match':<25} {s.education_match:>6.1f} / 100  (weight 10%)")
    print(f"  {'Keyword Coverage':<25} {s.keyword_coverage:>6.1f} / 100  (weight 10%)")
    print(f"  {'-'*40}")
    print(f"  {'TOTAL SCORE':<25} {s.total_score:>6.1f} / 100")
    print(f"\n  DECISION: {s.decision_label}")
    print(sep)

    # Skills
    summary = result.skill_analysis.get("_summary", {})
    matched = summary.get("matched_skills", [])
    missing = summary.get("missing_skills", [])
    bonus   = summary.get("bonus_skills", [])
    if matched:
        print(f"\n  MATCHED SKILLS ({len(matched)})")
        print(f"  {', '.join(matched)}")
    if missing:
        print(f"\n  MISSING SKILLS ({len(missing)})")
        print(f"  {', '.join(missing)}")
    if bonus:
        print(f"\n  BONUS SKILLS (not required)")
        print(f"  {', '.join(bonus)}")

    # Recommendations
    print(f"\n  RECOMMENDATIONS")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  {i}. {rec}")
    print(f"\n  Screened at: {result.screened_at}")
    print(sep + "\n")


# ─── VISUALIZATION ───────────────────────────────────────────────────────────

def _score_color(score: float) -> str:
    if score >= 80: return "#4CAF50"
    if score >= 65: return "#2196F3"
    if score >= 50: return "#FF9800"
    if score >= 35: return "#FF5722"
    return "#F44336"


def plot_score_breakdown(result, save_path: str = None):
    """Radar/bar chart of score breakdown for a single candidate."""
    s = result.scores
    categories = ["TF-IDF\nSimilarity", "Skill\nMatch",
                  "Experience\nMatch", "Education\nMatch", "Keyword\nCoverage"]
    values = [s.tfidf_similarity, s.skill_match,
              s.experience_match, s.education_match, s.keyword_coverage]
    colors = [_score_color(v) for v in values]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Resume Screening Report — {result.candidate.name}",
                 fontsize=14, fontweight="bold", y=1.02)

    # Bar chart
    bars = ax1.barh(categories, values, color=colors, edgecolor="white", height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel("Score (out of 100)")
    ax1.set_title("Score Breakdown", fontweight="bold")
    ax1.axvline(x=65, color="gray", linestyle="--", alpha=0.5, label="Good match threshold")
    for bar, val in zip(bars, values):
        ax1.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}", va="center", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=9)

    # Gauge — total score
    total = s.total_score
    gauge_color = _score_color(total)
    ax2.pie([total, 100 - total],
            colors=[gauge_color, "#EEEEEE"],
            startangle=90,
            counterclock=False,
            wedgeprops={"width": 0.4, "edgecolor": "white"})
    ax2.text(0, 0, f"{total:.0f}", ha="center", va="center",
             fontsize=36, fontweight="bold", color=gauge_color)
    ax2.text(0, -0.35, "Total Score", ha="center", va="center",
             fontsize=11, color="gray")
    ax2.text(0, 0.35, result.scores.decision.replace("_", " ").title(),
             ha="center", va="center", fontsize=10,
             fontweight="bold", color=gauge_color)
    ax2.set_title("Overall Decision", fontweight="bold")

    plt.tight_layout()
    path = save_path or f"{REPORT_DIR}/score_{result.candidate.name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[REPORT] Score chart saved → {path}")
    return path


def plot_batch_ranking(results: list, save_path: str = None):
    """Horizontal bar chart ranking all candidates."""
    names  = [r.candidate.name for r in results]
    scores = [r.scores.total_score for r in results]
    colors = [_score_color(s) for s in scores]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.7 + 1)))
    bars = ax.barh(names, scores, color=colors, edgecolor="white", height=0.55)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Total Score (out of 100)", fontsize=11)
    ax.set_title("Candidate Ranking — AI Resume Screener", fontsize=13, fontweight="bold")

    # Threshold lines
    ax.axvline(x=80, color="#4CAF50", linestyle="--", alpha=0.6, label="Strong match (80)")
    ax.axvline(x=65, color="#2196F3", linestyle="--", alpha=0.6, label="Good match (65)")
    ax.axvline(x=50, color="#FF9800", linestyle="--", alpha=0.6, label="Potential (50)")

    for bar, score, result in zip(bars, scores, results):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}  {result.scores.decision.replace('_', ' ').title()}",
                va="center", fontsize=9)

    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="lower right")

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Strong match"),
        mpatches.Patch(color="#2196F3", label="Good match"),
        mpatches.Patch(color="#FF9800", label="Potential"),
        mpatches.Patch(color="#FF5722", label="Weak match"),
        mpatches.Patch(color="#F44336", label="Not suitable"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)
    plt.tight_layout()

    path = save_path or f"{REPORT_DIR}/candidate_ranking.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[REPORT] Ranking chart saved → {path}")
    return path


def plot_skill_heatmap(results: list, save_path: str = None):
    """Skill match heatmap across all candidates."""
    from src.nlp_engine import SKILL_TAXONOMY
    categories = [c for c in SKILL_TAXONOMY if c != "soft_skills"]
    names = [r.candidate.name for r in results]

    matrix = []
    for result in results:
        row = []
        for cat in categories:
            cat_data = result.skill_analysis.get(cat, {})
            score = cat_data.get("score", 0) if isinstance(cat_data, dict) else 0
            row.append(score * 100)
        matrix.append(row)

    matrix = np.array(matrix)
    labels = [c.replace("_", "\n") for c in categories]

    fig, ax = plt.subplots(figsize=(max(8, len(categories) * 1.4), max(4, len(names) * 0.7 + 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_title("Skill Match Heatmap — All Candidates", fontsize=13, fontweight="bold")

    for i in range(len(names)):
        for j in range(len(categories)):
            val = matrix[i, j]
            color = "white" if val < 30 or val > 70 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Skill Match %", shrink=0.8)
    plt.tight_layout()

    path = save_path or f"{REPORT_DIR}/skill_heatmap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[REPORT] Skill heatmap saved → {path}")
    return path


# ─── HTML REPORT ─────────────────────────────────────────────────────────────

def generate_html_report(results: list, jd_title: str = "Software Engineer") -> str:
    """Generate a polished HTML ranking report for all candidates."""

    def score_badge(score):
        if score >= 80: color, label = "#4CAF50", "Strong"
        elif score >= 65: color, label = "#2196F3", "Good"
        elif score >= 50: color, label = "#FF9800", "Potential"
        elif score >= 35: color, label = "#FF5722", "Weak"
        else: color, label = "#F44336", "No"
        return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:600">{label}</span>'

    rows = ""
    for rank, r in enumerate(results, 1):
        c, s = r.candidate, r.scores
        summary = r.skill_analysis.get("_summary", {})
        matched = ", ".join(summary.get("matched_skills", [])[:6]) or "—"
        missing = ", ".join(summary.get("missing_skills", [])[:4]) or "—"
        rows += f"""
        <tr>
          <td style="text-align:center;font-weight:700;color:#555">#{rank}</td>
          <td><strong>{c.name}</strong><br><small style="color:#888">{c.email or ''}</small></td>
          <td style="text-align:center">
            <div style="font-size:22px;font-weight:700;color:{'#4CAF50' if s.total_score>=80 else '#2196F3' if s.total_score>=65 else '#FF9800' if s.total_score>=50 else '#F44336'}">{s.total_score:.0f}</div>
            {score_badge(s.total_score)}
          </td>
          <td style="text-align:center">{s.skill_match:.0f}%</td>
          <td style="text-align:center">{s.experience_match:.0f}%</td>
          <td style="text-align:center">{s.education_match:.0f}%</td>
          <td style="font-size:12px;color:#2e7d32">{matched}</td>
          <td style="font-size:12px;color:#c62828">{missing}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>AI Resume Screener Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #333; }}
    h1 {{ color: #1a237e; }} h2 {{ color: #283593; border-bottom: 2px solid #e8eaf6; padding-bottom: 8px; }}
    table {{ width:100%; border-collapse:collapse; margin-top:16px; }}
    th {{ background:#1a237e; color:white; padding:10px 12px; text-align:left; font-size:13px; }}
    td {{ padding:10px 12px; border-bottom:1px solid #e8eaf6; vertical-align:top; font-size:13px; }}
    tr:hover {{ background:#f5f5f5; }}
    .meta {{ background:#e8eaf6; padding:16px; border-radius:8px; margin-bottom:24px; }}
    footer {{ margin-top:40px; color:#888; font-size:12px; border-top:1px solid #eee; padding-top:12px; }}
  </style>
</head>
<body>
  <h1>AI Resume Screener — Screening Report</h1>
  <div class="meta">
    <strong>Role:</strong> {jd_title} &nbsp;|&nbsp;
    <strong>Candidates screened:</strong> {len(results)} &nbsp;|&nbsp;
    <strong>Generated:</strong> {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}
  </div>
  <h2>Candidate Rankings</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Candidate</th><th>Total Score</th>
        <th>Skills</th><th>Experience</th><th>Education</th>
        <th>Matched Skills</th><th>Missing Skills</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  <footer>
    Generated by AI Resume Screener &nbsp;|&nbsp;
    Author: Aleena Anam &nbsp;|&nbsp;
    <a href="https://github.com/anam-aleena/ai-resume-screener">github.com/anam-aleena</a>
  </footer>
</body>
</html>"""

    path = f"{REPORT_DIR}/screening_report.html"
    with open(path, "w") as f:
        f.write(html)
    print(f"[REPORT] HTML report saved → {path}")
    return path
