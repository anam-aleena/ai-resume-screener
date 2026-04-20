# AI Resume Screener — NLP-Powered Candidate Ranking System

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Skill%20Matching-orange)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

> **Automatically screen, score, and rank resumes against any job description — in seconds.**  
> Built with NLP, TF-IDF semantic similarity, a 200+ skill taxonomy, and a production-ready REST API.

---

## What It Does

Recruiting teams spend **23 hours** screening resumes per hire on average. This system automates that — scoring every resume across 5 weighted dimensions and ranking candidates instantly.

```
Resume + Job Description  →  AI Screener  →  Score (0–100) + Decision + Recommendations
```

**Input:** Any raw resume text + any job description text  
**Output:** Ranked candidates with detailed scores, skill gap analysis, and hiring recommendations

---

## Features

| Feature | Description |
|---|---|
| TF-IDF Semantic Matching | Measures overall resume-JD semantic alignment |
| 200+ Skill Taxonomy | Categorized skills across 7 domains (ML/AI, Cloud, Data Science, etc.) |
| Skill Gap Analysis | Matched skills, missing skills, bonus skills per category |
| Experience Scoring | Extracts and compares years of experience |
| Education Matching | Detects degree level (PhD, Master's, Bachelor's, etc.) |
| Keyword Coverage | JD keyword presence/absence in resume |
| Weighted Scoring | Configurable weights per dimension |
| Batch Screening | Screen 50+ candidates, auto-ranked by score |
| REST API | FastAPI endpoints for integration into any system |
| Visual Reports | Score charts, skill heatmaps, ranking plots, HTML report |

---

## Scoring System

| Dimension | Weight | What It Measures |
|---|---|---|
| TF-IDF Similarity | 25% | Semantic closeness of resume to JD |
| Skill Match | 35% | % of required skills found in resume |
| Experience Match | 20% | Years of experience vs. requirement |
| Education Match | 10% | Degree level vs. requirement |
| Keyword Coverage | 10% | JD keyword presence in resume |
| **Total** | **100%** | **Final score out of 100** |

### Decision Thresholds

| Score | Decision |
|---|---|
| 80–100 | STRONG MATCH — Recommend for interview |
| 65–79 | GOOD MATCH — Worth a conversation |
| 50–64 | POTENTIAL — Review carefully |
| 35–49 | WEAK MATCH — Missing key requirements |
| 0–34 | NOT SUITABLE — Significant gaps |

---

## Project Structure

```
ai-resume-screener/
│
├── src/
│   ├── nlp_engine.py         # Core NLP: TF-IDF, skill extraction, similarity
│   ├── scorer.py             # Scoring engine with weighted dimensions
│   └── report_generator.py  # Charts, heatmaps, HTML reports
│
├── api/
│   └── app.py               # FastAPI REST API (screen & batch endpoints)
│
├── notebooks/
│   └── 01_NLP_Analysis.ipynb # EDA, keyword analysis, scoring demo
│
├── data/
│   ├── sample_resumes/      # Sample resume text files
│   └── job_descriptions/    # Sample JD text files
│
├── reports/                 # Generated charts and HTML reports
├── main.py                  # Demo: batch screen 5 candidates
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone & install
```bash
git clone https://github.com/anam-aleena/ai-resume-screener.git
cd ai-resume-screener
pip install -r requirements.txt
```

### 2. Run batch demo (5 sample candidates)
```bash
python main.py
```

### 3. Screen a single resume in Python
```python
from src.scorer import ResumeScorer

jd = """
  Senior Data Scientist — Python, SQL, scikit-learn, XGBoost, NLP,
  AWS, 3+ years experience, Bachelor's required.
"""

resume = """
  Jane Doe | jane@email.com
  4 years Data Scientist — Python, SQL, XGBoost, scikit-learn,
  NLP, BERT, AWS, Pandas, NumPy. M.Sc. Data Science.
"""

screener = ResumeScorer(jd, min_years_experience=3, min_education_score=3)
result   = screener.screen(resume, candidate_name="Jane Doe")

print(f"Score:    {result.scores.total_score}/100")
print(f"Decision: {result.scores.decision_label}")
print(f"Missing:  {result.skill_analysis['_summary']['missing_skills']}")
```

**Output:**
```
Score:    82.4/100
Decision: STRONG MATCH  — Recommend for interview
Missing:  ['docker', 'mlops']
```

### 4. Batch screen & rank
```python
candidates = [
    {"name": "Alice", "text": "5 years ML engineer, Python, TensorFlow, AWS..."},
    {"name": "Bob",   "text": "2 years analyst, Excel, SQL, some Python..."},
    {"name": "Carol", "text": "PhD CS, 3 years NLP research, PyTorch, BERT..."},
]

results = screener.screen_batch(candidates)
for rank, r in enumerate(results, 1):
    print(f"#{rank} {r.candidate.name}: {r.scores.total_score:.1f} — {r.scores.decision}")
```

### 5. Launch the REST API
```bash
uvicorn api.app:app --reload
# → http://localhost:8000/docs   (interactive API docs)
```

---

## REST API

### `POST /screen` — Single resume
```json
{
  "job_description": "Senior ML Engineer — Python, PyTorch, AWS...",
  "resume_text": "Jane Doe — 5 years ML, Python, PyTorch, AWS, Docker...",
  "candidate_name": "Jane Doe",
  "min_years_experience": 3,
  "min_education_score": 3
}
```

**Response:**
```json
{
  "candidate": { "name": "Jane Doe", "email": "jane@email.com", "years_experience": 5 },
  "scores": {
    "tfidf_similarity": 78.4,
    "skill_match": 85.2,
    "experience_match": 100.0,
    "education_match": 80.0,
    "keyword_coverage": 72.0,
    "total_score": 84.1,
    "decision": "strong_match",
    "decision_label": "STRONG MATCH  — Recommend for interview"
  },
  "recommendations": ["Strong semantic alignment with JD.", "Missing: docker, kubernetes"]
}
```

### `POST /screen/batch` — Rank multiple candidates
Returns all candidates sorted by total score, with full analysis per candidate.

---

## Skill Taxonomy (200+ skills across 7 domains)

| Domain | Examples |
|---|---|
| ML & AI | scikit-learn, XGBoost, PyTorch, TensorFlow, BERT, LLMs, RAG |
| Data Science | Pandas, NumPy, EDA, statistical analysis, feature engineering |
| Programming | Python, SQL, Java, R, Scala, C++ |
| Cloud & DevOps | AWS, Azure, GCP, Docker, Kubernetes, MLOps |
| Databases | PostgreSQL, MongoDB, Redis, Elasticsearch |
| Visualization | Matplotlib, Seaborn, Plotly, Tableau, Power BI |
| Soft Skills | communication, agile, documentation, leadership |

---

## Sample Output

```
══════════════════════════════════════════════════════════════
  FINAL RANKING SUMMARY
══════════════════════════════════════════════════════════════
  #1   Divya Nair           91.3   Strong Match
  #2   Priya Sharma         87.6   Strong Match
  #3   Sneha Kulkarni       74.2   Good Match
  #4   Arjun Patel          46.8   Potential
  #5   Rahul Mehta          28.1   Not Suitable
══════════════════════════════════════════════════════════════
```

---

## Tech Stack

| Layer | Tech |
|---|---|
| NLP | TF-IDF (scikit-learn), regex, custom skill taxonomy |
| Scoring | Weighted multi-dimensional scoring engine |
| API | FastAPI + Pydantic |
| Visualization | Matplotlib, Seaborn |
| Language | Python 3.10+ |

---

## Author

**Aleena Anam** — AI/ML Engineer & Data Scientist  
📧 anamaleena0@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/aleena-anam-2056a4368) | [GitHub](https://github.com/anam-aleena)

---

## License

MIT License — free to use, modify, and distribute.
