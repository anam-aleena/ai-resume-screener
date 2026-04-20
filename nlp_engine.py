"""
AI Resume Screener — NLP Engine
Author: Aleena Anam | github.com/anam-aleena

Core NLP processing: text extraction, keyword analysis,
skill matching, semantic similarity scoring.
"""

import re
import string
from collections import Counter
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ─── SKILL TAXONOMY ──────────────────────────────────────────────────────────

SKILL_TAXONOMY = {
    "programming_languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
        "go", "rust", "kotlin", "swift", "sql", "bash", "ruby", "php"
    ],
    "ml_ai": [
        "machine learning", "deep learning", "nlp", "natural language processing",
        "computer vision", "reinforcement learning", "neural network", "transformer",
        "bert", "gpt", "llm", "large language model", "rag", "vector search",
        "scikit-learn", "sklearn", "tensorflow", "keras", "pytorch", "xgboost",
        "random forest", "gradient boosting", "logistic regression", "svm",
        "clustering", "classification", "regression", "feature engineering",
        "model deployment", "mlops", "huggingface", "langchain", "openai"
    ],
    "data_science": [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy",
        "statistics", "statistical analysis", "data mining", "data wrangling",
        "exploratory data analysis", "eda", "hypothesis testing", "a/b testing",
        "predictive analytics", "data visualization", "feature selection",
        "dimensionality reduction", "pca", "time series"
    ],
    "cloud_devops": [
        "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "ci/cd",
        "jenkins", "terraform", "ansible", "devops", "mlops", "airflow",
        "spark", "hadoop", "databricks", "snowflake", "bigquery", "redshift"
    ],
    "databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "sqlite",
        "cassandra", "dynamodb", "neo4j", "oracle", "sql server"
    ],
    "tools": [
        "git", "github", "jupyter", "vs code", "linux", "rest api", "fastapi",
        "flask", "django", "streamlit", "tableau", "power bi", "excel", "jira"
    ],
    "soft_skills": [
        "communication", "teamwork", "leadership", "problem solving",
        "analytical", "collaboration", "agile", "scrum", "project management",
        "documentation", "research", "critical thinking", "adaptability"
    ]
}

# Flatten for quick lookup
ALL_SKILLS = {
    skill: category
    for category, skills in SKILL_TAXONOMY.items()
    for skill in skills
}

EXPERIENCE_PATTERNS = [
    r"(\d+)\+?\s*years?\s+of\s+experience",
    r"(\d+)\+?\s*years?\s+experience",
    r"experience\s+of\s+(\d+)\+?\s*years?",
    r"(\d+)\+?\s*yrs?\s+experience",
]

EDUCATION_KEYWORDS = {
    "phd": 5, "ph.d": 5, "doctorate": 5,
    "master": 4, "m.s": 4, "m.sc": 4, "mba": 4, "m.tech": 4,
    "bachelor": 3, "b.s": 3, "b.sc": 3, "b.tech": 3, "b.e": 3,
    "associate": 2, "diploma": 1, "certification": 1
}

SECTION_HEADERS = [
    "experience", "education", "skills", "projects", "certifications",
    "summary", "objective", "achievements", "publications", "awards"
]


# ─── TEXT PREPROCESSING ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, normalize whitespace."""
    text = text.lower()
    text = re.sub(r'[^\w\s\+\#]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_skills(text: str) -> dict:
    """
    Extract skills from text and categorize them.
    Returns dict: {category: [found_skills]}
    """
    text_lower = text.lower()
    found = {cat: [] for cat in SKILL_TAXONOMY}

    for skill, category in ALL_SKILLS.items():
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            if skill not in found[category]:
                found[category].append(skill)

    return {k: v for k, v in found.items() if v}


def extract_experience_years(text: str) -> Optional[int]:
    """Extract years of experience mentioned in text."""
    text_lower = text.lower()
    years_found = []
    for pattern in EXPERIENCE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        years_found.extend([int(m) for m in matches])
    return max(years_found) if years_found else None


def extract_education_level(text: str) -> tuple:
    """
    Returns (education_label, education_score 1-5).
    """
    text_lower = text.lower()
    best_score = 0
    best_label = "Not specified"
    for keyword, score in EDUCATION_KEYWORDS.items():
        if keyword in text_lower and score > best_score:
            best_score = score
            best_label = keyword.upper()
    return best_label, best_score


def extract_email(text: str) -> Optional[str]:
    match = re.search(r'[\w.+-]+@[\w-]+\.[a-z]{2,}', text, re.I)
    return match.group(0) if match else None


def extract_phone(text: str) -> Optional[str]:
    match = re.search(r'[\+\(]?[\d\s\-\(\)]{9,15}', text)
    return match.group(0).strip() if match else None


# ─── SIMILARITY SCORING ──────────────────────────────────────────────────────

def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    """Cosine similarity between resume and job description using TF-IDF."""
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )
    try:
        tfidf_matrix = vectorizer.fit_transform([clean_text(resume_text),
                                                  clean_text(jd_text)])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(similarity), 4)
    except Exception:
        return 0.0


def compute_skill_match_score(resume_skills: dict, jd_skills: dict) -> dict:
    """
    Compare skills found in resume vs job description.
    Returns match scores per category and overall.
    """
    results = {}
    all_matched, all_missing, all_extra = [], [], []

    for category in SKILL_TAXONOMY:
        r_skills = set(resume_skills.get(category, []))
        j_skills = set(jd_skills.get(category, []))

        if not j_skills:
            continue

        matched = r_skills & j_skills
        missing = j_skills - r_skills
        extra   = r_skills - j_skills

        score = len(matched) / len(j_skills) if j_skills else 0.0

        results[category] = {
            "matched": sorted(matched),
            "missing": sorted(missing),
            "extra":   sorted(extra),
            "score":   round(score, 3),
            "required": len(j_skills),
            "found":    len(matched),
        }
        all_matched.extend(matched)
        all_missing.extend(missing)
        all_extra.extend(extra)

    overall = (len(all_matched) / (len(all_matched) + len(all_missing))
               if (all_matched or all_missing) else 0.0)

    results["_summary"] = {
        "overall_skill_match": round(overall, 3),
        "total_matched": len(all_matched),
        "total_missing": len(all_missing),
        "matched_skills": sorted(all_matched),
        "missing_skills": sorted(all_missing),
        "bonus_skills":   sorted(all_extra),
    }
    return results


# ─── KEYWORD DENSITY ─────────────────────────────────────────────────────────

def keyword_frequency(text: str, top_n: int = 20) -> list:
    """Top N most frequent meaningful words in text."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = clean_text(text).split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 2]
    return Counter(words).most_common(top_n)


def keyword_overlap(resume_text: str, jd_text: str, top_n: int = 15) -> dict:
    """Keywords in JD that are present/absent in resume."""
    jd_keywords = {w for w, _ in keyword_frequency(jd_text, top_n * 2)}
    resume_words = set(clean_text(resume_text).split())
    present = sorted(jd_keywords & resume_words)
    absent  = sorted(jd_keywords - resume_words)
    return {"present": present, "absent": absent,
            "coverage": round(len(present) / len(jd_keywords), 3) if jd_keywords else 0}
