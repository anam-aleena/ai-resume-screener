"""
AI Resume Screener — Resume Scorer
Author: Aleena Anam | github.com/anam-aleena

Orchestrates all scoring components into a final weighted score.
Produces structured output ready for ranking or API response.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime

from nlp_engine import (
    extract_skills, extract_experience_years, extract_education_level,
    extract_email, extract_phone, compute_tfidf_similarity,
    compute_skill_match_score, keyword_overlap, clean_text
)


# ─── SCORING WEIGHTS ─────────────────────────────────────────────────────────
# Total = 100 points

WEIGHTS = {
    "tfidf_similarity":   0.25,   # 25 pts — overall semantic match
    "skill_match":        0.35,   # 35 pts — keyword/skill overlap
    "experience_match":   0.20,   # 20 pts — years of experience
    "education_match":    0.10,   # 10 pts — degree level
    "keyword_coverage":   0.10,   # 10 pts — JD keyword presence
}

DECISION_THRESHOLDS = {
    "strong_match":   80,
    "good_match":     65,
    "potential":      50,
    "weak_match":     35,
    "not_suitable":    0,
}

DECISION_LABELS = {
    "strong_match":  "STRONG MATCH  — Recommend for interview",
    "good_match":    "GOOD MATCH    — Worth a conversation",
    "potential":     "POTENTIAL     — Review carefully",
    "weak_match":    "WEAK MATCH    — Missing key requirements",
    "not_suitable":  "NOT SUITABLE  — Significant gaps",
}


# ─── DATA STRUCTURES ─────────────────────────────────────────────────────────

@dataclass
class CandidateProfile:
    name:            str
    email:           Optional[str]
    phone:           Optional[str]
    education:       str
    education_score: int
    years_experience: Optional[int]
    skills_found:    dict
    raw_text_length: int


@dataclass
class ScoreBreakdown:
    tfidf_similarity:   float
    skill_match:        float
    experience_match:   float
    education_match:    float
    keyword_coverage:   float
    total_score:        float
    decision:           str
    decision_label:     str


@dataclass
class ScreeningResult:
    candidate:      CandidateProfile
    scores:         ScoreBreakdown
    skill_analysis: dict
    keyword_gaps:   dict
    recommendations: list
    screened_at:    str


# ─── SCORER ──────────────────────────────────────────────────────────────────

class ResumeScorer:
    """
    Full resume screening pipeline.

    Usage:
        scorer = ResumeScorer(job_description_text, min_years=2, min_education=3)
        result = scorer.screen(resume_text, candidate_name="Jane Doe")
        print(result.scores.total_score)
    """

    def __init__(self,
                 job_description: str,
                 min_years_experience: int = 0,
                 min_education_score: int = 0,
                 weights: dict = None):
        self.jd = job_description
        self.jd_clean = clean_text(job_description)
        self.jd_skills = extract_skills(job_description)
        self.min_years = min_years_experience
        self.min_edu_score = min_education_score
        self.weights = weights or WEIGHTS

    # ── individual scorers ──────────────────────────────────────────────────

    def _score_tfidf(self, resume_text: str) -> float:
        raw = compute_tfidf_similarity(resume_text, self.jd)
        # Scale: cosine similarity 0–0.5+ maps to 0–100
        return min(raw * 200, 100.0)

    def _score_skills(self, resume_skills: dict) -> tuple:
        analysis = compute_skill_match_score(resume_skills, self.jd_skills)
        overall  = analysis.get("_summary", {}).get("overall_skill_match", 0)
        return overall * 100, analysis

    def _score_experience(self, resume_text: str) -> float:
        years = extract_experience_years(resume_text)
        if years is None:
            return 40.0   # neutral — can't penalize if not stated
        if self.min_years == 0:
            return min(years * 15, 100.0)
        ratio = years / self.min_years
        if ratio >= 1.5:   return 100.0
        if ratio >= 1.0:   return 85.0
        if ratio >= 0.75:  return 60.0
        if ratio >= 0.5:   return 35.0
        return 15.0

    def _score_education(self, edu_score: int) -> float:
        if self.min_edu_score == 0:
            return min(edu_score * 20, 100.0)
        if edu_score >= self.min_edu_score + 1: return 100.0
        if edu_score == self.min_edu_score:     return 80.0
        if edu_score == self.min_edu_score - 1: return 50.0
        return 20.0

    def _score_keywords(self, resume_text: str) -> tuple:
        kw = keyword_overlap(resume_text, self.jd, top_n=20)
        return kw["coverage"] * 100, kw

    # ── recommendation engine ───────────────────────────────────────────────

    def _generate_recommendations(self,
                                   scores: ScoreBreakdown,
                                   skill_analysis: dict,
                                   keyword_gaps: dict,
                                   years_exp: Optional[int]) -> list:
        recs = []
        summary = skill_analysis.get("_summary", {})
        missing = summary.get("missing_skills", [])

        if scores.skill_match < 50:
            top_missing = missing[:5]
            if top_missing:
                recs.append(f"Critical skill gaps: {', '.join(top_missing)}. "
                            f"Candidate should upskill in these areas.")

        if scores.experience_match < 50 and self.min_years > 0:
            recs.append(f"Experience may be below requirement "
                        f"(minimum {self.min_years} years expected).")

        if scores.education_match < 50 and self.min_edu_score > 0:
            recs.append("Educational qualification may not meet the minimum requirement.")

        absent_kw = keyword_gaps.get("absent", [])[:5]
        if absent_kw:
            recs.append(f"Resume is missing key JD terms: {', '.join(absent_kw)}. "
                        f"Candidate should tailor resume language.")

        if scores.tfidf_similarity > 70:
            recs.append("Strong overall semantic alignment with job description.")

        bonus = skill_analysis.get("_summary", {}).get("bonus_skills", [])[:4]
        if bonus:
            recs.append(f"Candidate brings additional skills not required: "
                        f"{', '.join(bonus)} — potential added value.")

        if not recs:
            recs.append("Candidate profile aligns well with the job requirements.")

        return recs

    # ── main screen ─────────────────────────────────────────────────────────

    def screen(self, resume_text: str, candidate_name: str = "Unknown") -> ScreeningResult:
        """
        Full screening pipeline for one resume.
        Returns a ScreeningResult with all scores, analysis, and recommendations.
        """
        # Extract candidate info
        email   = extract_email(resume_text)
        phone   = extract_phone(resume_text)
        edu_label, edu_score = extract_education_level(resume_text)
        years_exp = extract_experience_years(resume_text)
        skills  = extract_skills(resume_text)

        candidate = CandidateProfile(
            name=candidate_name,
            email=email,
            phone=phone,
            education=edu_label,
            education_score=edu_score,
            years_experience=years_exp,
            skills_found=skills,
            raw_text_length=len(resume_text),
        )

        # Score each dimension
        tfidf_s    = self._score_tfidf(resume_text)
        skill_s, skill_analysis = self._score_skills(skills)
        exp_s      = self._score_experience(resume_text)
        edu_s      = self._score_education(edu_score)
        kw_s, kw_gaps = self._score_keywords(resume_text)

        # Weighted total
        total = (
            self.weights["tfidf_similarity"] * tfidf_s +
            self.weights["skill_match"]       * skill_s +
            self.weights["experience_match"]  * exp_s   +
            self.weights["education_match"]   * edu_s   +
            self.weights["keyword_coverage"]  * kw_s
        )
        total = round(min(total, 100.0), 2)

        # Decision
        decision = "not_suitable"
        for key, threshold in sorted(DECISION_THRESHOLDS.items(),
                                     key=lambda x: x[1], reverse=True):
            if total >= threshold:
                decision = key
                break

        scores = ScoreBreakdown(
            tfidf_similarity  = round(tfidf_s, 2),
            skill_match       = round(skill_s, 2),
            experience_match  = round(exp_s, 2),
            education_match   = round(edu_s, 2),
            keyword_coverage  = round(kw_s, 2),
            total_score       = total,
            decision          = decision,
            decision_label    = DECISION_LABELS[decision],
        )

        recommendations = self._generate_recommendations(
            scores, skill_analysis, kw_gaps, years_exp
        )

        return ScreeningResult(
            candidate       = candidate,
            scores          = scores,
            skill_analysis  = skill_analysis,
            keyword_gaps    = kw_gaps,
            recommendations = recommendations,
            screened_at     = datetime.utcnow().isoformat() + "Z",
        )

    def screen_batch(self, resumes: list) -> list:
        """
        Screen multiple resumes. Each item: {"name": str, "text": str}
        Returns sorted list of ScreeningResult by total_score descending.
        """
        results = []
        for item in resumes:
            result = self.screen(item["text"], candidate_name=item.get("name", "Unknown"))
            results.append(result)
        results.sort(key=lambda r: r.scores.total_score, reverse=True)
        return results

    def to_dict(self, result: ScreeningResult) -> dict:
        """Serialize ScreeningResult to JSON-friendly dict."""
        return {
            "candidate": asdict(result.candidate),
            "scores":    asdict(result.scores),
            "skill_analysis":  result.skill_analysis,
            "keyword_gaps":    result.keyword_gaps,
            "recommendations": result.recommendations,
            "screened_at":     result.screened_at,
        }

    def to_json(self, result: ScreeningResult, indent: int = 2) -> str:
        return json.dumps(self.to_dict(result), indent=indent)
