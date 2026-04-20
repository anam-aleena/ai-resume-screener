"""
AI Resume Screener — REST API
Author: Aleena Anam | github.com/anam-aleena

FastAPI endpoints for integrating the resume screener
into any application or workflow.

Run:
    pip install fastapi uvicorn
    uvicorn api.app:app --reload
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

from scorer import ResumeScorer


app = FastAPI(
    title="AI Resume Screener API",
    description=(
        "NLP-powered resume screening API. "
        "Screen single or batch resumes against a job description. "
        "Returns ranked candidates with detailed scoring and recommendations.\n\n"
        "Author: Aleena Anam | github.com/anam-aleena"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── REQUEST / RESPONSE MODELS ───────────────────────────────────────────────

class ScreenRequest(BaseModel):
    job_description: str
    resume_text: str
    candidate_name: Optional[str] = "Unknown"
    min_years_experience: Optional[int] = 0
    min_education_score: Optional[int] = 0

    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "We need a Data Scientist with Python, ML, XGBoost, 3+ years experience...",
                "resume_text": "John Doe — Data Scientist with 4 years experience in Python, scikit-learn, XGBoost...",
                "candidate_name": "John Doe",
                "min_years_experience": 3,
                "min_education_score": 3
            }
        }


class BatchCandidate(BaseModel):
    name: str
    text: str


class BatchScreenRequest(BaseModel):
    job_description: str
    candidates: List[BatchCandidate]
    min_years_experience: Optional[int] = 0
    min_education_score: Optional[int] = 0

    class Config:
        json_schema_extra = {
            "example": {
                "job_description": "We need a Senior ML Engineer with Python, TensorFlow, AWS...",
                "candidates": [
                    {"name": "Alice Smith", "text": "5 years ML engineer, Python, TensorFlow, AWS..."},
                    {"name": "Bob Jones",   "text": "1 year junior developer, Java, basic Python..."},
                ],
                "min_years_experience": 3,
            }
        }


# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    return {
        "service": "AI Resume Screener API",
        "version": "1.0.0",
        "author": "Aleena Anam",
        "github": "github.com/anam-aleena/ai-resume-screener",
        "endpoints": {
            "POST /screen":       "Screen a single resume",
            "POST /screen/batch": "Screen multiple resumes and rank them",
            "GET  /health":       "Health check",
            "GET  /docs":         "Interactive API docs",
        }
    }


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "service": "ai-resume-screener"}


@app.post("/screen", tags=["Screening"])
def screen_single(request: ScreenRequest):
    """
    Screen a single resume against a job description.
    Returns detailed scores, skill analysis, and recommendations.
    """
    if len(request.job_description.strip()) < 50:
        raise HTTPException(status_code=400,
                            detail="Job description too short (min 50 characters).")
    if len(request.resume_text.strip()) < 30:
        raise HTTPException(status_code=400,
                            detail="Resume text too short (min 30 characters).")

    screener = ResumeScorer(
        job_description=request.job_description,
        min_years_experience=request.min_years_experience,
        min_education_score=request.min_education_score,
    )
    result = screener.screen(request.resume_text, request.candidate_name)
    return screener.to_dict(result)


@app.post("/screen/batch", tags=["Screening"])
def screen_batch(request: BatchScreenRequest):
    """
    Screen multiple resumes at once.
    Returns candidates ranked by total score (highest first).
    """
    if not request.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided.")
    if len(request.candidates) > 50:
        raise HTTPException(status_code=400,
                            detail="Maximum 50 candidates per batch request.")

    screener = ResumeScorer(
        job_description=request.job_description,
        min_years_experience=request.min_years_experience,
        min_education_score=request.min_education_score,
    )

    resumes = [{"name": c.name, "text": c.text} for c in request.candidates]
    results = screener.screen_batch(resumes)

    ranked = []
    for rank, result in enumerate(results, 1):
        data = screener.to_dict(result)
        data["rank"] = rank
        ranked.append(data)

    return {
        "total_screened": len(ranked),
        "top_candidate":  ranked[0]["candidate"]["name"] if ranked else None,
        "candidates":     ranked,
    }
