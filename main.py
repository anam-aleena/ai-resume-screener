"""
AI Resume Screener — Main Demo
Author: Aleena Anam | github.com/anam-aleena

Runs a complete batch screening of 5 sample candidates
against a Data Scientist job description.
Generates full reports, charts, and HTML output.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scorer import ResumeScorer
from report_generator import (
    print_screening_report,
    plot_score_breakdown,
    plot_batch_ranking,
    plot_skill_heatmap,
    generate_html_report,
)

# ─── SAMPLE JOB DESCRIPTION ──────────────────────────────────────────────────

JOB_DESCRIPTION = """
Job Title: Senior Data Scientist

We are looking for a Senior Data Scientist to join our growing AI team.
You will design, develop, and deploy machine learning models that solve
real business problems at scale.

Requirements:
- 3+ years of experience in data science or machine learning
- Strong proficiency in Python and SQL
- Experience with scikit-learn, XGBoost, or similar ML frameworks
- Deep knowledge of statistical analysis, data mining, and predictive analytics
- Hands-on experience with NLP and natural language processing
- Familiarity with cloud platforms: AWS, Azure, or GCP
- Experience with data visualization (Matplotlib, Seaborn, Plotly, Tableau)
- Proficiency in Pandas and NumPy for data manipulation
- Experience with model deployment and MLOps practices
- Strong communication skills and ability to document findings clearly
- Bachelor's degree or higher in Computer Science, Statistics, or related field

Nice to have:
- Experience with deep learning frameworks (PyTorch, TensorFlow)
- Knowledge of LLMs, RAG, or generative AI
- Experience with Docker, Kubernetes, or CI/CD pipelines
- Master's or PhD degree preferred
"""

# ─── SAMPLE RESUMES ──────────────────────────────────────────────────────────

RESUMES = [
    {
        "name": "Priya Sharma",
        "text": """
        Priya Sharma | priya.sharma@email.com | +91 9876543210

        SUMMARY
        Senior Data Scientist with 5 years of experience in machine learning,
        NLP, and statistical analysis. PhD in Computer Science from IIT Bombay.

        EXPERIENCE
        Senior Data Scientist — TechCorp (2021–Present)
        - Built NLP pipelines using Python, BERT, and spaCy for text classification
        - Deployed XGBoost and Random Forest models to AWS SageMaker
        - Led a team of 3 data scientists, delivered 8 production ML models

        Data Scientist — StartupAI (2019–2021)
        - Predictive analytics using scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
        - A/B testing, statistical hypothesis testing, feature engineering

        SKILLS
        Python, SQL, scikit-learn, XGBoost, PyTorch, TensorFlow, NLP, BERT,
        Pandas, NumPy, Matplotlib, Seaborn, AWS, Docker, MLOps, Git,
        Statistical Analysis, Data Mining, Feature Engineering, Predictive Analytics
        Communication, Teamwork, Leadership, Documentation

        EDUCATION
        PhD Computer Science — IIT Bombay (2019)
        """
    },
    {
        "name": "Rahul Mehta",
        "text": """
        Rahul Mehta | rahul.mehta@email.com

        EXPERIENCE
        Data Analyst — Analytics Co (2022–Present)
        - Excel, Power BI, SQL for business reporting
        - 2 years of experience in data analysis and visualization
        - Tableau dashboards, data wrangling in Python

        Junior Developer — IT firm (2020–2022)
        - Java backend development, REST APIs

        SKILLS
        SQL, Excel, Tableau, Power BI, Python (basic), Java, Git

        EDUCATION
        Bachelor B.Tech Computer Science — Mumbai University (2020)
        """
    },
    {
        "name": "Sneha Kulkarni",
        "text": """
        Sneha Kulkarni | sneha.k@email.com | +91 9988776655

        PROFESSIONAL SUMMARY
        Data Scientist with 3 years of experience building machine learning
        models for fintech and healthtech. Master's in Data Science.

        EXPERIENCE
        Data Scientist — FinAI Ltd (2021–Present)
        - Fraud detection using XGBoost, Logistic Regression, Random Forest
        - NLP-based document classification with scikit-learn and spaCy
        - Python, Pandas, NumPy, Matplotlib, Seaborn for data analysis
        - Statistical analysis, feature engineering, predictive analytics
        - Model deployment on Azure ML, CI/CD with Jenkins

        SKILLS
        Python, SQL, scikit-learn, XGBoost, NLP, Pandas, NumPy,
        Matplotlib, Seaborn, Azure, Statistical Analysis, Feature Engineering,
        Predictive Analytics, Machine Learning, Data Mining, Git, Documentation,
        Communication, Agile

        EDUCATION
        Master of Science (M.Sc) Data Science — Pune University (2021)
        """
    },
    {
        "name": "Arjun Patel",
        "text": """
        Arjun Patel | arjun.patel@email.com

        SUMMARY
        Fresh graduate with strong foundation in machine learning and Python.
        Built 2 personal ML projects during internship.

        INTERNSHIP
        ML Intern — DataStartup (6 months)
        - Python, scikit-learn, Pandas for classification tasks
        - Exploratory data analysis, data visualization with Matplotlib

        PROJECTS
        - Customer segmentation using K-means clustering
        - House price prediction using Random Forest

        SKILLS
        Python, scikit-learn, Pandas, NumPy, Matplotlib, SQL, Git, Machine Learning

        EDUCATION
        Bachelor B.Sc Computer Science — Gujarat University (2024)
        """
    },
    {
        "name": "Divya Nair",
        "text": """
        Divya Nair | divya.nair@email.com | +91 9123456789

        EXPERIENCE
        Machine Learning Engineer — AI Solutions (2020–Present)
        4 years of experience in machine learning and NLP.

        - Built and deployed LLM-based RAG pipelines using LangChain and OpenAI
        - Deep learning models with PyTorch and TensorFlow for CV and NLP tasks
        - XGBoost, Random Forest, Gradient Boosting for tabular ML
        - Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly
        - NLP, BERT, Transformers, Hugging Face
        - MLOps with Docker, Kubernetes, AWS, GCP, CI/CD pipelines
        - Statistical analysis, A/B testing, hypothesis testing, feature engineering

        SKILLS
        Python, SQL, scikit-learn, XGBoost, PyTorch, TensorFlow, NLP, LLMs, RAG,
        LangChain, HuggingFace, Pandas, NumPy, Matplotlib, Seaborn, Plotly,
        AWS, GCP, Docker, Kubernetes, MLOps, CI/CD, Git, Statistical Analysis,
        Predictive Analytics, Data Mining, Feature Engineering,
        Communication, Leadership, Documentation, Agile, Scrum

        EDUCATION
        Master M.Tech Artificial Intelligence — NIT Trichy (2020)
        """
    },
]


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  AI RESUME SCREENER — BATCH SCREENING DEMO")
    print("  Author: Aleena Anam | github.com/anam-aleena")
    print("=" * 62)
    print(f"\n  Job: Senior Data Scientist")
    print(f"  Candidates: {len(RESUMES)}")
    print(f"  Scoring: TF-IDF + Skill Match + Experience + Education + Keywords\n")

    # Initialize screener
    screener = ResumeScorer(
        job_description=JOB_DESCRIPTION,
        min_years_experience=3,
        min_education_score=3  # Bachelor's minimum
    )

    # Batch screen all candidates
    results = screener.screen_batch(RESUMES)

    # Print individual reports
    for result in results:
        print_screening_report(result)

    # Generate visualizations
    print("[CHARTS] Generating visualizations...")
    plot_batch_ranking(results)
    plot_skill_heatmap(results)

    # Score breakdown for top candidate
    plot_score_breakdown(results[0])

    # HTML report
    generate_html_report(results, jd_title="Senior Data Scientist")

    # Summary
    print("\n" + "=" * 62)
    print("  FINAL RANKING SUMMARY")
    print("=" * 62)
    print(f"  {'Rank':<5} {'Candidate':<20} {'Score':>6}  Decision")
    print(f"  {'-'*55}")
    for i, r in enumerate(results, 1):
        print(f"  #{i:<4} {r.candidate.name:<20} {r.scores.total_score:>5.1f}  "
              f"{r.scores.decision.replace('_', ' ').title()}")

    print(f"\n  Reports saved to /reports/")
    print(f"  Open reports/screening_report.html for full visual report")
    print("=" * 62)
