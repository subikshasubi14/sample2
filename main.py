from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from app.database import engine, Base
import app.models  # Import models to ensure they are registered with Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create all tables (if they don't exist)
    Base.metadata.create_all(bind=engine)
    yield
    # Clean up resources if necessary

from dotenv import load_dotenv
load_dotenv()

from app.services.candidate_analyzer import CandidateAnalysisError, analyze_candidate
from app.services.resume_parser import ResumeParsingError, parse_resume
from app.services.interview_generator import generate_interview_questions
from app.services.candidate_ranker import rank_candidates
from app.routers import jobs, candidates, applications, auth, dashboard, assessments, candidate_assessment

app = FastAPI(
    title="TalentMind AI",
    description="AI-Powered Recruitment & Candidate Evaluation Platform",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(jobs.router)
app.include_router(candidates.router)
app.include_router(applications.router)
app.include_router(auth.router)
app.include_router(dashboard.router)
app.include_router(assessments.router)
app.include_router(candidate_assessment.router)

# CORS – allow the Vite dev-server and any localhost variant used during POC
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeUploadResponse(BaseModel):
    filename: str
    file_type: str
    text_length: int
    extracted_text: str


class CandidateAnalysisResponse(BaseModel):
    filename: str
    candidate_name: str
    ats_score: int
    keyword_skill_match: int
    semantic_match_score: int
    candidate_summary: str
    key_strengths: List[str]
    skill_gaps: List[str]
    experience_assessment: str
    hiring_recommendation: str
    recommendation_reason: str


class InterviewQuestion(BaseModel):
    question_number: int
    skill: str
    question: str
    difficulty: str
    question_type: str
    expected_topics: List[str]


class InterviewQuestionsResponse(BaseModel):
    candidate_name: str
    ats_score: int
    interview_questions: List[InterviewQuestion]


class RankedCandidate(BaseModel):
    rank: int
    filename: str
    candidate_name: str
    ats_score: int
    skill_match_percentage: int
    semantic_similarity: int
    experience_match: str
    recommendation: str
    matched_skills: List[str]
    missing_skills: List[str]


class CandidateRankingResponse(BaseModel):
    job_title: str
    total_candidates: int
    ranked_candidates: List[RankedCandidate]


@app.get("/", tags=["General"])
def root():
    """Root endpoint returning a welcome message."""
    return {
        "message": "Welcome to TalentMind AI – AI-Powered Recruitment & Candidate Evaluation Platform",
        "status": "online",
        "version": "1.0.0",
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint returning the backend service status."""
    return {
        "status": "healthy",
        "service": "TalentMind AI",
        "version": "1.0.0",
    }


@app.post(
    "/resume/upload",
    response_model=ResumeUploadResponse,
    tags=["Resume"],
    summary="Upload and extract text from candidate resume",
    description="Upload a candidate resume in .pdf or .docx format. Extracts text and returns metadata.",
)
async def upload_resume(
    file: UploadFile = File(..., description="Candidate resume file (.pdf or .docx)"),
):
    """
    Accepts a candidate resume in PDF or DOCX format and extracts readable text.
    Validates file extension, empty files, and format corruption.
    """
    try:
        file_bytes = await file.read()
        result = parse_resume(filename=file.filename or "", content=file_bytes)
        return result
    except ResumeParsingError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred while processing the resume: {str(err)}",
        ) from err


@app.post(
    "/candidate/analyze",
    response_model=CandidateAnalysisResponse,
    tags=["Candidate Evaluation"],
    summary="Analyze candidate resume against job description",
    description=(
        "Compares a candidate resume (PDF or DOCX) against a job description, "
        "calculates ATS compatibility, matched/missing skills, experience fit, and recommendation."
    ),
)
async def analyze_candidate_endpoint(
    job_description: str = Form(..., description="Target Job Description text"),
    resume_file: UploadFile = File(..., description="Candidate resume file (.pdf or .docx)"),
):
    """
    Receives job description and resume file, parses text, evaluates skill and experience alignment,
    and returns a transparent ATS score breakdown.
    """
    try:
        file_bytes = await resume_file.read()
        analysis = analyze_candidate(
            job_description=job_description,
            filename=resume_file.filename or "",
            file_bytes=file_bytes,
        )
        analysis["filename"] = resume_file.filename or ""
        return analysis
    except CandidateAnalysisError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred during candidate analysis: {str(err)}",
        ) from err


@app.post(
    "/candidate/interview-questions",
    response_model=InterviewQuestionsResponse,
    tags=["Candidate Evaluation"],
    summary="Generate personalized technical interview questions",
    description="Analyzes the candidate's resume and generates interview questions based on skills.",
)
async def generate_interview_questions_endpoint(
    job_description: str = Form(..., description="Target Job Description text"),
    resume_file: UploadFile = File(..., description="Candidate resume file (.pdf or .docx)"),
):
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    try:
        file_bytes = await resume_file.read()
        analysis = analyze_candidate(
            job_description=job_description,
            filename=resume_file.filename or "",
            file_bytes=file_bytes,
        )

        questions_result = generate_interview_questions(analysis)
        return questions_result

    except CandidateAnalysisError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected internal error occurred: {str(err)}",
        ) from err


@app.post(
    "/candidates/rank",
    response_model=CandidateRankingResponse,
    tags=["Candidate Evaluation"],
    summary="Rank multiple candidates against a single job description",
)
async def rank_candidates_endpoint(
    job_description: str = Form(..., description="Target Job Description text"),
    resume_files: List[UploadFile] = File(..., description="Multiple candidate resume files (.pdf or .docx)")
):
    if not job_description or not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    if not resume_files or len(resume_files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 candidates must be provided for ranking.")

    if len(resume_files) > 10:
        raise HTTPException(status_code=400, detail="Maximum of 10 candidates allowed for this POC.")

    profiles = []

    # Try parsing and analyzing each resume
    for file in resume_files:
        try:
            file_bytes = await file.read()
            analysis = analyze_candidate(
                job_description=job_description,
                filename=file.filename or "",
                file_bytes=file_bytes,
            )
            analysis["filename"] = file.filename or ""
            profiles.append(analysis)
        except (ResumeParsingError, CandidateAnalysisError) as err:
            raise HTTPException(status_code=400, detail=str(err)) from err
        except Exception as err:
            raise HTTPException(
                status_code=500,
                detail=f"An unexpected internal error occurred while analyzing {file.filename}: {str(err)}",
            ) from err

    # Rank the candidates
    ranked_list = rank_candidates(profiles)

    # Derive a simple job title from the first line of the JD
    job_title = job_description.split('\n')[0][:50].strip() if '\n' in job_description else "Target Role"

    return {
        "job_title": job_title,
        "total_candidates": len(ranked_list),
        "ranked_candidates": ranked_list
    }