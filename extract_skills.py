"""Skill extraction and embedding helpers.

This module focuses on small, testable functions with robust logging and
defensive checks. Heavy models are lazy-loaded to avoid import-time overhead.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import os
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from rapidfuzz import fuzz, process

from utils.logging_config import get_logger

logger = get_logger(__name__)


# Lazy-loaded resources
_nlp = None
_embedder = None


def _load_models() -> None:
    """Load heavy models (spaCy and SentenceTransformer) on demand."""
    global _nlp, _embedder
    if _nlp is None:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm")
            logger.debug("spaCy model loaded")
        except Exception as exc:
            logger.exception("Failed to load spaCy model: %s", exc)
            raise

    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.debug("SentenceTransformer model loaded")
        except Exception as exc:
            logger.exception("Failed to load SentenceTransformer: %s", exc)
            raise


def load_skills(csv_path: str = "skills.csv") -> List[str]:
    """Read skills from a CSV file and return a deduplicated, normalized list.

    Returns an empty list on failure.
    """
    try:
        df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)
    except Exception as exc:
        logger.exception("Failed to read skills CSV '%s': %s", csv_path, exc)
        return []

    skills_set = set()
    for row in df.values.flatten():
        if isinstance(row, str):
            for skill in row.split(","):
                clean_skill = skill.strip().lower()
                if (
                    clean_skill
                    and len(clean_skill) > 2
                    and clean_skill not in STOP_WORDS
                ):
                    skills_set.add(clean_skill)

    skills = sorted(skills_set)
    logger.debug("Loaded %d skills", len(skills))
    return skills


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(skills_set: List[str], text: str, fuzzy_threshold: int = 88) -> Dict[str, Any]:
    """Extract skills from text using exact and fuzzy matching.

    Returns a dict with keys 'dict_skills' and 'fuzzy_skills'.
    """
    if not text:
        return {"dict_skills": [], "fuzzy_skills": []}

    if _nlp is None:
        _load_models()

    doc = _nlp(clean_text(text))

    candidates = set([t.text.lower() for t in doc if t.is_alpha and t.text.lower() not in STOP_WORDS])
    candidates.update([chunk.text.strip().lower() for chunk in doc.noun_chunks if 2 <= len(chunk.text.strip()) <= 40])

    dict_matches = set()
    fuzzy_matches = set()

    for cand in candidates:
        if cand in skills_set:
            dict_matches.add(cand)
            continue

        try:
            res = process.extractOne(cand, skills_set, scorer=fuzz.token_sort_ratio)
            if res:
                matched_skill, score, _ = res
                if score >= fuzzy_threshold:
                    fuzzy_matches.add(matched_skill)
        except Exception:
            logger.debug("Fuzzy match error for candidate: %s", cand)

    return {"dict_skills": sorted(dict_matches), "fuzzy_skills": sorted(fuzzy_matches)}


def get_embeddings(text: str):
    """Return embedding for `text`, using on-disk cache to avoid recomputation.

    Cache is stored under `.cache/embeddings/` as SHA256(text).npy
    """
    import hashlib
    import json
    import numpy as _np

    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    cleaned = clean_text(text)
    key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{key}.npy")

    # Load from cache if present
    try:
        if os.path.exists(cache_path):
            logger.debug("Loading embedding from cache: %s", cache_path)
            return _np.load(cache_path)
    except Exception:
        logger.debug("Failed to load embedding cache at %s", cache_path)

    # Compute and save
    if _embedder is None:
        _load_models()

    emb = _embedder.encode([cleaned])[0]

    try:
        _np.save(cache_path, emb)
        logger.debug("Saved embedding to cache: %s", cache_path)
    except Exception:
        logger.debug("Failed to save embedding cache at %s", cache_path)

    return emb


def process_resume_and_job(resume_text: str, job_text: str):
    try:
        resume_clean = clean_text(resume_text)
        job_clean = clean_text(job_text)
        skills_set = load_skills()

        resume_skills = extract_skills(skills_set, resume_clean)
        job_skills = extract_skills(skills_set, job_clean)

        resume_emb = get_embeddings(resume_clean)
        job_emb = get_embeddings(job_clean)

        return {
            "resume_clean": resume_clean,
            "job_clean": job_clean,
            "resume_skills": resume_skills,
            "job_skills": job_skills,
            "resume_embedding": resume_emb,
            "job_embedding": job_emb,
        }
    except Exception as exc:
        logger.exception("process_resume_and_job failed: %s", exc)
        raise

# extract_skills.py

import re
import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
from spacy.lang.en.stop_words import STOP_WORDS
from rapidfuzz import fuzz, process  # faster fuzzy matching

# Load spaCy English model
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer model for embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load skills dictionary ---
def load_skills(csv_path="skills.csv"):
    df = pd.read_csv(csv_path, header=None, dtype=str, low_memory=False)
    
    skills_set = set()
    for row in df.values.flatten():
        if isinstance(row, str):
            for skill in row.split(","):
                clean_skill = skill.strip().lower()
                if (
                    clean_skill 
                    and len(clean_skill) > 2  # min length
                    and clean_skill not in STOP_WORDS  # skip stopwords
                ):
                    skills_set.add(clean_skill)
    return list(skills_set)


# ---------- TEXT CLEANING ----------
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Lowercase
    text = text.lower()
    # Remove emails, URLs, phone numbers
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " ", text)
    # Remove special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------- SKILL EXTRACTION ----------
def extract_skills(skills_set, text: str, fuzzy_threshold=80):
    text = clean_text(text)
    doc = nlp(text)

    # Candidate tokens/noun chunks
    candidates = set([t.text.lower() for t in doc if t.is_alpha and t.text not in STOP_WORDS])
    candidates.update(
        [chunk.text.strip().lower() for chunk in doc.noun_chunks if 2 <= len(chunk.text.strip()) <= 40]
    )

    dict_matches = []
    fuzzy_matches = []

    for cand in candidates:
        # Exact dictionary hit
        if cand in skills_set:
            dict_matches.append(cand)
        else:
            # Fuzzy match against known skills
            match, score, _ = process.extractOne(
                cand, skills_set, scorer=fuzz.token_sort_ratio
            )
            if score >= fuzzy_threshold:
                fuzzy_matches.append(match)

    return {
        "dict_skills": sorted(set(dict_matches)),
        "fuzzy_skills": sorted(set(fuzzy_matches)),
    }

# ---------- EMBEDDING GENERATION ----------
def get_embeddings(text: str):
    return embedder.encode([clean_text(text)])[0]

# ---------- MAIN PIPELINE ----------
def process_resume_and_job(resume_text: str, job_text: str):
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)
    skills_set = load_skills()
    resume_skills = extract_skills(skills_set, resume_clean)
    job_skills = extract_skills(skills_set, job_clean)

    resume_emb = get_embeddings(resume_clean)
    job_emb = get_embeddings(job_clean)

    return {
        "resume_clean": resume_clean,
        "job_clean": job_clean,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "resume_embedding": resume_emb,
        "job_embedding": job_emb
    }

# ---------- Example Run ----------
if __name__ == "__main__":
    resume = """Experienced Data Scientist skilled in Python, SQL, machine learning, Tableau, 
                and cloud platforms like AWS and Azure."""
    job = """We are hiring a Data Scientist with strong skills in Python, SQL, cloud (AWS), 
             and visualization tools such as Tableau or Power BI."""

    results = process_resume_and_job(resume, job)
    print("Resume Skills:", results["resume_skills"])
    print("Job Skills:", results["job_skills"])
