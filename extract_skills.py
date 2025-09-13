from __future__ import annotations

import re
from typing import Any, Dict, List

import os
import numpy as np
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS
from rapidfuzz import fuzz, process
import streamlit as st

from utils.logging_config import get_logger

logger = get_logger(__name__)


@st.cache_resource
def load_nlp():
    import spacy
    nlp = spacy.load("en_core_web_sm")
    # Optimize: disable unused pipes
    nlp.select_pipes(disable=["ner", "senter"])
    logger.debug("spaCy model loaded with optimized pipes")
    return nlp


@st.cache_resource
def load_embedder():
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    logger.debug("SentenceTransformer model loaded")
    return embedder


@st.cache_data
def load_skills_set(csv_path: str = "skills.csv") -> List[str]:
    """Read skills from a CSV file and return a deduplicated, normalized list."""
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
                if clean_skill and len(clean_skill) > 2 and clean_skill not in STOP_WORDS:
                    skills_set.add(clean_skill)

    skills = sorted(skills_set)
    logger.debug("Loaded %d skills", len(skills))
    return skills


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text[:10000]  # Truncate for memory safety
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\+?\d[\d\s\-]{7,}\d", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_skills(skills_set: List[str], text: str, fuzzy_threshold: int = 88) -> Dict[str, Any]:
    if not text:
        return {"dict_skills": [], "fuzzy_skills": []}

    nlp = load_nlp()
    doc = nlp(clean_text(text))

    candidates = set([t.text.lower()
                     for t in doc if t.is_alpha and t.text.lower() not in STOP_WORDS])
    candidates.update([chunk.text.strip().lower()
                      for chunk in doc.noun_chunks if 2 <= len(chunk.text.strip()) <= 40])
    candidates = list(candidates)[:200]  # Limit for memory/efficiency

    dict_matches = set()
    fuzzy_matches = set()

    for cand in candidates:
        if cand in skills_set:
            dict_matches.add(cand)
            continue

        try:
            res = process.extractOne(
                cand, skills_set, scorer=fuzz.token_sort_ratio)
            if res:
                matched_skill, score, _ = res
                if score >= fuzzy_threshold:
                    fuzzy_matches.add(matched_skill)
        except Exception:
            logger.debug("Fuzzy match error for candidate: %s", cand)

    return {"dict_skills": sorted(dict_matches), "fuzzy_skills": sorted(fuzzy_matches)}


def get_embeddings(text: str, embedder):
    """Return embedding for text, with caching."""
    import hashlib
    cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "embeddings")
    os.makedirs(cache_dir, exist_ok=True)

    cleaned = clean_text(text)
    key = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    cache_path = os.path.join(cache_dir, f"{key}.npy")

    try:
        if os.path.exists(cache_path):
            logger.debug("Loading embedding from cache: %s", cache_path)
            return np.load(cache_path)
    except Exception:
        logger.debug("Failed to load embedding cache at %s", cache_path)

    emb = embedder.encode([cleaned])[0]

    try:
        np.save(cache_path, emb)
        logger.debug("Saved embedding to cache: %s", cache_path)
    except Exception:
        logger.debug("Failed to save embedding cache at %s", cache_path)

    return emb


def process_resume_and_job_wrapper(resume_text: str, job_text: str, nlp, embedder, skills_set):
    """Wrapper for picklability in multiprocessing."""
    resume_clean = clean_text(resume_text)
    job_clean = clean_text(job_text)

    resume_skills = extract_skills(skills_set, resume_clean)
    job_skills = extract_skills(skills_set, job_clean)

    resume_emb = get_embeddings(resume_clean, embedder)
    job_emb = get_embeddings(job_clean, embedder)

    return {
        "resume_clean": resume_clean,
        "job_clean": job_clean,
        "resume_skills": resume_skills,
        "job_skills": job_skills,
        "resume_embedding": resume_emb,
        "job_embedding": job_emb,
    }
