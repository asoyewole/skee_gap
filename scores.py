from sklearn.metrics.pairwise import cosine_similarity
from utils.logging_config import get_logger
import numpy as np
from typing import Any, Dict
from collections import Counter

logger = get_logger(__name__)


def compute_similarity(resume_emb: Any, job_emb: Any) -> float:
    try:
        if resume_emb is None or job_emb is None:
            logger.warning("One or both embeddings are None")
            return 0.0
        score = float(cosine_similarity([resume_emb], [job_emb])[0][0])
        logger.debug("Computed cosine similarity: %s", score)
        return score
    except Exception as exc:
        logger.exception("Failed to compute similarity: %s", exc)
        return 0.0


def compute_skill_match(resume_skills: Dict[str, Any], job_skills: Dict[str, Any], job_text: str, top_n: int = 20) -> Dict[str, Any]:
    try:
        resume_set = set(resume_skills.get("dict_skills", []) +
                         resume_skills.get("fuzzy_skills", []))
        job_list = job_skills.get("dict_skills", []) + \
            job_skills.get("fuzzy_skills", [])
        job_set = set(job_list)

        overlap = resume_set & job_set
        missing = job_set - resume_set

        if len(job_set) == 0:
            skill_score = 0.0
        else:
            skill_score = len(overlap) / len(job_set)

        # --- Rank missing skills by frequency in job description ---
        job_tokens = [t.lower() for t in job_text.split()]
        freq_counter = Counter(job_tokens)

        # Score missing skills by frequency in job description
        ranked_missing = sorted(
            missing,
            key=lambda skill: freq_counter.get(skill.lower(), 0),
            reverse=True
        )[:top_n]

        result = {
            "skill_score": round(skill_score, 2),
            "overlap": sorted(list(overlap)),
            "missing": ranked_missing,  # now limited & ranked
        }
        logger.debug("Computed skill match with ranking: %s", result)
        return result
    except Exception as exc:
        logger.exception("Failed to compute skill match: %s", exc)
        return {"skill_score": 0.0, "overlap": [], "missing": []}

def interpret_similarity(score: float) -> str:
    try:
        if score >= 0.8:
            return "✅ Excellent match! You should definitely apply for this job."
        elif score >= 0.65:
            return "👍 Good match. You stand a strong chance — applying is recommended."
        elif score >= 0.5:
            return "⚠️ Partial match. Consider improving your resume by adding missing relevant skills."
        else:
            return "❌ Weak match. Your resume and the job description differ significantly. Tailoring your resume is highly recommended."
    except Exception as exc:
        logger.exception("Failed to interpret similarity score: %s", exc)
        return "Score interpretation unavailable."
    