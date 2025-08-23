from sklearn.metrics.pairwise import cosine_similarity
from extract_skills import process_resume_and_job
from utils.logging_config import get_logger
import numpy as np
from typing import Any, Dict

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


def compute_skill_match(resume_skills: Dict[str, Any], job_skills: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resume_set = set(resume_skills.get("dict_skills", []) + resume_skills.get("fuzzy_skills", []))
        job_set = set(job_skills.get("dict_skills", []) + job_skills.get("fuzzy_skills", []))

        overlap = resume_set & job_set
        missing = job_set - resume_set

        if len(job_set) == 0:
            skill_score = 0.0
        else:
            skill_score = len(overlap) / len(job_set)

        result = {
            "skill_score": round(skill_score, 2),
            "overlap": sorted(list(overlap)),
            "missing": sorted(list(missing)),
        }
        logger.debug("Computed skill match: %s", result)
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
    
# Example usage
if __name__ == "__main__":
    resume = """Experienced Data Scientist skilled in Python, SQL, machine learning, Tableau, 
                and cloud platforms like AWS and Azure."""
    job = """We are hiring a Data Scientist with strong skills in Python, SQL, cloud (AWS), 
             and visualization tools such as Tableau or Power BI."""

    try:
        results = process_resume_and_job(resume, job)
        similarity_score = compute_similarity(results.get("resume_embedding"), results.get("job_embedding"))
        skill_match = compute_skill_match(results.get("resume_skills", {}), results.get("job_skills", {}))

        print("Cosine Similarity Score:", similarity_score)
        print("Skill Match:", skill_match)
    except Exception as exc:
        logger.exception("Example run failed: %s", exc)
