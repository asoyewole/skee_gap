import os
import tempfile

from extract_skills import clean_text, load_skills
from scores import compute_skill_match


def test_clean_text_basic():
    s = "Hello, Email me at test@example.com. Visit https://example.com!"
    out = clean_text(s)
    assert "test@example.com" not in out
    assert "https" not in out
    assert "hello" in out


def test_load_skills_tmp_csv():
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        os.close(fd)
        with open(path, "w", encoding="utf-8") as f:
            f.write("Python, SQL, Tableau\n")

        skills = load_skills(path)
        assert "python" in skills
        assert "sql" in skills
        assert "tableau" in skills
    finally:
        os.remove(path)


def test_compute_skill_match_basic():
    resume = {"dict_skills": ["python"], "fuzzy_skills": []}
    job = {"dict_skills": ["python", "sql"], "fuzzy_skills": []}
    result = compute_skill_match(resume, job)
    assert result["skill_score"] == 0.5
    assert "python" in result["overlap"]
    assert "sql" in result["missing"]
