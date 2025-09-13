import streamlit as st
from utils.extract_text import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt
from extract_skills import process_resume_and_job_wrapper, load_nlp, load_embedder, load_skills_set
from scores import compute_similarity, compute_skill_match, interpret_similarity
import gc
import psutil
from concurrent.futures import ProcessPoolExecutor

st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📊 Resume vs Job Description Analyzer")

MIN_CHAR_COUNT = 300

# --- Layout: 2 columns ---
col1, col2 = st.columns(2)

# Resume Input (left)
with col1:
    st.subheader("Resume")
    resume_text = ""
    resume_file = st.file_uploader(
        "Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="resume")
    if resume_file:
        if resume_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(resume_file)
        elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            resume_text = extract_text_from_docx(resume_file)
        elif resume_file.type == "text/plain":
            resume_text = extract_text_from_txt(resume_file)

    resume_text_paste = st.text_area(
        "Or paste resume text here", height=200, key="resume_paste")
    if resume_text_paste.strip():
        resume_text = resume_text_paste

    if resume_text:
        st.write(f"**Characters:** {len(resume_text)}")
        if len(resume_text) < MIN_CHAR_COUNT:
            st.warning(
                "⚠️ Resume text seems too short, extraction may have failed.")

# Job Input (right)
with col2:
    st.subheader("Job Description")
    job_text = ""
    job_file = st.file_uploader(
        "Upload Job Description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"], key="job")
    if job_file:
        if job_file.type == "application/pdf":
            job_text = extract_text_from_pdf(job_file)
        elif job_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            job_text = extract_text_from_docx(job_file)
        elif job_file.type == "text/plain":
            job_text = extract_text_from_txt(job_file)

    job_text_paste = st.text_area(
        "Or paste job description here", height=200, key="job_paste")
    if job_text_paste.strip():
        job_text = job_text_paste

    if job_text:
        st.write(f"**Characters:** {len(job_text)}")
        if len(job_text) < MIN_CHAR_COUNT:
            st.warning(
                "⚠️ Job description seems too short, extraction may have failed.")

# --- Centered Analyze Button ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        analyze_button = st.button("🔍 Compare")

# Initialize session state for analyzing flag
if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False

# --- Action ---
if analyze_button:
    if not resume_text or not job_text:
        st.error("Please provide both resume and job description.")
    elif st.session_state.analyzing:
        st.warning("Another analysis is in progress. Please wait.")
    else:
        st.session_state.analyzing = True
        try:
            with st.spinner("🔄 Analyzing your resume ↔ job description... 🚀"):
                # Offload to subprocess for non-blocking
                with ProcessPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        process_resume_and_job_wrapper,
                        resume_text,
                        job_text,
                        load_nlp(),
                        load_embedder(),
                        load_skills_set()
                    )
                    results = future.result()

                similarity_score = compute_similarity(
                    results["resume_embedding"], results["job_embedding"])
                skill_match = compute_skill_match(
                    results["resume_skills"], results["job_skills"], job_text, top_n=30)

            # Results Header
            st.subheader("📈 Results")
            st.metric("Cosine Similarity Score", f"{similarity_score:.2f}")
            st.info(interpret_similarity(similarity_score))

            # --- Split Results into 2 Columns ---
            res_col1, res_col2 = st.columns(2)

            with res_col1:
                with st.expander("✅ Matched Skills: click to expand"):
                    st.success(f"{len(skill_match['overlap'])} skills matched")
                    st.write(", ".join(
                        skill_match["overlap"]) if skill_match["overlap"] else "No matched skills found")

            with res_col2:
                with st.expander("❌ Top Missing Skills: click to expand"):
                    st.error("skills missing")
                    st.write(", ".join(
                        skill_match["missing"]) if skill_match["missing"] else "No missing skills found")

            # Memory monitoring
            st.write(
                f"Memory usage after analysis: {psutil.Process().memory_info().rss / (1024 ** 2):.2f} MB")

        finally:
            st.session_state.analyzing = False
            gc.collect()  # Force garbage collection
