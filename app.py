import streamlit as st
from utils.extract_text import extract_text_from_pdf, extract_text_from_docx, extract_text_from_txt

st.title("Resume Analyzer: Input Upload")

MIN_CHAR_COUNT = 300  # threshold for suspiciously short extraction

# --- Resume Upload ---
st.subheader("Upload or Paste Your Resume")

resume_text = ""

resume_file = st.file_uploader("Choose your resume (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
if resume_file:
    if resume_file.type == "application/pdf":
        resume_text = extract_text_from_pdf(resume_file)
    elif resume_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume_text = extract_text_from_docx(resume_file)
    elif resume_file.type == "text/plain":
        resume_text = extract_text_from_txt(resume_file)

# Option: Paste resume text manually
resume_text_paste = st.text_area("Or paste your resume text here", height=200)
if resume_text_paste.strip():
    resume_text = resume_text_paste

# Show resume and character count
if resume_text:
    st.write("### Resume Extracted Text")
    st.text_area("Resume Content", resume_text[:2000], height=200)
    st.write(f"**Character count:** {len(resume_text)}")
    
    if len(resume_text) < MIN_CHAR_COUNT:
        st.warning("⚠️ Extracted resume text seems very short. Consider pasting your resume text manually for better accuracy.")

# --- Job Description Input ---
st.subheader("Paste or Upload Job Description")

job_text = st.text_area("Paste Job Description", height=200)

job_file = st.file_uploader("Or upload a job description (PDF/DOCX/TXT)", type=["pdf", "docx", "txt"])
if job_file:
    if job_file.type == "application/pdf":
        job_text = extract_text_from_pdf(job_file)
    elif job_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        job_text = extract_text_from_docx(job_file)
    elif job_file.type == "text/plain":
        job_text = extract_text_from_txt(job_file)

# Show job description and character count
if job_text:
    st.write("### Job Description Extracted Text")
    st.text_area("Job Description Content", job_text[:2000], height=200)
    st.write(f"**Character count:** {len(job_text)}")
    
    if len(job_text) < MIN_CHAR_COUNT:
        st.warning("⚠️ Extracted job description seems very short. Consider pasting the text manually for better accuracy.")
