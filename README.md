# Project Skee Gap

## Resume-to-Job Skill Matcher

A resume and job description skill extraction and matching tool that leverages Natural Language Processing (NLP), Sentence Transformers, and fuzzy matching to help candidates and recruiters quickly identify how well a resume aligns with a given job description.

This project provides a Streamlit-based web application where users can:

1. Upload or paste a resume (.pdf, txt, or .docx)

2. Upload or Paste a job description

3. Extract and match skills against a skills knowledge base (skills.csv)

4. View expandable sections of Matched Skills and Missing Skills

5. Gain insights into resume-job fit in an interactive and user-friendly manner

🔑 Key Features

1. Skill Extraction from Resume:
   Uses spaCy and regex rules to extract relevant skills from the resume text.

2. Job Description Skill Extraction:
   Identifies required skills from the job description using NLP and similarity scoring.

3. Fuzzy Matching with Skills Dataset:
   Even if the exact skill wording differs, fuzzy string matching ensures relevant skills are captured (e.g., “PyTorch” vs. “Torch”).

4. Sentence Transformer Embeddings:
   Improves semantic matching of skills by comparing embeddings rather than just keywords.

5. Expandable Matched/Missing Skills Panels:
   Avoids screen clutter by hiding detailed lists until the user chooses to expand them.

6. Interactive Streamlit App:
   Clean, user-friendly interface to upload resumes, paste job descriptions, and instantly view results.

📂 Project Structure
resume-skill-matcher/
│
├── app.py # Streamlit main app
├── extract_skills.py # Skill extraction + fuzzy/semantic matching logic
├── skills.csv # Master skills dataset (expandable/customizable)
├── requirements.txt # Project dependencies
├── README.md # Documentation (this file)
└── sample_resumes/ # (Optional) Example resumes for testing
├── scores.py # compute the scores, compares
└── extract_text.py # Extract text from resume and job desc.

🏗️ Architecture Overview
flowchart TD
A[User Uploads Resume] -->|PDF/DOCX Parsing| B[Resume Text Extraction]
C[User Pastes Job Description] --> D[Job Description Text Extraction]

    B --> E[Skill Extraction via spaCy + Regex]
    D --> F[Skill Extraction via spaCy + Regex]

    E --> G[Fuzzy Matching with skills.csv]
    F --> G

    G --> H[Sentence Transformer Embeddings for Semantic Similarity]
    H --> I[Matched Skills / Missing Skills Classification]

    I --> J[Streamlit Expandable Panels]

🧩 Skills Dataset (skills.csv)
The skills dataset use in the project is from Skill2vec (2017).

@article{van2017skill2vec,
title={Skill2vec: Machine Learning Approach for Determining the Relevant Skills from Job Description},
author={Van-Duyet, Le and Quan, Vo Minh and An, Dang Quang},
journal={arXiv preprint arXiv:1707.09751},
year={2017}
}

The file skills.csv contains the reference set of skills used for matching. You can customize it for different industries.

Example structure:

skill
Python
SQL
Data Analysis
Machine Learning
Deep Learning
Project Management
Docker
AWS

You can expand this dataset to cover domain-specific skill sets (e.g., finance, healthcare, cybersecurity).

📊 Example Workflow

Upload resume.pdf

Paste job description into the app

Click Analyze

View results:

✅ Matched Skills (expandable list)

❌ Missing Skills (expandable list)

Use insights to improve your resume or assess candidate fit.

🚀 Roadmap / Future Enhancements

Add resume scoring system (percentage match score)

Generate recommendations for missing skills

Extend support for multi-page resumes and multiple job postings

Add export to PDF/Excel feature for results

Enhance semantic similarity using large language models (LLMs)

Integrate with LinkedIn or job boards for automatic skill extraction

📜 License

This project is licensed under the MIT License.

🙌 Acknowledgments

spaCy for NLP pipelines

Sentence Transformers for semantic similarity

RapidFuzz for efficient fuzzy string matching

Streamlit for building the interactive web app

@article{van2017skill2vec,
title={Skill2vec: Machine Learning Approach for Determining the Relevant Skills from Job Description},
author={Van-Duyet, Le and Quan, Vo Minh and An, Dang Quang},
journal={arXiv preprint arXiv:1707.09751},
year={2017}
}
