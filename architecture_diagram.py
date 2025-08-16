from graphviz import Digraph

# Create Digraph
dot = Digraph("ResumeAnalyzer", format="png")
dot.attr(rankdir="TB", size="8")

# Nodes
dot.node("input", "User Input\n- Upload Resume (PDF)\n- Paste/Upload Job Description", shape="box", style="filled", fillcolor="#cce5ff")

dot.node("preprocess", "Data Preprocessing\n- Extract & Clean Text\n- Tokenization\n- Skills/NER Extraction", shape="box", style="filled", fillcolor="#e2e3e5")

dot.node("embedding", "Embedding & Feature Extraction\n- Sentence Transformers\n- Resume Embedding\n- JD Embedding\n- Skills Dictionary", shape="box", style="filled", fillcolor="#d4edda")

dot.node("scoring", "Matching & Scoring Engine\n- Cosine Similarity\n- Skill Overlap\n- Match Score (0-100%)", shape="box", style="filled", fillcolor="#fff3cd")

dot.node("viz", "Visualization & Results\n- Score Gauge\n- Matched/Missing Skills\n- Wordcloud / Venn Diagram\n- Feedback Report", shape="box", style="filled", fillcolor="#f8d7da")

dot.node("webapp", "Web App Layer (Streamlit)\n- Interactive UI\n- Upload Widgets\n- Real-time Scoring\n- Public URL", shape="box", style="filled", fillcolor="#d1ecf1")

# Edges
dot.edges([("input", "preprocess"),
           ("preprocess", "embedding"),
           ("embedding", "scoring"),
           ("scoring", "viz"),
           ("viz", "webapp")])

# Render diagram
file_path = "./skee_gap_architecture"
dot.render(file_path, format="png", cleanup=True)
file_path + ".png"
