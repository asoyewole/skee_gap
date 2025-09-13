import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Create figure
fig, ax = plt.subplots(figsize=(12, 8))

# Define components and their positions
components = {
    "S3\n(Storage)": (0.1, 0.8),
    "Lambda\n(Text Extraction)": (0.4, 0.8),
    "SageMaker\n(Embeddings & Similarity)": (0.7, 0.8),
    "DynamoDB\n(Cache)": (0.7, 0.5),
    "API Gateway": (0.4, 0.5),
    "Lambda\n(Backend)": (0.55, 0.5),
    "Streamlit\n(Frontend - EC2/Amplify)": (0.4, 0.2),
    "CloudWatch\n(Logs)": (0.85, 0.65),
    "CloudTrail\n(Auditing)": (0.85, 0.45)
}

# Draw boxes for components
for comp, (x, y) in components.items():
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), 0.18, 0.1, boxstyle="round,pad=0.05",
        edgecolor="black", facecolor="lightblue"
    ))
    ax.text(x + 0.09, y + 0.05, comp, ha="center", va="center", fontsize=9)

# Define connections (start -> end)
connections = [
    ("S3\n(Storage)", "Lambda\n(Text Extraction)"),
    ("Lambda\n(Text Extraction)", "SageMaker\n(Embeddings & Similarity)"),
    ("SageMaker\n(Embeddings & Similarity)", "DynamoDB\n(Cache)"),
    ("API Gateway", "Lambda\n(Backend)"),
    ("Lambda\n(Backend)", "SageMaker\n(Embeddings & Similarity)"),
    ("Lambda\n(Backend)", "DynamoDB\n(Cache)"),
    ("Streamlit\n(Frontend - EC2/Amplify)", "API Gateway"),
    ("Lambda\n(Backend)", "CloudWatch\n(Logs)"),
    ("API Gateway", "CloudTrail\n(Auditing)")
]

# Draw arrows
for start, end in connections:
    x1, y1 = components[start]
    x2, y2 = components[end]
    ax.annotate("", xy=(x2+0.09, y2+0.05), xytext=(x1+0.09, y1+0.05),
                arrowprops=dict(arrowstyle="->", lw=1.2))

# Formatting
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.set_title("AWS Architecture for Resume-JD Skill Matching System",
             fontsize=14, weight="bold")

plt.show()
