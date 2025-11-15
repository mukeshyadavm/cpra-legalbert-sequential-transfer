

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Paths
files = {
    "Article 2": "/mnt/gdrive/MyDrive/NLI_Results/article2_nli_semantic_pairs_labeled.csv",
    "Article 3": "/mnt/gdrive/MyDrive/NLI_Results/article3_nli_semantic_pairs_labeled.csv",
    "Article 7": "/mnt/gdrive/MyDrive/NLI_Results/article7_nli_semantic_pairs_labeled.csv",
    "Article 8": "/mnt/gdrive/MyDrive/NLI_Results/article8_nli_semantic_pairs_labeled.csv"
}

# Collect row counts
rows_data = []
for name, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        rows_data.append((name, len(df)))
    else:
        rows_data.append((name, 0))

articles, counts = zip(*rows_data)

# Research-friendly gray
gray_color = "#6e6e6e"

plt.figure(figsize=(3.4, 2.5), dpi=300)
x = np.arange(len(articles))
bars = plt.bar(x, counts, width=0.6, color=gray_color, edgecolor="black", linewidth=0.8)

plt.xticks(x, articles, fontsize=8)
plt.ylabel("Rows", fontsize=9)
plt.title("Dataset Size per Article", fontsize=9.5)

plt.ylim(0, max(counts) * 1.15)

for bar, val in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + max(counts)*0.02,
             f"{val:,}",
             ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.margins(x=0.15)
plt.subplots_adjust(left=0.18, right=0.95, top=0.88, bottom=0.18)
plt.tight_layout()

# FIXED PATH HERE
out_base = "/mnt/gdrive/MyDrive/NLI_Results/article_row_counts_gray_singlecol"

plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_base}.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_base}.png and {out_base}.pdf")





import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---------- Config ----------
files = {
    "Article 2": "/mnt/gdrive/MyDrive/NLI_Results/article2_nli_semantic_pairs_labeled.csv",
    "Article 3": "/mnt/gdrive/MyDrive/NLI_Results/article3_nli_semantic_pairs_labeled.csv",
    "Article 7": "/mnt/gdrive/MyDrive/NLI_Results/article7_nli_semantic_pairs_labeled.csv",
    "Article 8": "/mnt/gdrive/MyDrive/NLI_Results/article8_nli_semantic_pairs_labeled.csv"
}


class_order = ["Neutral", "Entailment", "Contradiction"]

# Colors + hatches for research paper style
colors = {"Neutral": "#66c2a5", "Entailment": "#fc8d62", "Contradiction": "#8da0cb"}
hatches = {"Neutral": "//", "Entailment": "\\", "Contradiction": "xx"}

# Data prep
data = {cls: [] for cls in class_order}
percentages_data = {cls: [] for cls in class_order}

for title, path in files.items():
    df = pd.read_csv(path)
    counts = df["label"].value_counts().reindex(class_order).fillna(0).astype(int)
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    for cls in class_order:
        data[cls].append(counts[cls])
        percentages_data[cls].append(percentages[cls])

# Grouped bar chart
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bar_width = 0.2
x = np.arange(len(files))  # one position per article

for i, cls in enumerate(class_order):
    bars = ax.bar(x + i * bar_width, data[cls], width=bar_width,
                  label=cls, color=colors[cls], edgecolor="black", hatch=hatches[cls])

    # Add percentage labels above bars
    for bar, pct in zip(bars, percentages_data[cls]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + (0.02 * max(max(v) for v in data.values())),
                f"{pct:.1f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

# Formatting
ax.set_xlabel("Articles", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Class Distribution Across Articles", fontsize=12)
ax.set_xticks(x + bar_width)
ax.set_xticklabels(list(files.keys()), fontsize=9)
ax.legend(title="Class", fontsize=9)

plt.tight_layout()

# Save
out_base = "/mnt/gdrive/MyDrive/NLI_Results/class_distribution_grouped"
plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{out_base}.pdf", dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved: {out_base}.png and {out_base}.pdf")
