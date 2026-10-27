"""
Script to generate publication-grade figures for the Mini Project Report:
1. Target Severity Class Imbalance
2. Diurnal Temporal Collision Pattern (24-Hour Cycle)
3. Model Benchmark Performance Comparison
4. Multi-Model Confusion Matrices (2x2 Grid)
5. Random Forest Feature Importance Attribution
6. Per-Class Recall Sensitivity & Accuracy Paradox Breakdown
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.data_loader import load_rta_data, SEVERITY_ORDER
from src.train_models import MODELS_DIR, MODEL_FILENAMES, METRICS_CACHE_PATH

# Set styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

out_dir = project_root / "reports" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

df = load_rta_data()
cached = joblib.load(METRICS_CACHE_PATH)
comp_df = cached["comparison_df"]
metrics_cache = cached["metrics_cache"]

# Palette
C_ROSE = "#ef4444"
C_AMBER = "#f59e0b"
C_CYAN = "#06b6d4"
C_BLUE = "#3b82f6"
C_NAVY = "#1e293b"

# -----------------------------------------------------------------------------
# 1. Target Class Imbalance
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=300)
counts = df["Accident_severity"].value_counts().reindex(SEVERITY_ORDER)
colors = [C_ROSE, C_AMBER, C_CYAN]
bars = ax.bar(counts.index, counts.values, color=colors, width=0.55, edgecolor="#0f172a", linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    pct = yval / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 180, f"{yval:,}\n({pct:.2f}%)", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

ax.set_title("Road Traffic Accident Severity Class Distribution (N=12,316)", fontweight='bold', pad=14)
ax.set_ylabel("Number of Recorded Collisions")
ax.set_ylim(0, max(counts.values) * 1.18)
ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
plt.tight_layout()
fig.savefig(out_dir / "fig1_severity_imbalance.png", dpi=300)
plt.close(fig)
print("[Saved] fig1_severity_imbalance.png")

# -----------------------------------------------------------------------------
# 2. Diurnal Temporal Collision Pattern
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.2), dpi=300)
hourly = df["Hour"].value_counts().sort_index()
hours = list(range(24))
counts_h = [hourly.get(h, 0) for h in hours]

ax.plot(hours, counts_h, color=C_CYAN, marker='o', linewidth=2.4, markersize=5.5, label="Collision Volume")
ax.fill_between(hours, counts_h, color=C_CYAN, alpha=0.18)

peak_h = int(np.argmax(counts_h))
peak_val = max(counts_h)
ax.annotate(f"Peak: {peak_h:02d}:00\n({peak_val:,} collisions)",
            xy=(peak_h, peak_val), xytext=(peak_h - 4.5, peak_val - 120),
            arrowprops=dict(facecolor=C_ROSE, shrink=0.08, width=1.5, headwidth=6),
            fontweight='bold', color=C_ROSE, fontsize=9.5)

ax.set_title("Diurnal Collision Frequency Distribution Across the 24-Hour Cycle", fontweight='bold', pad=12)
ax.set_xlabel("Hour of the Day (0:00 - 23:00)")
ax.set_ylabel("Total Recorded Crashes")
ax.set_xticks(hours)
ax.set_xticklabels([f"{h}" for h in hours], fontsize=8.5)
ax.set_xlim(-0.5, 23.5)
ax.set_ylim(0, peak_val * 1.15)
plt.tight_layout()
fig.savefig(out_dir / "fig2_diurnal_hourly_pattern.png", dpi=300)
plt.close(fig)
print("[Saved] fig2_diurnal_hourly_pattern.png")

# -----------------------------------------------------------------------------
# 3. Model Benchmark Performance Comparison
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=300)
models = comp_df["Model"].tolist()
accs = comp_df["Accuracy"].values * 100
bal_accs = comp_df["Balanced Accuracy"].values * 100
macro_f1s = comp_df["Macro F1"].values * 100

x = np.arange(len(models))
width = 0.26

rects1 = ax.bar(x - width, accs, width, label='Overall Accuracy', color='#64748b', edgecolor='black', linewidth=0.8)
rects2 = ax.bar(x, bal_accs, width, label='Balanced Accuracy', color='#38bdf8', edgecolor='black', linewidth=0.8)
rects3 = ax.bar(x + width, macro_f1s, width, label='Macro F1-Score', color=C_CYAN, edgecolor='black', linewidth=0.8)

for rects in [rects1, rects2, rects3]:
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, rotation=0)

ax.set_title("Benchmarking Syllabus Classifiers on Held-Out Test Set (N=2,464)", fontweight='bold', pad=14)
ax.set_ylabel("Score (%)")
ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='bold', fontsize=9.5)
ax.set_ylim(0, 105)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
fig.savefig(out_dir / "fig3_model_comparison_benchmark.png", dpi=300)
plt.close(fig)
print("[Saved] fig3_model_comparison_benchmark.png")

# -----------------------------------------------------------------------------
# 4. Multi-Model Confusion Matrices (2x2 Grid)
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), dpi=300)
axes = axes.flatten()

model_keys = ["K-Nearest Neighbors", "Logistic Regression", "Decision Tree", "Random Forest"]

for idx, m_name in enumerate(model_keys):
    ax = axes[idx]
    cm = metrics_cache[m_name]["confusion_matrix"]
    # Normalize by true class row
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_title(f"{m_name}", fontweight='bold', fontsize=11, pad=8)
    
    tick_marks = np.arange(len(SEVERITY_ORDER))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(["Fatal", "Serious", "Slight"], fontsize=9)
    ax.set_yticklabels(["Fatal", "Serious", "Slight"], fontsize=9)
    
    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)"
            ax.text(j, i, val_str, ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black", fontsize=8.5, fontweight='bold')
            
    ax.set_ylabel("True Severity" if idx in [0, 2] else "")
    ax.set_xlabel("Predicted Severity" if idx in [2, 3] else "")
    ax.grid(False)

plt.suptitle("Confusion Matrices & Normalized Recall per Severity Tier (Held-Out Test Split)", fontsize=13, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(out_dir / "fig4_confusion_matrices.png", dpi=300)
plt.close(fig)
print("[Saved] fig4_confusion_matrices.png")

# -----------------------------------------------------------------------------
# 5. Random Forest Feature Importance Attribution
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)
rf_fi = metrics_cache["Random Forest"].get("feature_importances")
if rf_fi is not None:
    top12 = rf_fi.head(12).iloc[::-1]
    y_pos = np.arange(len(top12))
    ax.barh(y_pos, top12.values * 100, color='#0284c7', edgecolor='black', linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    clean_labels = [lbl.replace("cat__", "").replace("num__", "").replace("_", " ") for lbl in top12.index]
    ax.set_yticklabels(clean_labels, fontsize=9)
    ax.set_xlabel("Gini Feature Importance (%)")
    ax.set_title("Top-12 Predictors of Collision Severity (Random Forest Attribution)", fontweight='bold', pad=12)
    for i, v in enumerate(top12.values * 100):
        ax.text(v + 0.15, i, f"{v:.2f}%", va='center', fontsize=8.5, fontweight='bold')
    ax.set_xlim(0, max(top12.values * 100) * 1.18)

plt.tight_layout()
fig.savefig(out_dir / "fig5_feature_importance.png", dpi=300)
plt.close(fig)
print("[Saved] fig5_feature_importance.png")

# -----------------------------------------------------------------------------
# 6. Accuracy Paradox: Per-Class Recall Sensitivity
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4.4), dpi=300)
recalls_fatal = []
recalls_serious = []
recalls_slight = []

for m in model_keys:
    rep = metrics_cache[m]["classification_report"]
    recalls_fatal.append(rep["Fatal injury"]["recall"] * 100)
    recalls_serious.append(rep["Serious Injury"]["recall"] * 100)
    recalls_slight.append(rep["Slight Injury"]["recall"] * 100)

x = np.arange(len(model_keys))
width = 0.26

r1 = ax.bar(x - width, recalls_fatal, width, label='Fatal Recall (31 cases)', color=C_ROSE, edgecolor='black', linewidth=0.8)
r2 = ax.bar(x, recalls_serious, width, label='Serious Recall (349 cases)', color=C_AMBER, edgecolor='black', linewidth=0.8)
r3 = ax.bar(x + width, recalls_slight, width, label='Slight Recall (2,084 cases)', color=C_CYAN, edgecolor='black', linewidth=0.8)

for rects in [r1, r2, r3]:
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2.0, h + 1.2, f"{h:.1f}%", ha='center', va='bottom', fontsize=8, rotation=0)

ax.set_title("The 'Accuracy Paradox': Per-Class Recall across Severity Tiers", fontweight='bold', pad=14)
ax.set_ylabel("Class Recall / Sensitivity (%)")
ax.set_xticks(x)
ax.set_xticklabels(model_keys, fontweight='bold', fontsize=9.5)
ax.set_ylim(0, 115)
ax.legend(loc='upper right', frameon=True)
plt.tight_layout()
fig.savefig(out_dir / "fig6_accuracy_paradox_recall.png", dpi=300)
plt.close(fig)
print("[Saved] fig6_accuracy_paradox_recall.png")

print("\nALL 6 PUBLICATION FIGURES SUCCESSFULLY GENERATED IN reports/figures/!")
