"""
Student Mental Health & Burnout – Machine Learning Analysis
Dataset : student_mental_health_burnout.csv (150,000 records, 20 columns)
Source  : Kaggle
Models  : Decision Tree | Naïve Bayes | K-Nearest Neighbors | Logistic Regression
Target  : burnout_level (High / Medium / Low)
"""

import os
import warnings
import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

_BASE      = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, "..", "output")
REPORT_DIR = os.path.join(_BASE, "..", "report")
FONTS_DIR  = os.path.join(_BASE, "..", "fonts")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# Register font
for _ttf in ["DMSans-Regular.ttf", "DMSans-Bold.ttf", "DMSans-Italic.ttf"]:
    fm.fontManager.addfont(os.path.join(FONTS_DIR, _ttf))

plt.rcParams["font.family"] = "DM Sans"

def save(fig, filename):
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# DATA
# =============================================================================
DATA_PATH = os.path.join(_BASE, "..", "data", "student_mental_health_burnout.csv")
df = pd.read_csv(DATA_PATH)


# =============================================================================
# VISUALISE THE DATASET
# =============================================================================
palette = ["#6C5CE7", "#00CEC9", "#FD7272"]
order   = ["Low", "Medium", "High"]

# Target distribution — check class balance before training
fig, ax = plt.subplots(figsize=(7, 4))
counts = df["burnout_level"].value_counts().reindex(order)
bars = ax.bar(order, counts.values, color=palette, edgecolor="white", linewidth=1.5)
for bar, val in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Distribution of Burnout Level (Target Variable)", fontsize=13, fontweight="bold")
ax.set_xlabel("Burnout Level")
ax.set_ylabel("Number of Students")
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")
plt.tight_layout()
save(fig, "01_target_distribution.png")

# Age distribution
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["age"], bins=9, color="#6C5CE7", edgecolor="white", linewidth=1.3)
ax.set_title("Age Distribution of Students", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.set_facecolor("#F8F9FA")
plt.tight_layout()
save(fig, "02_age_distribution.png")

# Stress level vs burnout — cross-tabulation of two categorical features
fig, ax = plt.subplots(figsize=(8, 4))
ct = pd.crosstab(df["stress_level"], df["burnout_level"])[order]
ct.plot(kind="bar", ax=ax, color=palette, edgecolor="white", linewidth=1.2)
ax.set_title("Stress Level vs Burnout Level", fontsize=13, fontweight="bold")
ax.set_xlabel("Stress Level")
ax.set_ylabel("Number of Students")
ax.legend(title="Burnout Level")
plt.xticks(rotation=0)
ax.set_facecolor("#F8F9FA")
plt.tight_layout()
save(fig, "03_stress_vs_burnout.png")

# CGPA by burnout level — box plot shows spread and median per group
fig, ax = plt.subplots(figsize=(8, 4))
for i, (level, color) in enumerate(zip(order, palette)):
    data = df[df["burnout_level"] == level]["cgpa"]
    ax.boxplot(data, positions=[i], patch_artist=True, widths=0.5,
               boxprops=dict(facecolor=color, alpha=0.7),
               medianprops=dict(color="black", linewidth=2),
               whiskerprops=dict(linewidth=1.5),
               capprops=dict(linewidth=1.5),
               flierprops=dict(marker=".", markersize=2, alpha=0.3))
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order)
ax.set_title("CGPA Distribution by Burnout Level", fontsize=13, fontweight="bold")
ax.set_xlabel("Burnout Level")
ax.set_ylabel("CGPA")
ax.set_facecolor("#F8F9FA")
plt.tight_layout()
save(fig, "04_cgpa_vs_burnout.png")


# Sleep hours by burnout level — overlapping histograms per class
fig, ax = plt.subplots(figsize=(8, 4))
for level, color in zip(order, palette):
    data = df[df["burnout_level"] == level]["daily_sleep_hours"]
    ax.hist(data, bins=30, alpha=0.6, label=level, color=color)
ax.set_title("Daily Sleep Hours by Burnout Level", fontsize=13, fontweight="bold")
ax.set_xlabel("Daily Sleep Hours")
ax.set_ylabel("Frequency")
ax.legend(title="Burnout Level")
ax.set_facecolor("#F8F9FA")
plt.tight_layout()
save(fig, "06_sleep_vs_burnout.png")


# =============================================================================
# DETERMINE X (FEATURES) AND Y (TARGET)
# =============================================================================
# Y = burnout_level (what we predict) — 3 classes: Low, Medium, High
# X = all other columns except student_id

df.drop(columns=["student_id"], inplace=True)

categorical_cols = ["gender", "course", "year", "stress_level",
                    "sleep_quality", "internet_quality"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

target_col = "burnout_level"
df[target_col] = le.fit_transform(df[target_col]) # High=0, Low=1, Medium=2

feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols] # 18 input features
y = df[target_col] # target


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================
# 80/20 split; stratify=y keeps class proportions equal in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler: normalises features to mean=0, std=1
# Required by KNN (distance-based) and Logistic Regression (gradient-based)
# Decision Tree and Naïve Bayes are scale-invariant — they use the raw split
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# Decision Tree — splits data on feature thresholds to form a decision tree
# max_depth=10 limits tree size to prevent overfitting
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)

# Naïve Bayes — probabilistic classifier based on Bayes' theorem
# Assumes each feature is normally distributed and independent of the others
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_score = nb_model.score(X_test, y_test)

# K-Nearest Neighbors — classifies by majority vote of the 5 nearest neighbours
# Uses Euclidean distance, so scaled features are required
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_sc, y_train)
knn_score = knn_model.score(X_test_sc, y_test)

# Logistic Regression — estimates class probabilities using the softmax function
# Requires scaled input; max_iter=500 ensures convergence on this dataset
lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
lr_model.fit(X_train_sc, y_train)
lr_score = lr_model.score(X_test_sc, y_test)

# Accuracy = correct predictions / total predictions, evaluated on the test set
results = {
    "Decision Tree"      : dt_score,
    "Naïve Bayes"        : nb_score,
    "KNN (k=5)"          : knn_score,
    "Logistic Regression": lr_score,
}
best_model = max(results, key=results.get)

# Bar chart — side-by-side accuracy comparison
model_colors = ["#6C5CE7", "#00CEC9", "#FD7272", "#FDCB6E"]
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(results.keys(), results.values(),
              color=model_colors, edgecolor="white", linewidth=1.5, width=0.5)
for bar, acc in zip(bars, results.values()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{acc:.4f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold")
ax.set_ylim(0, 1.08)
ax.set_title("Model Accuracy Comparison\n(Student Mental Health & Burnout Dataset)",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Accuracy Score", fontsize=11)
ax.axhline(y=max(results.values()), color="red", linestyle="--",
           linewidth=1, alpha=0.5, label=f"Best: {max(results.values()):.4f}")
ax.legend(fontsize=9)
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")
plt.xticks(rotation=10)
plt.tight_layout()
save(fig, "07_model_accuracy_comparison.png")

# Line chart — accuracy trend across models
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(results.keys()), list(results.values()),
        marker="o", markersize=10, linewidth=2.5,
        color="#6C5CE7", markerfacecolor="#FD7272", markeredgewidth=2)
for i, (name, acc) in enumerate(results.items()):
    ax.annotate(f"{acc:.4f}", xy=(i, acc),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold", color="#2D3436")
ax.set_ylim(0, 1.08)
ax.set_title("Model Accuracy – Line Chart", fontsize=14, fontweight="bold")
ax.set_xlabel("Model", fontsize=11)
ax.set_ylabel("Accuracy Score", fontsize=11)
ax.set_facecolor("#F8F9FA")
fig.patch.set_facecolor("white")
plt.xticks(rotation=10)
plt.tight_layout()
save(fig, "08_model_accuracy_line.png")


# =============================================================================
# PDF REPORT
# =============================================================================
PDF_PATH = os.path.join(REPORT_DIR, "ML_Student_Mental_Health_Report.pdf")

# Each entry: (image file, title, explanation text, code snippet lines)
chart_pages = [
    ("01_target_distribution.png",
     "Chart 1 – Burnout Level Distribution",
     "This bar chart shows the number of students in each burnout category: Low, Medium, and High. "
     "Checking the class balance is the first step before training any classifier — a heavily skewed "
     "distribution can cause models to favour the majority class. A roughly equal distribution "
     "means the models can learn meaningful patterns for all three burnout levels.",
     [
         'counts = df["burnout_level"].value_counts().reindex(order)',
         'bars = ax.bar(order, counts.values, color=palette, edgecolor="white", linewidth=1.5)',
         'for bar, val in zip(bars, counts.values):',
         '    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,',
         '            f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")',
         'ax.set_title("Distribution of Burnout Level (Target Variable)")',
     ]),

    ("02_age_distribution.png",
     "Chart 2 – Age Distribution",
     "This histogram shows the age range of students in the dataset (17–25 years). "
     "The narrow range is expected for a university-level study population. "
     "Age is kept as a feature because coping strategies and academic pressure may differ "
     "as students progress through their studies, which can influence burnout outcomes.",
     [
         'ax.hist(df["age"], bins=9, color="#6C5CE7", edgecolor="white", linewidth=1.3)',
         'ax.set_title("Age Distribution of Students")',
         'ax.set_xlabel("Age")',
         'ax.set_ylabel("Frequency")',
     ]),

    ("03_stress_vs_burnout.png",
     "Chart 3 – Stress Level vs Burnout Level",
     "This grouped bar chart crosses two categorical variables: self-reported stress level "
     "(Low / Medium / High) and burnout level. It visually tests whether students who report "
     "high stress are disproportionately represented in the High burnout group. "
     "A strong visible pattern confirms that stress level is a meaningful predictor for burnout.",
     [
         'ct = pd.crosstab(df["stress_level"], df["burnout_level"])[order]',
         'ct.plot(kind="bar", ax=ax, color=palette, edgecolor="white", linewidth=1.2)',
         'ax.set_title("Stress Level vs Burnout Level")',
         'ax.legend(title="Burnout Level")',
     ]),

    ("04_cgpa_vs_burnout.png",
     "Chart 4 – CGPA by Burnout Level",
     "These box plots compare the distribution of CGPA (Cumulative Grade Point Average) "
     "across the three burnout groups. The box spans the 25th to 75th percentile, the line "
     "inside is the median, and the whiskers cover the data range. Differences in median CGPA "
     "between groups indicate that academic performance is a useful predictor.",
     [
         'for i, (level, color) in enumerate(zip(order, palette)):',
         '    data = df[df["burnout_level"] == level]["cgpa"]',
         '    ax.boxplot(data, positions=[i], patch_artist=True, widths=0.5)',
         'ax.set_title("CGPA Distribution by Burnout Level")',
     ]),

    ("06_sleep_vs_burnout.png",
     "Chart 5 – Daily Sleep Hours by Burnout Level",
     "These overlapping histograms compare the number of hours per day students sleep, grouped by "
     "burnout level. If students with High burnout consistently sleep fewer hours, sleep duration is "
     "a strong predictor. The overlap between distributions reflects real-world variability and is "
     "well handled by both probabilistic (Naïve Bayes) and distance-based (KNN) classifiers.",
     [
         'for level, color in zip(order, palette):',
         '    data = df[df["burnout_level"] == level]["daily_sleep_hours"]',
         '    ax.hist(data, bins=30, alpha=0.6, label=level, color=color)',
         'ax.set_title("Daily Sleep Hours by Burnout Level")',
         'ax.legend(title="Burnout Level")',
     ]),

    ("07_model_accuracy_comparison.png",
     "Chart 6 – Model Accuracy Comparison (Bar Chart)",
     "This bar chart provides a side-by-side comparison of the test-set accuracy achieved by each "
     "of the four trained models. Accuracy is the proportion of correctly classified samples out of "
     "all test samples (30,000 records). The dashed red line marks the best-performing model. "
     "Accuracy is a valid primary metric given the roughly balanced class distribution.",
     [
         'results = {"Decision Tree": dt_score, "Naive Bayes": nb_score,',
         '           "KNN (k=5)": knn_score, "Logistic Regression": lr_score}',
         'bars = ax.bar(results.keys(), results.values(), color=model_colors, width=0.5)',
         'ax.axhline(y=max(results.values()), color="red", linestyle="--")',
     ]),

    ("08_model_accuracy_line.png",
     "Chart 7 – Model Accuracy Comparison (Line Chart)",
     "This line chart presents the same accuracy values as Chart 6 but as a connected trend line, "
     "making relative differences between models easier to see at a glance. Each point is annotated "
     "with its exact accuracy score. A steeper slope between two adjacent models indicates a more "
     "significant performance difference.",
     [
         'ax.plot(list(results.keys()), list(results.values()),',
         '        marker="o", markersize=10, linewidth=2.5, color="#6C5CE7")',
         'for i, (name, acc) in enumerate(results.items()):',
         '    ax.annotate(f"{acc:.4f}", xy=(i, acc), xytext=(0, 12),',
         '                textcoords="offset points", ha="center")',
     ]),
]


def pdf_text_page(pdf, title, body_paragraphs, code_lines=None):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.text(0.08, 0.90, title,
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            color="#1a1a1a", va="top")
    ax.axhline(y=0.852, xmin=0.08, xmax=0.92, color="#DDDDDD", linewidth=1)

    WRAP_WIDTH = 94
    LINE_H     = 0.020
    PARA_GAP   = 0.032

    y_cursor = 0.815
    for para in body_paragraphs:
        if para.strip() == "":
            y_cursor -= PARA_GAP
            continue
        wrapped    = textwrap.fill(para, width=WRAP_WIDTH)
        line_count = len(wrapped.split("\n"))
        ax.text(0.08, y_cursor, wrapped,
                transform=ax.transAxes, fontsize=10.5,
                color="#333333", va="top", linespacing=1.6)
        y_cursor -= LINE_H * line_count + PARA_GAP

    # Render styled code block if code lines are provided
    if code_lines:
        y_cursor -= 0.050

        CODE_LINE_H  = 0.037
        code_block_h = CODE_LINE_H * len(code_lines) + 0.022

        # Grey rounded background rectangle
        rect = FancyBboxPatch(
            (0.08, y_cursor - code_block_h + 0.010),
            0.84, code_block_h,
            transform=ax.transAxes,
            boxstyle="round,pad=0.008",
            facecolor="#F4F4F4", edgecolor="#DEDEDE", linewidth=0.8, zorder=0
        )
        ax.add_patch(rect)

        # Code lines in monospace
        y_code = y_cursor - 0.012
        for line in code_lines:
            ax.text(0.105, y_code, line,
                    transform=ax.transAxes, fontsize=9,
                    color="#1e1e1e", va="top", family="monospace", zorder=1)
            y_code -= CODE_LINE_H

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(PDF_PATH) as pdf:

    # Title page
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    ax.axhline(y=0.92, xmin=0.08, xmax=0.92, color="#6C5CE7", linewidth=3)
    ax.text(0.08, 0.83, "Student Mental Health & Burnout",
            transform=ax.transAxes, fontsize=26, fontweight="bold",
            color="#1a1a1a", va="top")
    ax.text(0.08, 0.74, "Machine Learning Analysis Report",
            transform=ax.transAxes, fontsize=16, color="#555555", va="top")
    ax.axhline(y=0.69, xmin=0.08, xmax=0.92, color="#EEEEEE", linewidth=1)

    rows = [
        ("Dataset",         "student_mental_health_burnout.csv  ·  150,000 records",    0.068),
        ("Target Variable", "burnout_level  (High / Medium / Low)  —  Multi-class Classification",                         0.068),
        ("Features (X)",    "age, CGPA, sleep hours, study hours, screen time, anxiety score, depression score,",           0.050),
        ("",                "academic pressure, financial stress, social support, physical activity, attendance,",           0.050),
        ("",                "gender, course, year, stress level, sleep quality, internet quality",                          0.068),
        ("Models",          "Decision Tree   ·   Naïve Bayes   ·   K-Nearest Neighbors (k=5)   ·   Logistic Regression",   0.068),
        ("Train / Test",    "80% training  /  20% testing  ·  stratified split  ·  random_state = 42",                     0.068),
        ("Scaling",         "StandardScaler applied for KNN and Logistic Regression",                                      0.068),
    ]

    y = 0.64
    for label, value, step in rows:
        if label:
            ax.text(0.08, y, label, transform=ax.transAxes,
                    fontsize=9, color="#888888", va="top", fontweight="bold")
        ax.text(0.26, y, value, transform=ax.transAxes,
                fontsize=10, color="#333333", va="top")
        y -= step

    ax.axhline(y=0.08, xmin=0.08, xmax=0.92, color="#EEEEEE", linewidth=1)
    ax.text(0.08, 0.055, "Libraries: pandas  ·  numpy  ·  matplotlib  ·  seaborn  ·  scikit-learn",
            transform=ax.transAxes, fontsize=9, color="#AAAAAA", va="top")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # Dataset overview
    pdf_text_page(pdf, "1. Dataset Overview", [
        "The dataset used in this analysis is the Student Mental Health & Burnout dataset sourced from Mansehaj Preet on Kaggle. "
        "It contains 150,000 student records across 20 columns, covering a mix of demographic, academic, and "
        "mental health variables. There are no missing values, and each row represents a unique student.",
        "",
        "Numerical features (12): age, daily study hours, daily sleep hours, screen time hours, anxiety score, "
        "depression score, academic pressure score, financial stress score, social support score, "
        "physical activity hours, attendance percentage, and CGPA.",
        "",
        "Categorical features (6): gender, course, year of study, stress level, sleep quality, and internet quality.",
        "",
        "The dataset is well-suited for classification because it combines both types of variables, has a "
        "sufficiently large sample size (150,000 records), and contains a clearly defined target variable "
        "with three classes that reflect real-world burnout levels.",
        "",
        "Kaggle Link: http://bit.ly/4bKuJuX"
    ])

    # X and Y explanation
    pdf_text_page(pdf, "2. Determining X (Features) and Y (Target)", [
        "Y — Target Variable:",
        "burnout_level is the column the models are trained to predict. "
        "It has three possible values: Low, Medium, and High, making this a multi-class classification problem. "
        "The target is Label-Encoded to integers (High = 0, Low = 1, Medium = 2) so scikit-learn can process it.",
        "",
        "X — Feature Variables:",
        "All remaining columns, after removing student_id (a row identifier with no predictive value), "
        "are used as input features. This gives 18 features per student. A combination of raw numerical "
        "measurements and encoded categorical labels.",
        "",
        "Encoding & Scaling:",
        "Categorical columns (gender, course, year, stress_level, sleep_quality, internet_quality) are "
        "converted to integers via Label Encoding. For Decision Tree and Naïve Bayes, features are used "
        "as-is. These algorithms are not affected by feature scale. For KNN and Logistic Regression, "
        "all features are standardised using StandardScaler (mean = 0, standard deviation = 1) to ensure "
        "that features with larger numerical ranges do not dominate the model.",
    ])

    # Chart pages + explanation + code pages
    for fname, chart_title, explanation, code in chart_pages:
        img_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(img_path):
            continue
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.5, 0.01, chart_title, ha="center", va="bottom",
                 fontsize=10, color="#777777", style="italic")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        pdf_text_page(pdf, chart_title, [explanation], code_lines=code)

    # Accuracy summary
    pdf_text_page(pdf, "Model Accuracy Summary", [
        "The table below summarises the test-set accuracy of all four trained models. "
        "Accuracy is defined as the proportion of correctly classified samples out of the total test samples "
        "(30,000 records — 20% of the full dataset).",
        "",
        f"  Decision Tree        :  {dt_score:.4f}   ({dt_score * 100:.2f}%)",
        f"  Naïve Bayes          :  {nb_score:.4f}   ({nb_score * 100:.2f}%)",
        f"  KNN (k=5)            :  {knn_score:.4f}   ({knn_score * 100:.2f}%)",
        f"  Logistic Regression  :  {lr_score:.4f}   ({lr_score * 100:.2f}%)",
        "",
        f"Best performing model: {best_model}  ({results[best_model] * 100:.2f}%)",
        "",
    ])

    d = pdf.infodict()
    d["Title"]   = "Student Mental Health & Burnout – ML Report"
    d["Author"]  = "ML Pipeline (main.py)"
    d["Subject"] = "Machine Learning Classification"