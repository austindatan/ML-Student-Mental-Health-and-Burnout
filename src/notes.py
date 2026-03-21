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
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

# Use abspath so paths resolve correctly regardless of where the script is launched from
_BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_BASE, "..", "output")
REPORT_DIR = os.path.join(_BASE, "..", "report")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

def save(fig, filename):
    """Save a matplotlib figure to the output folder and close it."""
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 1.  LOAD DATA
# =============================================================================
# Read the CSV into a pandas DataFrame.
# The dataset has 150,000 rows and 20 columns covering student demographics,
# academic performance metrics, mental health scores, and a burnout label.
DATA_PATH = os.path.join(_BASE, "..", "data", "student_mental_health_burnout.csv")
df = pd.read_csv(DATA_PATH)


# =============================================================================
# 2.  VISUALISE THE DATASET
# =============================================================================
# A shared color palette and consistent class ordering are used across all charts
palette = ["#6C5CE7", "#00CEC9", "#FD7272"]
order   = ["Low", "Medium", "High"]   # fixed burnout-level order for all charts

# Chart 1 – Target variable distribution
# Shows how many students fall into each burnout category (class balance check)
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

# Chart 2 – Age distribution
# Verifies the dataset represents a typical university-age population (17–25 yrs)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(df["age"], bins=9, color="#6C5CE7", edgecolor="white", linewidth=1.3)
ax.set_title("Age Distribution of Students", fontsize=13, fontweight="bold")
ax.set_xlabel("Age")
ax.set_ylabel("Frequency")
ax.set_facecolor("#F8F9FA")
plt.tight_layout()
save(fig, "02_age_distribution.png")

# Chart 3 – Stress level vs Burnout level
# Cross-tabulation shows whether high stress correlates with high burnout
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

# Chart 4 – CGPA by Burnout level (box plot)
# Compares academic performance distributions across the three burnout groups
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

# Chart 5 – Correlation heatmap (numerical features only)
# Pearson correlations reveal which features move together;
# multicollinearity is important context for Logistic Regression
numerical_cols = ["age", "daily_study_hours", "daily_sleep_hours",
                  "screen_time_hours", "anxiety_score", "depression_score",
                  "academic_pressure_score", "financial_stress_score",
                  "social_support_score", "physical_activity_hours",
                  "attendance_percentage", "cgpa"]
fig, ax = plt.subplots(figsize=(12, 8))
corr = df[numerical_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, annot_kws={"size": 8})
ax.set_title("Correlation Heatmap – Numerical Features", fontsize=13, fontweight="bold")
plt.tight_layout()
save(fig, "05_correlation_heatmap.png")

# Chart 6 – Daily sleep hours by Burnout level
# Overlapping histograms reveal whether more burned-out students sleep less
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
# 3.  DETERMINE X (FEATURES) AND Y (TARGET)
# =============================================================================
# Y – Target variable:
#   burnout_level is the column we aim to predict. It has three classes:
#   Low, Medium, High → making this a multi-class classification problem.
#
# X – Feature variables:
#   Every column except student_id (a meaningless row identifier) and the
#   target itself is used as a feature. This gives 18 predictive features.
#
# Encoding:
#   Categorical columns are Label-Encoded (strings → integers) because all
#   scikit-learn models require numerical input.
#   The target column is also Label-Encoded: High=0, Low=1, Medium=2.

df.drop(columns=["student_id"], inplace=True)   # drop non-predictive ID column

categorical_cols = ["gender", "course", "year", "stress_level",
                    "sleep_quality", "internet_quality"]
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])         # encode each categorical column

target_col   = "burnout_level"
df[target_col] = le.fit_transform(df[target_col])   # encode target

feature_cols = [c for c in df.columns if c != target_col]
X = df[feature_cols]   # 18 features (numerical + encoded categorical)
y = df[target_col]     # 1 target column


# =============================================================================
# 4.  TRAIN / TEST SPLIT + FEATURE SCALING
# =============================================================================
# 80% of data is used to train the models; 20% is held out for testing.
# stratify=y preserves the class proportions in both train and test sets.
# random_state=42 ensures the same split every run (reproducibility).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# StandardScaler normalises each feature to mean=0, standard deviation=1.
# KNN (distance-based) and Logistic Regression (gradient-based) require scaled data.
# Decision Tree and Naïve Bayes are scale-invariant and use the raw split directly.
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train)   # fit on train, then transform
X_test_sc    = scaler.transform(X_test)         # transform test using train statistics


# =============================================================================
# 5.  FIT THE 4 MODELS
# =============================================================================

# Model 1 – Decision Tree
# Recursively partitions the feature space using threshold rules at each node.
# max_depth=10 caps tree depth to prevent overfitting.
# Uses raw (unscaled) feature values.
dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)

# Model 2 – Naïve Bayes (Gaussian)
# Applies Bayes' theorem assuming each feature independently follows a
# normal (Gaussian) distribution. Fast and simple; scale-invariant.
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_score = nb_model.score(X_test, y_test)

# Model 3 – K-Nearest Neighbors (k=5)
# Classifies each test sample by a majority vote of its 5 nearest training
# neighbours (Euclidean distance). Scaling is required because large-range
# features would otherwise dominate the distance calculation.
knn_model = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn_model.fit(X_train_sc, y_train)
knn_score = knn_model.score(X_test_sc, y_test)

# Model 4 – Logistic Regression
# Linear model that estimates class probabilities via the softmax function.
# max_iter=500 gives the solver enough steps to converge on this dataset.
# Requires scaled input for numerical stability.
lr_model = LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)
lr_model.fit(X_train_sc, y_train)
lr_score = lr_model.score(X_test_sc, y_test)


# =============================================================================
# 6.  COMPARE ACCURACY OF THE 4 MODELS
# =============================================================================
# Accuracy = (number of correct predictions) / (total predictions)
# Reported on the test set (30,000 unseen records)
results = {
    "Decision Tree"      : dt_score,
    "Naïve Bayes"        : nb_score,
    "KNN (k=5)"          : knn_score,
    "Logistic Regression": lr_score,
}
best_model = max(results, key=results.get)

# Chart 7 – Bar chart comparison of the 4 model accuracies
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

# Chart 8 – Line chart (accuracy trend across models)
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
# 7.  GENERATE PDF REPORT
# =============================================================================
PDF_PATH = os.path.join(REPORT_DIR, "ML_Student_Mental_Health_Report.pdf")

# Each entry: (chart filename, chart title, explanation text)
chart_pages = [
    ("01_target_distribution.png",
     "Chart 1 – Burnout Level Distribution",
     "This bar chart shows the number of students in each burnout category: Low, Medium, and High. "
     "Checking the class balance is the first step before training any classifier — a heavily skewed "
     "distribution can cause models to favour the majority class. A roughly equal distribution, as seen here, "
     "means the models can learn meaningful patterns for all three burnout levels."),

    ("02_age_distribution.png",
     "Chart 2 – Age Distribution",
     "This histogram shows the age range of students in the dataset (17–25 years). "
     "The narrow range is expected for a university-level study population. "
     "Age is kept as a feature because coping strategies and academic pressure tend to differ "
     "as students progress through their studies, which may influence burnout outcomes."),

    ("03_stress_vs_burnout.png",
     "Chart 3 – Stress Level vs Burnout Level",
     "This grouped bar chart crosses two categorical variables: self-reported stress level "
     "(Low / Medium / High) and burnout level. It visually tests whether students who report "
     "high stress are disproportionately represented in the High burnout group. "
     "A strong visible pattern here confirms that stress level is a meaningful predictor for burnout."),

    ("04_cgpa_vs_burnout.png",
     "Chart 4 – CGPA by Burnout Level",
     "These box plots compare the distribution of CGPA (Cumulative Grade Point Average) "
     "across the three burnout groups. The box spans the 25th to 75th percentile, the line "
     "inside is the median, and the whiskers cover the data range. Differences in median CGPA "
     "between groups indicate that academic performance is a useful predictor. "
     "Overlap between groups is natural and is handled by multi-feature models."),

    ("05_correlation_heatmap.png",
     "Chart 5 – Correlation Heatmap",
     "This heatmap shows the Pearson correlation coefficient between every pair of numerical features. "
     "Values near +1 (dark red) indicate a strong positive relationship; values near -1 (dark blue) "
     "indicate a strong negative relationship; values near 0 show little to no linear relationship. "
     "High correlations between features (multicollinearity) can reduce the interpretability of "
     "Logistic Regression coefficients but generally do not affect Decision Tree or Naïve Bayes accuracy."),

    ("06_sleep_vs_burnout.png",
     "Chart 6 – Daily Sleep Hours by Burnout Level",
     "These overlapping histograms compare the number of hours per day students sleep, grouped by "
     "burnout level. If students with High burnout consistently sleep fewer hours, sleep duration is "
     "a strong predictor. The overlap between distributions reflects real-world variability and is "
     "well handled by both probabilistic (Naïve Bayes) and distance-based (KNN) classifiers."),

    ("07_model_accuracy_comparison.png",
     "Chart 7 – Model Accuracy Comparison (Bar Chart)",
     "This bar chart provides a side-by-side comparison of the test-set accuracy achieved by each "
     "of the four trained models. Accuracy is the proportion of correctly classified samples out of "
     "all test samples (30,000 records). The dashed red line marks the best-performing model. "
     "Accuracy is a valid primary metric for this dataset given the roughly balanced class distribution."),

    ("08_model_accuracy_line.png",
     "Chart 8 – Model Accuracy Comparison (Line Chart)",
     "This line chart presents the same accuracy values as Chart 7 but as a connected trend line, "
     "making relative differences between models easier to see at a glance. Each point is annotated "
     "with its exact accuracy score. A steeper slope between two adjacent models indicates a more "
     "significant performance difference."),
]


def pdf_text_page(pdf, title, body_paragraphs):
    """Render a clean, minimal explanation page into the PDF."""
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.patch.set_facecolor("white")
    ax.axis("off")

    # Section title
    ax.text(0.08, 0.90, title,
            transform=ax.transAxes, fontsize=15, fontweight="bold",
            color="#1a1a1a", va="top")

    # Thin rule beneath title
    ax.axhline(y=0.852, xmin=0.08, xmax=0.92, color="#DDDDDD", linewidth=1)

    # Body paragraphs
    # wrap_width controls characters per line; LINE_H is the height step per
    # rendered line (in axes-fraction units); PARA_GAP adds space between blocks.
    WRAP_WIDTH = 94
    LINE_H     = 0.020   # height per rendered text line
    PARA_GAP   = 0.032   # extra gap between paragraphs

    y_cursor = 0.815
    for para in body_paragraphs:
        if para.strip() == "":           # blank string = paragraph spacer
            y_cursor -= PARA_GAP
            continue
        wrapped    = textwrap.fill(para, width=WRAP_WIDTH)
        line_count = len(wrapped.split("\n"))
        ax.text(0.08, y_cursor, wrapped,
                transform=ax.transAxes, fontsize=10.5,
                color="#333333", va="top", linespacing=1.6)
        y_cursor -= LINE_H * line_count + PARA_GAP

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


with PdfPages(PDF_PATH) as pdf:

    # ── Page 1: Minimal title page ─────────────────────────────────────────────
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

    # Each row: (label, value, row_height)
    # label='' means it's a continuation value line with smaller step
    rows = [
        ("Dataset",         "student_mental_health_burnout.csv  ·  150,000 records  ·  20 features  ·  Source: Kaggle",    0.068),
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

    # ── Page 2: Dataset overview explanation ──────────────────────────────────
    pdf_text_page(pdf, "1. Dataset Overview", [
        "The dataset used in this analysis is the Student Mental Health & Burnout dataset sourced from Kaggle. "
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
    ])

    # ── Page 3: X and Y explanation ───────────────────────────────────────────
    pdf_text_page(pdf, "2. Determining X (Features) and Y (Target)", [
        "Y — Target Variable:",
        "burnout_level is the column the models are trained to predict. "
        "It has three possible values: Low, Medium, and High — making this a multi-class classification problem. "
        "The target is Label-Encoded to integers (High = 0, Low = 1, Medium = 2) so scikit-learn can process it.",
        "",
        "X — Feature Variables:",
        "All remaining columns, after removing student_id (a row identifier with no predictive value), "
        "are used as input features. This gives 18 features per student — a combination of raw numerical "
        "measurements and encoded categorical labels.",
        "",
        "Encoding & Scaling:",
        "Categorical columns (gender, course, year, stress_level, sleep_quality, internet_quality) are "
        "converted to integers via Label Encoding. For Decision Tree and Naïve Bayes, features are used "
        "as-is — these algorithms are not affected by feature scale. For KNN and Logistic Regression, "
        "all features are standardised using StandardScaler (mean = 0, standard deviation = 1) to ensure "
        "that features with larger numerical ranges do not dominate the model.",
    ])

    # ── Pages 4–19: Chart page + explanation page for each chart ──────────────
    for fname, chart_title, explanation in chart_pages:
        img_path = os.path.join(OUTPUT_DIR, fname)
        if not os.path.exists(img_path):
            continue
        # Chart page
        img = plt.imread(img_path)
        fig, ax = plt.subplots(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax.imshow(img)
        ax.axis("off")
        fig.text(0.5, 0.01, chart_title, ha="center", va="bottom",
                 fontsize=10, color="#777777", style="italic")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        # Explanation page immediately after the chart
        pdf_text_page(pdf, chart_title, [explanation])

    # ── Last page: Model accuracy summary ─────────────────────────────────────
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
        "Accuracy is used as the sole evaluation metric. No classification reports or confusion matrices "
        "are included in this analysis.",
    ])

    d = pdf.infodict()
    d["Title"]   = "Student Mental Health & Burnout – ML Report"
    d["Author"]  = "ML Pipeline (main.py)"
    d["Subject"] = "Machine Learning Classification"