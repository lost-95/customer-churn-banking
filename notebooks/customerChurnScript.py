# %% [markdown]
# # Customer Churn Analysis (Banking Dataset)
#
# ## Executive Summary
# This project analyzes customer churn (`Exited`) in a banking dataset to identify **high-risk segments**, understand **key churn drivers**, and propose **actionable retention initiatives**.
#
# **Key outputs**
# - Baseline churn rate and class balance
# - Churn rate by key customer attributes and behaviors
# - 2–3 high-risk, actionable segments (multivariate)
# - 3 recommendations + measurement plan (KPIs / A/B-test logic)
#
# > Optional (next step): build a baseline churn prediction model to support retention prioritization.

# %% [markdown]
# ## 0) Setup

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 0.1) Load data

# %%
df = pd.read_csv("../data/churn.csv")

# %% [markdown]
# ## 1) Context & business framing
#
# ### Business question
# Which customer segments are more likely to churn, and what patterns can explain churn behavior?
#
# ### Target definition
# `Exited`
# - 1 = customer churned
# - 0 = customer retained
#
# ### Unit of analysis
# 1 row = 1 customer
#
# ### Scope
# - Data checks & data quality
# - Exploratory analysis (univariate → bivariate → multivariate)
# - Segmentation and business recommendations
# - Optional baseline ML model (interpretable first)

# %% [markdown]
# ## 2) Data checks (sanity + integrity)
#
# **Goal:** verify dataset structure, target validity, and basic integrity before analysis.

# %%
print("Shape (rows, cols):", df.shape)
display(df.head(3))

display(df.info())

# Missing values
missing = df.isnull().sum()
print("\nMissing values per column:\n", missing[missing > 0] if (missing > 0).any() else "No missing values")

# Duplicates
dup_rows = df.duplicated().sum()
print("\nDuplicate rows:", dup_rows)

if "CustomerId" in df.columns:
    dup_ids = df["CustomerId"].duplicated().sum()
    print("Duplicate CustomerId:", dup_ids)

# Target validity
assert "Exited" in df.columns, "Target column 'Exited' not found!"
print("\nExited value counts (absolute):\n", df["Exited"].value_counts(dropna=False))
print("\nExited value counts (%):\n", (df["Exited"].value_counts(normalize=True, dropna=False) * 100).round(2))

unique_target = set(df["Exited"].dropna().unique())
print("\nUnique values in Exited:", unique_target)

# ID-like columns
id_like = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
print("\nID-like columns detected:", id_like)

# Basic range checks
def quick_range(col):
    return {"min": df[col].min(), "max": df[col].max()}

for col in ["Age", "Tenure", "CreditScore", "NumOfProducts", "Balance", "EstimatedSalary"]:
    if col in df.columns:
        print(col, quick_range(col))

# Optional spot checks
if "Age" in df.columns:
    print("\nCustomers with Age < 18:", (df["Age"] < 18).sum())
    print("Customers with Age > 100:", (df["Age"] > 100).sum())

if "Tenure" in df.columns:
    print("Tenure < 0:", (df["Tenure"] < 0).sum())
    print("Tenure > 10:", (df["Tenure"] > 10).sum())

# %% [markdown]
# ### Data checks summary
# - Dataset: 10,000 rows, 14 columns (suitable for EDA/segmentation).
# - No missing values detected.
# - No duplicate rows / CustomerId duplicates found.
# - Target `Exited` is clean and binary (0/1).
# - ID-like columns (RowNumber, CustomerId, Surname) will not be used as analytical features.

# %% [markdown]
# ## 3) Target baseline (churn rate)
#
# **Goal:** quantify the baseline churn rate and describe class balance.

# %%
baseline_churn = df["Exited"].mean() * 100
n_total = len(df)
n_churn = int(df["Exited"].sum())
n_ret = n_total - n_churn

print(f"Overall churn rate: {baseline_churn:.2f}% ({n_churn}/{n_total})")
print(f"Class balance: retained={n_ret/n_total*100:.2f}% vs churned={n_churn/n_total*100:.2f}%")

# %% [markdown]
# #### Baseline churn rate & class balance
# **Overall churn rate:** **20.37%** (2,037 churned customers out of 10,000).  
# This indicates a **moderate** churn level (10–30%), with a noticeable class imbalance (**79.63% retained vs 20.37% churned**).
#
# **Industry note (banking):** in many retail banking contexts—especially for “sticky” products (e.g., primary accounts)—a ~20% churn rate can be considered **on the high side**. If the dataset represents a more “optional/commodity” product or a specific segment, this level may be more plausible.
#
# **Why class balance matters (EDA + modeling):**
# - Prefer **churn rate by segment** over raw counts.
# - Avoid the accuracy trap (always predicting “no churn” ≈ 79.6% accuracy).
# - Use **Precision/Recall**, **F1**, and especially **PR-AUC** for modeling.
# - Consider threshold tuning and `class_weight` later, if needed.

# %% [markdown]
# ## 4) Univariate analysis (feature overview — NO target)
#
# **Goal:** understand feature distributions, detect skew/heavy tails, and spot dataset imbalance.
#
# ### Numerical features (no `Exited`)
# - Age, Tenure, CreditScore, Balance, EstimatedSalary  
# Use: histogram + boxplot
#
# ### Categorical features (no `Exited`)
# - Geography, Gender, HasCrCard, IsActiveMember, NumOfProducts  
# Use: frequency table (%) + countplot

# %%
num_cols = ["Age", "Tenure", "CreditScore", "Balance", "EstimatedSalary"]
cat_cols = ["Geography", "Gender", "HasCrCard", "IsActiveMember", "NumOfProducts"]

# %%
# (Optional helper) simple univariate plots
def plot_univariate_numeric(df, col, bins=30):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col], bins=bins, kde=True, ax=ax[0], color="#4C72B0")
    ax[0].set_title(f"{col} distribution")
    sns.boxplot(x=df[col], ax=ax[1], color="#55A868")
    ax[1].set_title(f"{col} spread (boxplot)")
    plt.tight_layout()
    plt.show()

def plot_univariate_categorical(df, col):
    freq = (df[col].value_counts(normalize=True) * 100).round(2)
    display(freq.to_frame("percent"))
    plt.figure(figsize=(7,4))
    order = df[col].value_counts().index
    sns.countplot(data=df, x=col, order=order, color="#C44E52")
    plt.title(f"{col} frequency")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 4.1 Numerical features (univariate)

# %%
for c in num_cols:
    plot_univariate_numeric(df, c)

# %% [markdown]
# ### 4.2 Categorical features (univariate)

# %%
for c in cat_cols:
    plot_univariate_categorical(df, c)

# %% [markdown]
# ## 5) Bivariate analysis (feature vs churn — core insights)
#
# **Goal:** quantify churn differences across segments.  
# **Rule:** always show churn rate (%) + segment size (n).
#
# In this section, we compute churn rate for:
# - Age bands
# - Tenure (0–10)
# - CreditScore bands
# - Balance: 0 vs >0, plus quintiles for Balance >0
# - EstimatedSalary deciles
# - Categorical churn rate: Geography, IsActiveMember, NumOfProducts, Gender, HasCrCard

# %% [markdown]
# ### 5.1 Age → churn (Age bands)

# %%
bins_age = [18, 30, 40, 50, 60, 100]
labels_age = ["18–30", "31–40", "41–50", "51–60", "61+"]

df["AgeBand"] = pd.cut(df["Age"], bins=bins_age, labels=labels_age, include_lowest=True)

age_churn = (df.groupby("AgeBand", observed=True)
               .agg(churn_rate=("Exited", "mean"), n_customers=("Exited", "size"))
               .reset_index())
age_churn["churn_rate_pct"] = (age_churn["churn_rate"] * 100).round(2)

plt.figure(figsize=(8,4))
ax = sns.barplot(data=age_churn, x="AgeBand", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate by age band")
ax.set_xlabel("Age band")
ax.set_ylabel("Churn rate (%)")
for i, row in age_churn.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, age_churn["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** churn increases sharply for customers aged **41–60**, peaking in **51–60** (keep n visible for smaller cohorts).

# %% [markdown]
# ### 5.2 Tenure → churn (0–10)

# %%
tenure_churn = (df.groupby("Tenure", as_index=False)
                  .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size")))
tenure_churn["churn_rate_pct"] = (tenure_churn["churn_rate"] * 100).round(2)

plt.figure(figsize=(9,4))
ax = sns.barplot(data=tenure_churn, x="Tenure", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate by tenure (years)")
ax.set_xlabel("Tenure (years)")
ax.set_ylabel("Churn rate (%)")
for i, row in tenure_churn.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, tenure_churn["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** churn is relatively stable across tenure → weak standalone signal; test interactions with engagement/product variables.

# %% [markdown]
# ### 5.3 CreditScore → churn (domain-inspired bands)

# %%
bins_cs = [349, 499, 599, 699, 799, 850]
labels_cs = ["350–499", "500–599", "600–699", "700–799", "800–850"]

df["CreditScoreBand"] = pd.cut(df["CreditScore"], bins=bins_cs, labels=labels_cs)

cs_churn = (df.groupby("CreditScoreBand", observed=True, as_index=False)
              .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size")))
cs_churn["churn_rate_pct"] = (cs_churn["churn_rate"] * 100).round(2)

plt.figure(figsize=(9,4))
ax = sns.barplot(data=cs_churn, x="CreditScoreBand", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate by credit score band")
ax.set_xlabel("Credit score band")
ax.set_ylabel("Churn rate (%)")
for i, row in cs_churn.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, cs_churn["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** lowest credit score band shows slightly higher churn (~4–5pp); likely a secondary driver.

# %% [markdown]
# ### 5.4 Balance → churn (Balance = 0 vs Balance > 0)

# %%
df["HasBalance"] = (df["Balance"] > 0).astype(int)

bal_churn = (df.groupby("HasBalance", as_index=False)
               .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size")))
bal_churn["churn_rate_pct"] = (bal_churn["churn_rate"] * 100).round(2)
bal_churn["BalanceGroup"] = bal_churn["HasBalance"].map({0:"Balance = 0", 1:"Balance > 0"})

plt.figure(figsize=(7,4))
ax = sns.barplot(data=bal_churn, x="BalanceGroup", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate: Balance = 0 vs Balance > 0")
ax.set_xlabel("")
ax.set_ylabel("Churn rate (%)")
for i, row in bal_churn.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, bal_churn["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** Balance > 0 group has higher churn in this dataset; drill down within Balance > 0.

# %% [markdown]
# ### 5.5 Balance > 0 → churn (quintiles)

# %%
df_pos = df[df["Balance"] > 0].copy()
df_pos["BalanceQuintile"] = pd.qcut(df_pos["Balance"], q=5,
                                   labels=["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"])

bal_q = (df_pos.groupby("BalanceQuintile", observed=True, as_index=False)
           .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size")))
bal_q["churn_rate_pct"] = (bal_q["churn_rate"] * 100).round(2)

plt.figure(figsize=(9,4))
ax = sns.barplot(data=bal_q, x="BalanceQuintile", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate by Balance (quintiles) — Balance > 0 only")
ax.set_xlabel("Balance quintile (low → high)")
ax.set_ylabel("Churn rate (%)")
for i, row in bal_q.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, bal_q["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** balance level may refine churn risk within Balance>0; validate with engagement/product interactions.

# %% [markdown]
# ### 5.6 EstimatedSalary → churn (deciles)

# %%
df["SalaryDecile"] = pd.qcut(df["EstimatedSalary"], q=10, labels=[f"D{i}" for i in range(1, 11)])

sal_churn = (df.groupby("SalaryDecile", observed=True)
               .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size"))
               .reset_index())
sal_churn["churn_rate_pct"] = (sal_churn["churn_rate"] * 100).round(2)

plt.figure(figsize=(9,4))
ax = sns.barplot(data=sal_churn, x="SalaryDecile", y="churn_rate_pct", color="#E67E22")
ax.set_title("Churn rate by EstimatedSalary (deciles)")
ax.set_xlabel("EstimatedSalary decile (low → high)")
ax.set_ylabel("Churn rate (%)")
for i, row in sal_churn.iterrows():
    ax.text(i, row["churn_rate_pct"] + 0.5, f"n={int(row['n_customers'])}", ha="center", fontsize=9)
plt.ylim(0, sal_churn["churn_rate_pct"].max() + 5)
plt.tight_layout()
plt.show()

# %% [markdown]
# **Interpretation:** churn is nearly flat across salary deciles → weak standalone signal.

# %% [markdown]
# ### 5.7 Categorical features → churn (Geography, IsActiveMember, NumOfProducts, Gender, HasCrCard)

# %%
def churn_rate_by_category(df, col, title=None):
    tmp = (df.groupby(col, as_index=False)
             .agg(churn_rate=("Exited","mean"), n_customers=("Exited","size")))
    tmp["churn_rate_pct"] = (tmp["churn_rate"] * 100).round(2)
    tmp = tmp.sort_values("churn_rate_pct", ascending=False)

    plt.figure(figsize=(8,4))
    ax = sns.barplot(data=tmp, x=col, y="churn_rate_pct", color="#E67E22")
    ax.set_title(title or f"Churn rate by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Churn rate (%)")

    for i, row in enumerate(tmp.itertuples()):
        ax.text(i, row.churn_rate_pct + 0.5, f"n={int(row.n_customers)}", ha="center", fontsize=9)

    plt.ylim(0, tmp["churn_rate_pct"].max() + 5)
    plt.tight_layout()
    plt.show()

    display(tmp)

for c in ["Geography", "IsActiveMember", "NumOfProducts", "Gender", "HasCrCard"]:
    churn_rate_by_category(df, c)

# %% [markdown]
# ## 6) Multivariate analysis (segment discovery)
#
# **Goal:** identify 2–3 actionable high-risk segments using interpretable combinations.
#
# Recommended interaction views (heatmaps):
# 1) AgeBand × IsActiveMember
# 2) NumOfProducts × IsActiveMember
# 3) HasBalance × IsActiveMember (or BalanceQuintile × IsActiveMember)

# %%
def churn_heatmap(df, rows, cols, title):
    pivot = df.pivot_table(values="Exited", index=rows, columns=cols, aggfunc="mean")
    plt.figure(figsize=(8,5))
    sns.heatmap((pivot*100).round(1), annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title(title)
    plt.xlabel(cols)
    plt.ylabel(rows)
    plt.tight_layout()
    plt.show()
    return pivot

# 1) AgeBand × IsActiveMember
churn_heatmap(df, "AgeBand", "IsActiveMember", "Churn rate (%) — AgeBand × IsActiveMember")

# 2) NumOfProducts × IsActiveMember
churn_heatmap(df, "NumOfProducts", "IsActiveMember", "Churn rate (%) — NumOfProducts × IsActiveMember")

# 3) HasBalance × IsActiveMember
churn_heatmap(df, "HasBalance", "IsActiveMember", "Churn rate (%) — HasBalance × IsActiveMember")

# %% [markdown]
# ## 7) Key findings (5–7 max)
#
# Write 5–7 findings. Each finding must include:
# - segment/variable
# - churn rate vs baseline (20.37%)
# - sample size (n)
# - interpretation (one line)

# %% [markdown]
# **Template**
# 1) **Finding:** ...
#    **Evidence:** churn = __% vs baseline 20.37% (n=__)
#    **Interpretation:** ...
#
# 2) ...

# %% [markdown]
# ## 8) Recommendations (3 actions + measurement plan)
#
# For each recommendation include:
# - **Who:** segment definition
# - **What:** action
# - **Why:** finding-based rationale
# - **How to measure:** churn rate, retention at 30/60/90, uplift vs control group (A/B)

# %% [markdown]
# **Template**
# 1) **Segment:** ...
#    **Action:** ...
#    **Rationale:** ...
#    **Measurement:** churn rate in segment, 30/60/90-day retention, uplift vs control
#
# 2) ...

# %% [markdown]
# ## 9) Limitations & next steps
#
# ### Limitations
# - Observational dataset: correlation ≠ causation
# - Geography may proxy pricing/product differences
# - Snapshot data: no time dimension (no cohort retention)
#
# ### Next steps
# - Add time-based data (cohort retention, tenure over time) if available
# - Optional: baseline churn model (interpretable first)
# - Build a small dashboard to monitor churn by segment (Power BI)

# %% [markdown]
# ## 10) Optional modeling (baseline, interpretable)
#
# **Goal:** validate drivers and enable prioritization (not to “win Kaggle”).  
# Key principles:
# - Drop ID-like columns
# - Use a pipeline so preprocessing is fit on train only
# - Start with Logistic Regression + evaluate with PR-AUC / Recall / Precision

# %%
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

# Drop ID-like columns if present
drop_cols = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
df_model = df.drop(columns=drop_cols).copy()

X = df_model.drop("Exited", axis=1)
y = df_model["Exited"]

categorical_cols = [c for c in ["Geography", "Gender"] if c in X.columns]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

clf = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("PR-AUC (Average Precision):", average_precision_score(y_test, y_proba))