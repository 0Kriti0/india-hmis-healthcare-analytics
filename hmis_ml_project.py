# -*- coding: utf-8 -*-
"""
District-level Predictive Analytics project on India's HMIS data
(Performance of Key HMIS Indicators across Districts, 2017-18 to 2019-20)

Each model answers its OWN question (different target / different purpose):

  Unit I   Data preparation + EDA (not counted as models)
  Unit II  1. Multiple Linear Regression -> institutional delivery %  (from ANC indicators)
  Unit II  2. Logistic Regression        -> high low-birth-weight district (maternal health)
  Unit III 3. k-Nearest Neighbours       -> high C-section district (delivery-system profile)
  Unit III/VI 4. Decision Tree (baseline) vs Random Forest (bagging)
                                         -> high reported infant mortality
  Unit IV  5. K-Means                    -> types of districts (unsupervised)
  Unit V   6. PCA (technique, not a predictive model) -> structure of the indicators

Run:  python hmis_ml_project.py
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, log_loss, confusion_matrix, silhouette_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
SEED = 42
DATA_PATH = "kritipandey8916_17933662853790064.csv"   # change to your path
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
summary = []                                           # collects the headline metrics


def header(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=130)
    display(fig)
    plt.close(fig)

# =============================================================================
# UNIT I - DATA PREPARATION
# =============================================================================
header("UNIT I  |  1. LOADING DATA")
raw = pd.read_csv(DATA_PATH)
print("Raw shape:", raw.shape)


def col(prefix):
    """Find the one column whose name starts with `prefix` (names are very long)."""
    hit = [c for c in raw.columns if c.startswith(prefix)]
    assert len(hit) == 1, f"'{prefix}' matched {len(hit)} columns"
    return raw[hit[0]]


df = pd.DataFrame({
    "State": raw["State"].str.strip(),
    "District": raw["District"].str.strip(),
    # the file labels years 2017, 2018, 2019; we keep these labels as they are
    "Year": raw["Year"].str.extract(r"(\d{4})")[0].astype(int),
})
df["district_id"] = df["State"] + " | " + df["District"]   # names repeat across states

# ---- indicators reported as percentages in the file ------------------------
PCT = {
    "early_anc": "Registrations For Ante Natal Care (Anc) In The First Trimester",
    "anc4":      "Pregnant Women Received 4 Ante Natal Care (Anc) Check Ups",
    "tt":        "Pregnant Women Received Tetanus-Toxoid Vaccine (Tt2)",
    "ifa":       "Women Received 180 Ifa Tablets",
    "inst_pct":  "Institutional Deliveries (To Total Reported Deliveries)",
    "home_pct":  "Home Deliveries (To Total Reported Deliveries)",
    "csec":      "C-Section Deliveries To Reported Institutional Deliveries",
    "priv_sh":   "Deliveries Conducted At Private Facilities (To Reported Institutional",
    "weighed":   "Newborns Weighed At Birth (To Reported Live Births)",
    "lbw":       "Newborns Having Weight Less Than 25 Kg (To Newborns Weighed",
    "bf1h":      "New Borns Breastfed Within 1 Hour (To Total Live Births)",
    "obst":      "Pregnant Women With Obstetric Complications (Who Were Attended To)",
}
for k, p in PCT.items():
    df[k] = col(p)
df["live_births"] = col("Reported Live Births (UOM")
df["deliveries"] = col("Total Reported Deliveries (UOM")
df["infant_deaths"] = col("Infant Deaths Reported")

# ---- 2. check what each indicator really measures --------------------------
header("UNIT I  |  2. INDICATOR DEFINITION CHECK (numerator / denominator)")
# We rebuild each percentage from the raw counts in the file. If the rebuilt value
# matches the reported value we know the exact denominator.
ANC = col("Women Registered For Ante Natal Care(Anc) (UOM")
INST = col("Institutional Deliveries Includes Both")
LB = df["live_births"]
WEIGHED = col("Newborns Weighed At Birth (UOM")
tt_num = (col("Women Received The Second Dose Of Tetanus-Toxoid Vaccine (Tt2) (UOM")
          + col("Women Received The Dose Of Tetanus-Toxoid Vaccine (Tt) (UOM"))
csec_num = (col("Caesarean (C-Section) Deliveries Conducted At Public Facilities (UOM")
            + col("Caesarean (C-Section) Deliveries Conducted At Private Facilities (UOM"))
CHECKS = [  # (indicator, rebuilt value, denominator in words)
    ("early_anc", col("Women Registered For Ante Natal Care (Anc) Within First Trimester") / ANC, "total ANC registrations"),
    ("anc4",      col("Women Received 4 Or More (Anc) Check-Ups (UOM") / ANC,                  "total ANC registrations"),
    ("tt",        tt_num / ANC,                                                                 "total ANC registrations (TT2 + TT booster)"),
    ("ifa",       col("Pregnant Women Received 180 Iron And Folic Acid Tablets (UOM") / ANC,   "total ANC registrations"),
    ("inst_pct",  INST / df["deliveries"],                                                      "total reported deliveries"),
    ("home_pct",  col("Home Deliveries (UOM") / df["deliveries"],                              "total reported deliveries"),
    ("csec",      csec_num / INST,                                                              "institutional deliveries"),
    ("priv_sh",   col("Deliveries Conducted At Private Institutions (UOM") / INST,             "institutional deliveries"),
    ("weighed",   WEIGHED / LB,                                                                 "reported live births"),
    ("lbw",       col("Newborns Having Weight Less Than 25 Kg (UOM") / WEIGHED,                "newborns weighed at birth"),
    ("bf1h",      col("New Borns Breastfed Within 1 Hour (UOM") / LB,                          "reported live births"),
]
rows = []
for name, rebuilt, denom in CHECKS:
    rebuilt = (rebuilt * 100).replace([np.inf, -np.inf], np.nan)
    ok = (rebuilt - df[name]).abs() < 0.5
    rows.append([name, denom, round(100 * ok[rebuilt.notna()].mean(), 1),
                 round(100 * (df[name] > 100).mean(), 1), round(df[name].max(), 1)])
print(pd.DataFrame(rows, columns=["indicator", "denominator (found by rebuilding)",
                                  "% rows rebuilt exactly", "% rows >100", "max value"]).to_string(index=False))
print("\nHome % + Institutional % = 100 in",
      round(100 * ((df.inst_pct + df.home_pct - 100).abs() < 2).mean(), 1),
      "% of rows -> they are exact complements, so only one of them may be used as a predictor.")
print("Public share + Private share = 100 in",
      round(100 * ((col('Deliveries Conducted At Public Facilities (To Reported Institutional') + df.priv_sh - 100).abs() < 1).mean(), 1),
      "% of rows -> private share alone carries the information.")

# ---- 3. decision for values above 100 --------------------------------------
header("UNIT I  |  3. HANDLING OF VALUES OUTSIDE 0-100 (indicator by indicator)")
# (a) numerator is a SUBSET of the denominator -> a value above 100 is impossible -> invalid
SUBSET_TYPE = ["early_anc", "inst_pct", "home_pct", "csec", "priv_sh", "lbw"]
# (b) numerator counts events that can fall outside the denominator's period/cohort
#     (e.g. 4th ANC visit or 180 IFA tablets completed months after registration, babies
#      weighed/breastfed that are not in the 'reported live births' of that month,
#      complications attended that are not deliveries) -> values >100 are real HMIS
#      artefacts, NOT impossible. They are KEPT as reported; extremes are winsorised.
KEEP_ABOVE_100 = ["anc4", "tt", "ifa", "weighed", "bf1h", "obst"]
for c in df[list(PCT)]:
    neg = int((df[c] < 0).sum())
    df.loc[df[c] < 0, c] = np.nan                              # negative % is always invalid
    over = int((df[c] > 100).sum())
    if c in SUBSET_TYPE:
        df.loc[df[c] > 100, c] = np.nan
        print(f"{c:10s} subset-type  : {neg} negative, {over} above 100 set to missing")
    else:
        print(f"{c:10s} keep >100%   : {neg} negative set to missing, {over} values above 100 KEPT (winsorised later)")

# ---- 4. indicators deliberately NOT used -----------------------------------
print("""
Indicators examined but excluded from modelling:
 * Immunisation % (measles-rubella, BCG, vitamin A, polio): 18-72% of district-years exceed 100%,
   i.e. the problem is the denominator (live births of the month vs children vaccinated), not a few
   typing errors, so the values cannot be read as 'coverage'.
 * Severe-anaemia %: reported value cannot be rebuilt from the counts in the file and ranges up to ~666%.
 * Hypertension % and post-partum check-up %: denominator not stated / not reproducible.""")

# ---- 5. row filter and target construction ---------------------------------
header("UNIT I  |  4. ROW FILTER, INFANT MORTALITY, MISSING VALUES")
n0 = len(df)
df = df[(df.live_births >= 100) & (df.deliveries >= 100)].copy()   # tiny denominators -> unstable %
print(f"Rows kept: {len(df)} of {n0} (districts with <100 live births or <100 deliveries removed)")
df["IMR"] = df.infant_deaths / df.live_births * 1000        # REPORTED (HMIS) infant deaths per 1000 live births
df = df[np.isfinite(df.IMR)].reset_index(drop=True)
print("Districts:", df.district_id.nunique(), "| States/UTs:", df.State.nunique(), "| Rows:", len(df))
print("\nMissing values after cleaning (%):")
print((df[list(PCT) + ["IMR"]].isna().mean() * 100).round(1).sort_values(ascending=False).head(6).to_string())
print("\nReported HMIS IMR (per 1000 live births):")
print(df.IMR.describe().round(2).to_string())
print("District-years reporting zero infant deaths:", int((df.IMR == 0).sum()))

# =============================================================================
# UNIT I - EXPLORATORY DATA ANALYSIS
# =============================================================================
header("UNIT I  |  5. EXPLORATORY DATA ANALYSIS")
pooled = df.groupby("Year").apply(lambda g: g.infant_deaths.sum() / g.live_births.sum() * 1000)
print("Pooled HMIS infant deaths per 1000 live births by year label:\n", pooled.round(1).to_string())
print("\nMedian by year label:\n", df.groupby("Year")[["inst_pct", "csec", "lbw", "IMR"]].median().round(1).to_string())

# Fig 1: distributions
fig, ax = plt.subplots(2, 3, figsize=(12, 6))
for a, (c, t) in zip(ax.ravel(), [("inst_pct", "Institutional deliveries (%)"), ("early_anc", "ANC in first trimester (%)"),
                                  ("anc4", "4+ ANC check-ups (%) [can exceed 100]"), ("csec", "C-section (% of institutional deliveries)"),
                                  ("lbw", "Low birth weight (% of newborns weighed)"), ("IMR", "Reported infant deaths per 1000 live births")]):
    sns.histplot(df[c].dropna(), bins=40, ax=a, color="#3b6ea5")
    a.axvline(df[c].median(), color="red", ls="--", lw=1)
    a.set_title(t, fontsize=10); a.set_xlabel("")
fig.suptitle("Fig 1. Distribution of key indicators (red line = median)")
save(fig, "fig01_distributions.png")

# Fig 2: state-wise institutional delivery
order = df.groupby("State").inst_pct.median().sort_values().index
fig, ax = plt.subplots(figsize=(8, 9))
sns.boxplot(data=df, y="State", x="inst_pct", order=order, ax=ax, color="#9ecae1", fliersize=2)
ax.set_xlabel("Institutional deliveries (%)"); ax.set_title("Fig 2. Institutional deliveries by State/UT (district-years)")
save(fig, "fig02_state_institutional_delivery.png")

# Fig 3: correlation heatmap (home_pct left out: it equals 100 - inst_pct)
corr_cols = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "priv_sh", "csec", "weighed", "bf1h", "obst", "lbw", "IMR"]
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Fig 3. Correlation between modelling indicators")
save(fig, "fig03_correlation.png")

# Fig 4: C-section vs private share
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(df.priv_sh, df.csec, s=8, alpha=0.4, color="#3b6ea5")
ax.set_xlabel("Private facilities' share of institutional deliveries (%)"); ax.set_ylabel("C-section (% of institutional deliveries)")
ax.set_title(f"Fig 4. C-section rate vs private share (r = {df.priv_sh.corr(df.csec):.2f})")
save(fig, "fig04_csection_vs_private.png")

# Fig 5: reported HMIS IMR vs national estimates
SRS_2018, SRS_2019, NFHS5 = 32, 30, 35.2    # SRS Bulletin 2018 & 2019 (RGI); NFHS-5 2019-21 (MoHFW/IIPS)
labels = [f"HMIS pooled\n(year label {y})" for y in pooled.index] + ["SRS 2018", "SRS 2019", "NFHS-5\n2019-21"]
vals = list(pooled.values) + [SRS_2018, SRS_2019, NFHS5]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(labels, vals, color=["#9ecae1"] * len(pooled) + ["#d95f02"] * 3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Infant deaths per 1000 live births")
ax.set_title("Fig 5. Reported HMIS infant mortality vs national survey/SRS estimates")
save(fig, "fig05_hmis_vs_national_imr.png")

# =============================================================================
# TRAIN / TEST SPLIT (shared by all supervised models)
# =============================================================================
# Every district appears 3 times (3 years). A random row split could put the same district in
# both train and test, so we split by DISTRICT.
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df.district_id))
train, test = df.iloc[tr_idx], df.iloc[te_idx]
print(f"\nTrain rows: {len(train)} ({train.district_id.nunique()} districts) | "
      f"Test rows: {len(test)} ({test.district_id.nunique()} districts)")


def prep_features(cols, tr, te):
    """Winsorise at the 1st/99th percentile and fill missing values - limits learnt on TRAIN only."""
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    lo, hi = Xtr.quantile(0.01), Xtr.quantile(0.99)
    Xtr, Xte = Xtr.clip(lo, hi, axis=1), Xte.clip(lo, hi, axis=1)
    med = Xtr.median()
    return Xtr.fillna(med), Xte.fillna(med)


def binary_target(target):
    """1 = above the TRAIN median of the indicator (a 'high' district), else 0."""
    tr, te = train.dropna(subset=[target]), test.dropna(subset=[target])
    cut = tr[target].median()
    return tr, te, (tr[target] > cut).astype(int), (te[target] > cut).astype(int), cut


def clf_metrics(y, pred, prob):
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred), Recall=recall_score(y, pred),
                F1=f1_score(y, pred), AUC=roc_auc_score(y, prob), LogLoss=log_loss(y, prob))


def show_cm(y, pred, ax, title):
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)


# =============================================================================
# UNIT II - MODEL 1: MULTIPLE LINEAR REGRESSION
# Question: how much of institutional delivery % is explained by antenatal-care engagement?
# =============================================================================
header("UNIT II  |  MODEL 1: MULTIPLE LINEAR REGRESSION  (target: institutional delivery %)")
feat1 = ["early_anc", "anc4", "tt", "ifa"]
tr1, te1 = train.dropna(subset=["inst_pct"]), test.dropna(subset=["inst_pct"])
X1tr, X1te = prep_features(feat1, tr1, te1)
y1tr, y1te = tr1.inst_pct, te1.inst_pct

mlr = LinearRegression().fit(X1tr, y1tr)
p1 = mlr.predict(X1te)
mse = mean_squared_error(y1te, p1)
base = np.full(len(y1te), y1tr.mean())                       # 'always predict the training mean'
slr = LinearRegression().fit(X1tr[["anc4"]], y1tr)           # simple regression, one predictor
res = pd.DataFrame({
    "MAE": [mean_absolute_error(y1te, p1), mean_absolute_error(y1te, base), mean_absolute_error(y1te, slr.predict(X1te[["anc4"]]))],
    "MSE": [mse, mean_squared_error(y1te, base), mean_squared_error(y1te, slr.predict(X1te[["anc4"]]))],
    "RMSE": [np.sqrt(mse), np.sqrt(mean_squared_error(y1te, base)), np.sqrt(mean_squared_error(y1te, slr.predict(X1te[["anc4"]])))],
    "R2": [r2_score(y1te, p1), r2_score(y1te, base), r2_score(y1te, slr.predict(X1te[["anc4"]]))],
}, index=["Multiple LR (4 ANC predictors)", "Baseline: training mean", "Simple LR (ANC4 only)"])
print(res.round(3).to_string())
print(f"Test R2 of the training-mean baseline: {r2_score(y1te, base):.3f} (close to, but not exactly, zero because the test mean differs from the training mean)")
coef = pd.DataFrame({"coef (pp change in inst. delivery per +1 pp)": mlr.coef_,
                     "standardised coef": mlr.coef_ * X1tr.std().values / y1tr.std()}, index=feat1)
print("\nIntercept:", round(mlr.intercept_, 2)); print(coef.round(3).to_string())
# sensitivity: what if the >100% ANC values were capped at 100?
mlr_cap = LinearRegression().fit(X1tr.clip(upper=100), y1tr)
r2_cap = r2_score(y1te, mlr_cap.predict(X1te.clip(upper=100)))
print(f"\nSensitivity - ANC values capped at 100: test R2 = {r2_cap:.3f} (kept as reported: {r2_score(y1te, p1):.3f})")
summary.append(["1 Multiple Linear Regression", "Institutional delivery %", "Test R2", r2_score(y1te, p1), f"training-mean baseline test R2 = {r2_score(y1te, base):.3f}"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(y1te, p1, s=10, alpha=0.5); ax[0].plot([y1te.min(), 100], [y1te.min(), 100], "r--", lw=1)
ax[0].set_xlabel("Actual institutional delivery (%)"); ax[0].set_ylabel("Predicted"); ax[0].set_title(f"Test set (R² = {r2_score(y1te, p1):.2f})")
coef["standardised coef"].plot.barh(ax=ax[1], color="#3b6ea5"); ax[1].set_title("Standardised coefficients")
fig.suptitle("Fig 6. Multiple Linear Regression")
save(fig, "fig06_multiple_linear_regression.png")

# =============================================================================
# UNIT II - MODEL 2: LOGISTIC REGRESSION
# Question: which maternal-care factors change the odds that a district has a high LBW share?
# =============================================================================
header("UNIT II  |  MODEL 2: LOGISTIC REGRESSION  (target: high low-birth-weight district)")
feat2 = ["ifa", "early_anc", "anc4", "tt", "inst_pct", "weighed"]
tr2, te2, y2tr, y2te, cut2 = binary_target("lbw")
X2tr, X2te = prep_features(feat2, tr2, te2)
print(f"'High LBW' = LBW above the training median ({cut2:.1f}% of newborns weighed). Share of high in test: {y2te.mean():.2f}")
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X2tr, y2tr)
pr2, pp2 = logit.predict(X2te), logit.predict_proba(X2te)[:, 1]
m2 = clf_metrics(y2te, pr2, pp2)
print("Test metrics:", {k: round(v, 3) for k, v in m2.items()})
print(f"Baseline (always predict majority class) accuracy: {max(y2te.mean(), 1 - y2te.mean()):.3f}")
print("Confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y2te, pr2))
odds = pd.Series(np.exp(logit[-1].coef_[0]), index=feat2).sort_values()
print("\nOdds ratio for a +1 standard-deviation rise in each predictor (>1 raises odds of HIGH LBW):")
print(odds.round(3).to_string())
summary.append(["2 Logistic Regression", "High LBW district", "Test accuracy / AUC", m2["Accuracy"], f"AUC {m2['AUC']:.3f}; baseline acc {max(y2te.mean(), 1 - y2te.mean()):.3f}"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].barh(odds.index, odds.values - 1, left=1, color=["#d95f02" if v > 1 else "#1b9e77" for v in odds.values])
ax[0].axvline(1, color="k", lw=1); ax[0].set_xlabel("Odds ratio per +1 SD"); ax[0].set_title("Odds ratios (high LBW)")
show_cm(y2te, pr2, ax[1], "Confusion matrix (test)")
fig.suptitle("Fig 7. Logistic Regression")
save(fig, "fig07_logistic_regression.png")

# =============================================================================
# UNIT III - MODEL 3: k-NEAREST NEIGHBOURS
# Question: does a district's delivery-system profile predict a high C-section rate?
# =============================================================================
header("UNIT III |  MODEL 3: k-NEAREST NEIGHBOURS  (target: high C-section district)")
# Target  : C-section deliveries / institutional deliveries  (file definition, verified above)
# Features: none of them contains C-section counts or the public/private C-section rates.
#           home_pct is excluded (= 100 - inst_pct).
feat3 = ["priv_sh", "inst_pct", "obst", "early_anc", "anc4", "ifa", "weighed", "bf1h"]
tr3, te3, y3tr, y3te, cut3 = binary_target("csec")
X3tr, X3te = prep_features(feat3, tr3, te3)
print(f"'High C-section' = above the training median ({cut3:.1f}% of institutional deliveries).")
lk = LinearRegression().fit(X3tr[["priv_sh", "inst_pct"]], tr3.csec)
print(f"Leakage check: private share + institutional % explain only R2 = {lk.score(X3tr[['priv_sh', 'inst_pct']], tr3.csec):.2f} "
      "of the C-section % (linear fit) -> they do not determine the target.")
# choose k using TRAINING data only (5-fold, folds keep a district's years together)
gkf = GroupKFold(n_splits=5)
cv = {}
for k in [5, 10, 15, 25, 40]:
    pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k))
    cv[k] = cross_val_score(pipe, X3tr, y3tr, cv=gkf, groups=tr3.district_id).mean()
best_k = max(cv, key=cv.get)
print("Cross-validated accuracy on training districts:", {k: round(float(v), 3) for k, v in cv.items()}, "-> k =", best_k)
knn = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X3tr, y3tr)
pr3, pp3 = knn.predict(X3te), knn.predict_proba(X3te)[:, 1]
m3 = clf_metrics(y3te, pr3, pp3)
print("Test metrics:", {k: round(v, 3) for k, v in m3.items()})
print(f"Baseline (majority class) accuracy: {max(y3te.mean(), 1 - y3te.mean()):.3f}")
print("Confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y3te, pr3))
f_no = [f for f in feat3 if f != "priv_sh"]
knn_no = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X3tr[f_no], y3tr)
acc_no = accuracy_score(y3te, knn_no.predict(X3te[f_no]))
print(f"Ablation - without private share: test accuracy {acc_no:.3f} (with it: {m3['Accuracy']:.3f})")
summary.append(["3 k-Nearest Neighbours", "High C-section district", "Test accuracy / AUC", m3["Accuracy"], f"AUC {m3['AUC']:.3f}; baseline acc {max(y3te.mean(), 1 - y3te.mean()):.3f}; k={best_k}"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].plot(list(cv), list(cv.values()), "o-"); ax[0].set_xlabel("k (neighbours)"); ax[0].set_ylabel("CV accuracy (training districts)")
ax[0].set_title("Choosing k")
show_cm(y3te, pr3, ax[1], f"Confusion matrix (test, k={best_k})")
fig.suptitle("Fig 8. k-Nearest Neighbours")
save(fig, "fig08_knn.png")

# =============================================================================
# UNIT III + VI - MODEL 4: DECISION TREE (baseline) vs RANDOM FOREST (bagging)
# Question: which care indicators flag districts with high REPORTED infant mortality?
# =============================================================================
header("UNIT III/VI |  MODEL 4: DECISION TREE vs RANDOM FOREST  (target: high reported infant mortality)")
feat4 = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "csec", "priv_sh", "lbw", "weighed", "bf1h", "obst"]
tr4, te4, y4tr, y4te, cut4 = binary_target("IMR")
X4tr, X4te = tr4[feat4], te4[feat4]            # trees do not need scaling or winsorising; only missing values are filled
print(f"'High IMR' = reported infant deaths above the training median ({cut4:.1f} per 1000 live births).")
print("No infant-death column is used as a predictor.")
models = {
    "Decision Tree (depth 3, interpretable)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Decision Tree (fully grown)": DecisionTreeClassifier(random_state=SEED),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=SEED),
}
rows, fitted = [], {}
for name, m in models.items():
    pipe = make_pipeline(SimpleImputer(strategy="median"), m).fit(X4tr, y4tr)
    fitted[name] = pipe
    pr, pp = pipe.predict(X4te), pipe.predict_proba(X4te)[:, 1]
    r = clf_metrics(y4te, pr, pp)
    r["Train accuracy"] = accuracy_score(y4tr, pipe.predict(X4tr))
    rows.append(pd.Series(r, name=name))
tab4 = pd.DataFrame(rows)[["Train accuracy", "Accuracy", "Precision", "Recall", "F1", "AUC", "LogLoss"]].rename(columns={"Accuracy": "Test accuracy"})
print(tab4.round(3).to_string())
print(f"Baseline (majority class) test accuracy: {max(y4te.mean(), 1 - y4te.mean()):.3f}")
# the single test split is noisy, so the tree-vs-forest comparison is also checked with 5-fold grouped CV
# using the TRAINING districts only (X4tr, y4tr); the test set is not used here
cv_auc = {}
for name in ["Decision Tree (fully grown)", "Random Forest (200 trees)"]:
    pipe = make_pipeline(SimpleImputer(strategy="median"), models[name].__class__(**models[name].get_params()))
    s = cross_val_score(pipe, X4tr, y4tr, cv=GroupKFold(5), groups=tr4.district_id, scoring="roc_auc")
    cv_auc[name] = (s.mean(), s.std())
    print(f"5-fold grouped CV AUC on training districts - {name}: {s.mean():.3f} +/- {s.std():.3f}")
rf_pipe = fitted["Random Forest (200 trees)"]
print("Random Forest confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y4te, rf_pipe.predict(X4te)))
imp = pd.Series(rf_pipe[-1].feature_importances_, index=feat4).sort_values()
print("\nRandom Forest feature importance (mean decrease in Gini impurity):"); print(imp.sort_values(ascending=False).round(3).to_string())
summary.append(["4a Decision Tree (fully grown, baseline)", "High reported IMR", "Test accuracy / AUC", tab4.loc["Decision Tree (fully grown)", "Test accuracy"],
                f"AUC {tab4.loc['Decision Tree (fully grown)', 'AUC']:.3f}; train acc {tab4.loc['Decision Tree (fully grown)', 'Train accuracy']:.3f}"])
summary.append(["4b Random Forest", "High reported IMR", "Test accuracy / AUC", tab4.loc["Random Forest (200 trees)", "Test accuracy"],
                f"AUC {tab4.loc['Random Forest (200 trees)', 'AUC']:.3f}; CV AUC {cv_auc['Random Forest (200 trees)'][0]:.3f}"])

fig, ax = plt.subplots(figsize=(15, 6.5))
plot_tree(fitted["Decision Tree (depth 3, interpretable)"][-1], feature_names=feat4, class_names=["Low IMR", "High IMR"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Fig 9. Decision Tree (depth 3) - readable rules for high reported infant mortality")
save(fig, "fig09_decision_tree.png")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
tab4[["Train accuracy", "Test accuracy"]].plot.bar(ax=ax[0], rot=15, color=["#bdbdbd", "#3b6ea5"])
ax[0].set_ylim(0.4, 1.02); ax[0].set_title("Train vs test accuracy (variance of a single tree)")
ax[0].set_xticklabels(["Tree\n(depth 3)", "Tree\n(full)", "Random\nForest"], rotation=0)
imp.plot.barh(ax=ax[1], color="#3b6ea5"); ax[1].set_title("Random Forest feature importance")
fig.suptitle("Fig 10. Bagging: Decision Tree vs Random Forest")
save(fig, "fig10_random_forest.png")

# =============================================================================
# UNIT IV - MODEL 5: K-MEANS CLUSTERING (unsupervised)
# Question: what natural service-delivery PROFILES exist among district-year records,
#           and do outcomes (LBW, reported infant mortality) differ between the profiles?
# NOTE: every row is one district in one year, so the same district can fall in different
#       clusters in different years. We are clustering district-year records, not districts.
# =============================================================================
header("UNIT IV |  MODEL 5: K-MEANS CLUSTERING  (district-year service-delivery profiles)")
feat5 = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "priv_sh", "csec", "weighed", "bf1h", "obst"]   # service delivery only
Xc = df[feat5].copy()
Xc = Xc.clip(Xc.quantile(0.01), Xc.quantile(0.99), axis=1)
Xc = Xc.fillna(Xc.median())
Z = StandardScaler().fit_transform(Xc)           # K-Means uses distances, so features must be on one scale
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 0), "silhouette": np.round(sil, 3)}).to_string(index=False))
# k is chosen ONLY by the silhouette score: the highest value among ALL tested k (2..8)
K = list(ks)[int(np.argmax(sil))]
print(f"Chosen k = {K} (highest silhouette among all tested k = 2..8: {max(sil):.3f})")
# Trade-off (for discussion only, NOT used for selection): silhouette rewards well-separated clusters, so a small k
# often wins, while a larger k gives more detailed (often easier to interpret) profiles but lower silhouette.
best_big = max((k for k in ks if k >= 3), key=lambda k: sil[list(ks).index(k)])
print(f"For comparison only: the best silhouette with k >= 3 is k = {best_big} ({sil[list(ks).index(best_big)]:.3f}).")
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
df["cluster"] = km.labels_
# random-initialisation trap: a single random start can end in a worse solution
rand_inertia = [KMeans(n_clusters=K, init="random", n_init=1, random_state=s).fit(Z).inertia_ for s in range(10)]
print(f"Random-initialisation trap (10 single random starts): inertia {min(rand_inertia):.0f} to {max(rand_inertia):.0f}; "
      f"k-means++ with n_init=10: {km.inertia_:.0f}")
prof = df.groupby("cluster")[feat5 + ["lbw", "IMR"]].median().round(1)
prof.insert(0, "n_district_years", df.cluster.value_counts().sort_index())
print("\nMedian profile of each cluster:"); print(prof.to_string())
print("\nMost common States/UTs in each cluster (share of that cluster's district-years):")
for c in range(K):
    s = df[df.cluster == c].State.value_counts(normalize=True).head(4).round(2)
    print(f"  Cluster {c}: " + ", ".join(f"{i} {v:.0%}" for i, v in s.items()))
# clusters describe district-year records, so one district may switch clusters from year to year
n_clusters_per_district = df.groupby("district_id").cluster.nunique()
print(f"\nDistricts whose three years fall in more than one cluster: {(n_clusters_per_district > 1).sum()} of "
      f"{len(n_clusters_per_district)} ({(n_clusters_per_district > 1).mean():.0%}) -> a district can move between profiles over time.")
summary.append(["5 K-Means", "District-year service-delivery profiles (unsupervised)", "Silhouette", sil[list(ks).index(K)], f"k = {K}"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia"); ax[0].set_title("Elbow method")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].set_title("Silhouette score")
fig.suptitle("Fig 11. Choosing the number of clusters (district-year records)")
save(fig, "fig11_kmeans_selection.png")
zprof = df.groupby("cluster")[feat5 + ["lbw", "IMR"]].mean()
zprof = (zprof - df[feat5 + ["lbw", "IMR"]].mean()) / df[feat5 + ["lbw", "IMR"]].std()
fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * K))
sns.heatmap(zprof, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Fig 12. District-year cluster profiles (z-score of cluster mean)\nlast two columns (lbw, IMR) are outcomes and were NOT used for clustering")
save(fig, "fig12_cluster_profiles.png")

# =============================================================================
# UNIT V - TECHNIQUE 6: PCA (dimensionality reduction - not a predictive model)
# Question: what are the major patterns of variation among the service-delivery indicators,
#           how many uncorrelated principal components are needed to describe them, and can clusters be seen in 2-D?
# =============================================================================
header("UNIT V |  PCA  (same standardised indicators as K-Means)")
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
print("Explained variance ratio:", np.round(evr, 3)); print("Cumulative:", np.round(np.cumsum(evr), 3))
n80 = int(np.argmax(np.cumsum(evr) >= 0.80)) + 1
print(f"Uncorrelated principal components needed for 80% of the variance: {n80} (of {len(feat5)} indicators)")
load = pd.DataFrame(pca.components_[:3].T, index=feat5, columns=["PC1", "PC2", "PC3"])
print("\nLoadings of the first three components:"); print(load.round(2).to_string())
pcs = pca.transform(Z)
summary.append(["6 PCA (technique)", "Structure of indicators", "Variance in PC1+PC2", evr[:2].sum(), f"{n80} PCs for 80%"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].bar(range(1, len(evr) + 1), evr, label="each PC"); ax[0].plot(range(1, len(evr) + 1), np.cumsum(evr), "ro-", label="cumulative")
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Explained variance ratio"); ax[0].legend(); ax[0].set_title("Scree plot")
sc = ax[1].scatter(pcs[:, 0], pcs[:, 1], c=df.cluster, cmap="tab10", s=8, alpha=0.6)
ax[1].set_xlabel(f"PC1 ({evr[0]:.0%})"); ax[1].set_ylabel(f"PC2 ({evr[1]:.0%})"); ax[1].set_title("District-years in PC space, coloured by K-Means cluster")
ax[1].legend(*sc.legend_elements(), title="Cluster")
fig.suptitle("Fig 13. Principal Component Analysis")
save(fig, "fig13_pca.png")

# =============================================================================
# SUMMARY
# =============================================================================
header("SUMMARY OF HEADLINE RESULTS (all values come from the run above)")
out = pd.DataFrame(summary, columns=["Model", "Target / purpose", "Metric", "Value", "Note"])
out["Value"] = out["Value"].round(3)
print(out.to_string(index=False))
out.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
print("\nFigures and results_summary.csv saved in:", os.path.abspath(OUT_DIR))
# =============================================================================
# UNIT I - DATA PREPARATION
# =============================================================================
header("UNIT I  |  1. LOADING DATA")
raw = pd.read_csv(DATA_PATH)
print("Raw shape:", raw.shape)


def col(prefix):
    """Find the one column whose name starts with `prefix` (names are very long)."""
    hit = [c for c in raw.columns if c.startswith(prefix)]
    assert len(hit) == 1, f"'{prefix}' matched {len(hit)} columns"
    return raw[hit[0]]


df = pd.DataFrame({
    "State": raw["State"].str.strip(),
    "District": raw["District"].str.strip(),
    # the file labels years 2017, 2018, 2019; we keep these labels as they are
    "Year": raw["Year"].str.extract(r"(\d{4})")[0].astype(int),
})
df["district_id"] = df["State"] + " | " + df["District"]   # names repeat across states

# ---- indicators reported as percentages in the file ------------------------
PCT = {
    "early_anc": "Registrations For Ante Natal Care (Anc) In The First Trimester",
    "anc4":      "Pregnant Women Received 4 Ante Natal Care (Anc) Check Ups",
    "tt":        "Pregnant Women Received Tetanus-Toxoid Vaccine (Tt2)",
    "ifa":       "Women Received 180 Ifa Tablets",
    "inst_pct":  "Institutional Deliveries (To Total Reported Deliveries)",
    "home_pct":  "Home Deliveries (To Total Reported Deliveries)",
    "csec":      "C-Section Deliveries To Reported Institutional Deliveries",
    "priv_sh":   "Deliveries Conducted At Private Facilities (To Reported Institutional",
    "weighed":   "Newborns Weighed At Birth (To Reported Live Births)",
    "lbw":       "Newborns Having Weight Less Than 25 Kg (To Newborns Weighed",
    "bf1h":      "New Borns Breastfed Within 1 Hour (To Total Live Births)",
    "obst":      "Pregnant Women With Obstetric Complications (Who Were Attended To)",
}
for k, p in PCT.items():
    df[k] = col(p)
df["live_births"] = col("Reported Live Births (UOM")
df["deliveries"] = col("Total Reported Deliveries (UOM")
df["infant_deaths"] = col("Infant Deaths Reported")

# ---- 2. check what each indicator really measures --------------------------
header("UNIT I  |  2. INDICATOR DEFINITION CHECK (numerator / denominator)")
# We rebuild each percentage from the raw counts in the file. If the rebuilt value
# matches the reported value we know the exact denominator.
ANC = col("Women Registered For Ante Natal Care(Anc) (UOM")
INST = col("Institutional Deliveries Includes Both")
LB = df["live_births"]
WEIGHED = col("Newborns Weighed At Birth (UOM")
tt_num = (col("Women Received The Second Dose Of Tetanus-Toxoid Vaccine (Tt2) (UOM")
          + col("Women Received The Dose Of Tetanus-Toxoid Vaccine (Tt) (UOM"))
csec_num = (col("Caesarean (C-Section) Deliveries Conducted At Public Facilities (UOM")
            + col("Caesarean (C-Section) Deliveries Conducted At Private Facilities (UOM"))
CHECKS = [  # (indicator, rebuilt value, denominator in words)
    ("early_anc", col("Women Registered For Ante Natal Care (Anc) Within First Trimester") / ANC, "total ANC registrations"),
    ("anc4",      col("Women Received 4 Or More (Anc) Check-Ups (UOM") / ANC,                  "total ANC registrations"),
    ("tt",        tt_num / ANC,                                                                 "total ANC registrations (TT2 + TT booster)"),
    ("ifa",       col("Pregnant Women Received 180 Iron And Folic Acid Tablets (UOM") / ANC,   "total ANC registrations"),
    ("inst_pct",  INST / df["deliveries"],                                                      "total reported deliveries"),
    ("home_pct",  col("Home Deliveries (UOM") / df["deliveries"],                              "total reported deliveries"),
    ("csec",      csec_num / INST,                                                              "institutional deliveries"),
    ("priv_sh",   col("Deliveries Conducted At Private Institutions (UOM") / INST,             "institutional deliveries"),
    ("weighed",   WEIGHED / LB,                                                                 "reported live births"),
    ("lbw",       col("Newborns Having Weight Less Than 25 Kg (UOM") / WEIGHED,                "newborns weighed at birth"),
    ("bf1h",      col("New Borns Breastfed Within 1 Hour (UOM") / LB,                          "reported live births"),
]
rows = []
for name, rebuilt, denom in CHECKS:
    rebuilt = (rebuilt * 100).replace([np.inf, -np.inf], np.nan)
    ok = (rebuilt - df[name]).abs() < 0.5
    rows.append([name, denom, round(100 * ok[rebuilt.notna()].mean(), 1),
                 round(100 * (df[name] > 100).mean(), 1), round(df[name].max(), 1)])
print(pd.DataFrame(rows, columns=["indicator", "denominator (found by rebuilding)",
                                  "% rows rebuilt exactly", "% rows >100", "max value"]).to_string(index=False))
print("\nHome % + Institutional % = 100 in",
      round(100 * ((df.inst_pct + df.home_pct - 100).abs() < 2).mean(), 1),
      "% of rows -> they are exact complements, so only one of them may be used as a predictor.")
print("Public share + Private share = 100 in",
      round(100 * ((col('Deliveries Conducted At Public Facilities (To Reported Institutional') + df.priv_sh - 100).abs() < 1).mean(), 1),
      "% of rows -> private share alone carries the information.")

# ---- 3. decision for values above 100 --------------------------------------
header("UNIT I  |  3. HANDLING OF VALUES OUTSIDE 0-100 (indicator by indicator)")
# (a) numerator is a SUBSET of the denominator -> a value above 100 is impossible -> invalid
SUBSET_TYPE = ["early_anc", "inst_pct", "home_pct", "csec", "priv_sh", "lbw"]
# (b) numerator counts events that can fall outside the denominator's period/cohort
#     (e.g. 4th ANC visit or 180 IFA tablets completed months after registration, babies
#      weighed/breastfed that are not in the 'reported live births' of that month,
#      complications attended that are not deliveries) -> values >100 are real HMIS
#      artefacts, NOT impossible. They are KEPT as reported; extremes are winsorised.
KEEP_ABOVE_100 = ["anc4", "tt", "ifa", "weighed", "bf1h", "obst"]
for c in df[list(PCT)]:
    neg = int((df[c] < 0).sum())
    df.loc[df[c] < 0, c] = np.nan                              # negative % is always invalid
    over = int((df[c] > 100).sum())
    if c in SUBSET_TYPE:
        df.loc[df[c] > 100, c] = np.nan
        print(f"{c:10s} subset-type  : {neg} negative, {over} above 100 set to missing")
    else:
        print(f"{c:10s} keep >100%   : {neg} negative set to missing, {over} values above 100 KEPT (winsorised later)")

# ---- 4. indicators deliberately NOT used -----------------------------------
print("""
Indicators examined but excluded from modelling:
 * Immunisation % (measles-rubella, BCG, vitamin A, polio): 18-72% of district-years exceed 100%,
   i.e. the problem is the denominator (live births of the month vs children vaccinated), not a few
   typing errors, so the values cannot be read as 'coverage'.
 * Severe-anaemia %: reported value cannot be rebuilt from the counts in the file and ranges up to ~666%.
 * Hypertension % and post-partum check-up %: denominator not stated / not reproducible.""")

# ---- 5. row filter and target construction ---------------------------------
header("UNIT I  |  4. ROW FILTER, INFANT MORTALITY, MISSING VALUES")
n0 = len(df)
df = df[(df.live_births >= 100) & (df.deliveries >= 100)].copy()   # tiny denominators -> unstable %
print(f"Rows kept: {len(df)} of {n0} (districts with <100 live births or <100 deliveries removed)")
df["IMR"] = df.infant_deaths / df.live_births * 1000        # REPORTED (HMIS) infant deaths per 1000 live births
df = df[np.isfinite(df.IMR)].reset_index(drop=True)
print("Districts:", df.district_id.nunique(), "| States/UTs:", df.State.nunique(), "| Rows:", len(df))
print("\nMissing values after cleaning (%):")
print((df[list(PCT) + ["IMR"]].isna().mean() * 100).round(1).sort_values(ascending=False).head(6).to_string())
print("\nReported HMIS IMR (per 1000 live births):")
print(df.IMR.describe().round(2).to_string())
print("District-years reporting zero infant deaths:", int((df.IMR == 0).sum()))

# =============================================================================
# UNIT I - EXPLORATORY DATA ANALYSIS
# =============================================================================
header("UNIT I  |  5. EXPLORATORY DATA ANALYSIS")
pooled = df.groupby("Year").apply(lambda g: g.infant_deaths.sum() / g.live_births.sum() * 1000)
print("Pooled HMIS infant deaths per 1000 live births by year label:\n", pooled.round(1).to_string())
print("\nMedian by year label:\n", df.groupby("Year")[["inst_pct", "csec", "lbw", "IMR"]].median().round(1).to_string())

# Fig 1: distributions
fig, ax = plt.subplots(2, 3, figsize=(12, 6))
for a, (c, t) in zip(ax.ravel(), [("inst_pct", "Institutional deliveries (%)"), ("early_anc", "ANC in first trimester (%)"),
                                  ("anc4", "4+ ANC check-ups (%) [can exceed 100]"), ("csec", "C-section (% of institutional deliveries)"),
                                  ("lbw", "Low birth weight (% of newborns weighed)"), ("IMR", "Reported infant deaths per 1000 live births")]):
    sns.histplot(df[c].dropna(), bins=40, ax=a, color="#3b6ea5")
    a.axvline(df[c].median(), color="red", ls="--", lw=1)
    a.set_title(t, fontsize=10); a.set_xlabel("")
fig.suptitle("Fig 1. Distribution of key indicators (red line = median)")
save(fig, "fig01_distributions.png")

# Fig 2: state-wise institutional delivery
order = df.groupby("State").inst_pct.median().sort_values().index
fig, ax = plt.subplots(figsize=(8, 9))
sns.boxplot(data=df, y="State", x="inst_pct", order=order, ax=ax, color="#9ecae1", fliersize=2)
ax.set_xlabel("Institutional deliveries (%)"); ax.set_title("Fig 2. Institutional deliveries by State/UT (district-years)")
save(fig, "fig02_state_institutional_delivery.png")

# Fig 3: correlation heatmap (home_pct left out: it equals 100 - inst_pct)
corr_cols = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "priv_sh", "csec", "weighed", "bf1h", "obst", "lbw", "IMR"]
fig, ax = plt.subplots(figsize=(9, 7.5))
sns.heatmap(df[corr_cols].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 8})
ax.set_title("Fig 3. Correlation between modelling indicators")
save(fig, "fig03_correlation.png")

# Fig 4: C-section vs private share
fig, ax = plt.subplots(figsize=(6.5, 5))
ax.scatter(df.priv_sh, df.csec, s=8, alpha=0.4, color="#3b6ea5")
ax.set_xlabel("Private facilities' share of institutional deliveries (%)"); ax.set_ylabel("C-section (% of institutional deliveries)")
ax.set_title(f"Fig 4. C-section rate vs private share (r = {df.priv_sh.corr(df.csec):.2f})")
save(fig, "fig04_csection_vs_private.png")

# Fig 5: reported HMIS IMR vs national estimates
SRS_2018, SRS_2019, NFHS5 = 32, 30, 35.2    # SRS Bulletin 2018 & 2019 (RGI); NFHS-5 2019-21 (MoHFW/IIPS)
labels = [f"HMIS pooled\n(year label {y})" for y in pooled.index] + ["SRS 2018", "SRS 2019", "NFHS-5\n2019-21"]
vals = list(pooled.values) + [SRS_2018, SRS_2019, NFHS5]
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(labels, vals, color=["#9ecae1"] * len(pooled) + ["#d95f02"] * 3)
for i, v in enumerate(vals):
    ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("Infant deaths per 1000 live births")
ax.set_title("Fig 5. Reported HMIS infant mortality vs national survey/SRS estimates")
save(fig, "fig05_hmis_vs_national_imr.png")

# =============================================================================
# TRAIN / TEST SPLIT (shared by all supervised models)
# =============================================================================
# Every district appears 3 times (3 years). A random row split could put the same district in
# both train and test, so we split by DISTRICT.
gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=SEED)
tr_idx, te_idx = next(gss.split(df, groups=df.district_id))
train, test = df.iloc[tr_idx], df.iloc[te_idx]
print(f"\nTrain rows: {len(train)} ({train.district_id.nunique()} districts) | "
      f"Test rows: {len(test)} ({test.district_id.nunique()} districts)")


def prep_features(cols, tr, te):
    """Winsorise at the 1st/99th percentile and fill missing values - limits learnt on TRAIN only."""
    Xtr, Xte = tr[cols].copy(), te[cols].copy()
    lo, hi = Xtr.quantile(0.01), Xtr.quantile(0.99)
    Xtr, Xte = Xtr.clip(lo, hi, axis=1), Xte.clip(lo, hi, axis=1)
    med = Xtr.median()
    return Xtr.fillna(med), Xte.fillna(med)


def binary_target(target):
    """1 = above the TRAIN median of the indicator (a 'high' district), else 0."""
    tr, te = train.dropna(subset=[target]), test.dropna(subset=[target])
    cut = tr[target].median()
    return tr, te, (tr[target] > cut).astype(int), (te[target] > cut).astype(int), cut


def clf_metrics(y, pred, prob):
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred), Recall=recall_score(y, pred),
                F1=f1_score(y, pred), AUC=roc_auc_score(y, prob), LogLoss=log_loss(y, prob))


def show_cm(y, pred, ax, title):
    sns.heatmap(confusion_matrix(y, pred), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)


# =============================================================================
# UNIT II - MODEL 1: MULTIPLE LINEAR REGRESSION
# Question: how much of institutional delivery % is explained by antenatal-care engagement?
# =============================================================================
header("UNIT II  |  MODEL 1: MULTIPLE LINEAR REGRESSION  (target: institutional delivery %)")
feat1 = ["early_anc", "anc4", "tt", "ifa"]
tr1, te1 = train.dropna(subset=["inst_pct"]), test.dropna(subset=["inst_pct"])
X1tr, X1te = prep_features(feat1, tr1, te1)
y1tr, y1te = tr1.inst_pct, te1.inst_pct

mlr = LinearRegression().fit(X1tr, y1tr)
p1 = mlr.predict(X1te)
mse = mean_squared_error(y1te, p1)
base = np.full(len(y1te), y1tr.mean())                       # 'always predict the training mean'
slr = LinearRegression().fit(X1tr[["anc4"]], y1tr)           # simple regression, one predictor
res = pd.DataFrame({
    "MAE": [mean_absolute_error(y1te, p1), mean_absolute_error(y1te, base), mean_absolute_error(y1te, slr.predict(X1te[["anc4"]]))],
    "MSE": [mse, mean_squared_error(y1te, base), mean_squared_error(y1te, slr.predict(X1te[["anc4"]]))],
    "RMSE": [np.sqrt(mse), np.sqrt(mean_squared_error(y1te, base)), np.sqrt(mean_squared_error(y1te, slr.predict(X1te[["anc4"]])))],
    "R2": [r2_score(y1te, p1), r2_score(y1te, base), r2_score(y1te, slr.predict(X1te[["anc4"]]))],
}, index=["Multiple LR (4 ANC predictors)", "Baseline: training mean", "Simple LR (4+ ANC only)"])
print(res.round(3).to_string())
coef = pd.DataFrame({"coef (pp change in inst. delivery per +1 pp)": mlr.coef_,
                     "standardised coef": mlr.coef_ * X1tr.std().values / y1tr.std()}, index=feat1)
print("\nIntercept:", round(mlr.intercept_, 2)); print(coef.round(3).to_string())
# sensitivity: what if the >100% ANC values were capped at 100?
mlr_cap = LinearRegression().fit(X1tr.clip(upper=100), y1tr)
r2_cap = r2_score(y1te, mlr_cap.predict(X1te.clip(upper=100)))
print(f"\nSensitivity - ANC values capped at 100: test R2 = {r2_cap:.3f} (kept as reported: {r2_score(y1te, p1):.3f})")
summary.append(["1 Multiple Linear Regression", "Institutional delivery %", "Test R2", r2_score(y1te, p1), "baseline R2 = 0"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].scatter(y1te, p1, s=10, alpha=0.5); ax[0].plot([y1te.min(), 100], [y1te.min(), 100], "r--", lw=1)
ax[0].set_xlabel("Actual institutional delivery (%)"); ax[0].set_ylabel("Predicted"); ax[0].set_title(f"Test set (R² = {r2_score(y1te, p1):.2f})")
coef["standardised coef"].plot.barh(ax=ax[1], color="#3b6ea5"); ax[1].set_title("Standardised coefficients")
fig.suptitle("Fig 6. Multiple Linear Regression")
save(fig, "fig06_multiple_linear_regression.png")

# =============================================================================
# UNIT II - MODEL 2: LOGISTIC REGRESSION
# Question: which maternal-care factors change the odds that a district has a high LBW share?
# =============================================================================
header("UNIT II  |  MODEL 2: LOGISTIC REGRESSION  (target: high low-birth-weight district)")
feat2 = ["ifa", "early_anc", "anc4", "tt", "inst_pct", "weighed"]
tr2, te2, y2tr, y2te, cut2 = binary_target("lbw")
X2tr, X2te = prep_features(feat2, tr2, te2)
print(f"'High LBW' = LBW above the training median ({cut2:.1f}% of newborns weighed). Share of high in test: {y2te.mean():.2f}")
logit = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X2tr, y2tr)
pr2, pp2 = logit.predict(X2te), logit.predict_proba(X2te)[:, 1]
m2 = clf_metrics(y2te, pr2, pp2)
print("Test metrics:", {k: round(v, 3) for k, v in m2.items()})
print(f"Baseline (always predict majority class) accuracy: {max(y2te.mean(), 1 - y2te.mean()):.3f}")
print("Confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y2te, pr2))
odds = pd.Series(np.exp(logit[-1].coef_[0]), index=feat2).sort_values()
print("\nOdds ratio for a +1 standard-deviation rise in each predictor (>1 raises odds of HIGH LBW):")
print(odds.round(3).to_string())
summary.append(["2 Logistic Regression", "High LBW district", "Test accuracy / AUC", m2["Accuracy"], f"AUC {m2['AUC']:.3f}; baseline acc {max(y2te.mean(), 1 - y2te.mean()):.3f}"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].barh(odds.index, odds.values - 1, left=1, color=["#d95f02" if v > 1 else "#1b9e77" for v in odds.values])
ax[0].axvline(1, color="k", lw=1); ax[0].set_xlabel("Odds ratio per +1 SD"); ax[0].set_title("Odds ratios (high LBW)")
show_cm(y2te, pr2, ax[1], "Confusion matrix (test)")
fig.suptitle("Fig 7. Logistic Regression")
save(fig, "fig07_logistic_regression.png")

# =============================================================================
# UNIT III - MODEL 3: k-NEAREST NEIGHBOURS
# Question: does a district's delivery-system profile predict a high C-section rate?
# =============================================================================
header("UNIT III |  MODEL 3: k-NEAREST NEIGHBOURS  (target: high C-section district)")
# Target  : C-section deliveries / institutional deliveries  (file definition, verified above)
# Features: none of them contains C-section counts or the public/private C-section rates.
#           home_pct is excluded (= 100 - inst_pct).
feat3 = ["priv_sh", "inst_pct", "obst", "early_anc", "anc4", "ifa", "weighed", "bf1h"]
tr3, te3, y3tr, y3te, cut3 = binary_target("csec")
X3tr, X3te = prep_features(feat3, tr3, te3)
print(f"'High C-section' = above the training median ({cut3:.1f}% of institutional deliveries).")
lk = LinearRegression().fit(X3tr[["priv_sh", "inst_pct"]], tr3.csec)
print(f"Leakage check: private share + institutional % explain only R2 = {lk.score(X3tr[['priv_sh', 'inst_pct']], tr3.csec):.2f} "
      "of the C-section % (linear fit) -> they do not determine the target.")
# choose k using TRAINING data only (5-fold, folds keep a district's years together)
gkf = GroupKFold(n_splits=5)
cv = {}
for k in [5, 10, 15, 25, 40]:
    pipe = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=k))
    cv[k] = cross_val_score(pipe, X3tr, y3tr, cv=gkf, groups=tr3.district_id).mean()
best_k = max(cv, key=cv.get)
print("Cross-validated accuracy on training districts:", {k: round(float(v), 3) for k, v in cv.items()}, "-> k =", best_k)
knn = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X3tr, y3tr)
pr3, pp3 = knn.predict(X3te), knn.predict_proba(X3te)[:, 1]
m3 = clf_metrics(y3te, pr3, pp3)
print("Test metrics:", {k: round(v, 3) for k, v in m3.items()})
print(f"Baseline (majority class) accuracy: {max(y3te.mean(), 1 - y3te.mean()):.3f}")
print("Confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y3te, pr3))
f_no = [f for f in feat3 if f != "priv_sh"]
knn_no = make_pipeline(MinMaxScaler(), KNeighborsClassifier(n_neighbors=best_k)).fit(X3tr[f_no], y3tr)
acc_no = accuracy_score(y3te, knn_no.predict(X3te[f_no]))
print(f"Ablation - without private share: test accuracy {acc_no:.3f} (with it: {m3['Accuracy']:.3f})")
summary.append(["3 k-Nearest Neighbours", "High C-section district", "Test accuracy / AUC", m3["Accuracy"], f"AUC {m3['AUC']:.3f}; baseline acc {max(y3te.mean(), 1 - y3te.mean()):.3f}; k={best_k}"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
ax[0].plot(list(cv), list(cv.values()), "o-"); ax[0].set_xlabel("k (neighbours)"); ax[0].set_ylabel("CV accuracy (training districts)")
ax[0].set_title("Choosing k")
show_cm(y3te, pr3, ax[1], f"Confusion matrix (test, k={best_k})")
fig.suptitle("Fig 8. k-Nearest Neighbours")
save(fig, "fig08_knn.png")

# =============================================================================
# UNIT III + VI - MODEL 4: DECISION TREE (baseline) vs RANDOM FOREST (bagging)
# Question: which care indicators flag districts with high REPORTED infant mortality?
# =============================================================================
header("UNIT III/VI |  MODEL 4: DECISION TREE vs RANDOM FOREST  (target: high reported infant mortality)")
feat4 = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "csec", "priv_sh", "lbw", "weighed", "bf1h", "obst"]
tr4, te4, y4tr, y4te, cut4 = binary_target("IMR")
X4tr, X4te = tr4[feat4], te4[feat4]            # trees do not need scaling or winsorising; only missing values are filled
print(f"'High IMR' = reported infant deaths above the training median ({cut4:.1f} per 1000 live births).")
print("No infant-death column is used as a predictor.")
models = {
    "Decision Tree (depth 3, interpretable)": DecisionTreeClassifier(max_depth=3, random_state=SEED),
    "Decision Tree (fully grown)": DecisionTreeClassifier(random_state=SEED),
    "Random Forest (200 trees)": RandomForestClassifier(n_estimators=200, random_state=SEED),
}
rows, fitted = [], {}
for name, m in models.items():
    pipe = make_pipeline(SimpleImputer(strategy="median"), m).fit(X4tr, y4tr)
    fitted[name] = pipe
    pr, pp = pipe.predict(X4te), pipe.predict_proba(X4te)[:, 1]
    r = clf_metrics(y4te, pr, pp)
    r["Train accuracy"] = accuracy_score(y4tr, pipe.predict(X4tr))
    rows.append(pd.Series(r, name=name))
tab4 = pd.DataFrame(rows)[["Train accuracy", "Accuracy", "Precision", "Recall", "F1", "AUC", "LogLoss"]].rename(columns={"Accuracy": "Test accuracy"})
print(tab4.round(3).to_string())
print(f"Baseline (majority class) test accuracy: {max(y4te.mean(), 1 - y4te.mean()):.3f}")
# the single test split is noisy, so the tree-vs-forest comparison is repeated with 5-fold CV on ALL districts
d4 = df.dropna(subset=["IMR"]); y_all = (d4.IMR > d4.IMR.median()).astype(int)
cv_auc = {}
for name in ["Decision Tree (fully grown)", "Random Forest (200 trees)"]:
    pipe = make_pipeline(SimpleImputer(strategy="median"), models[name].__class__(**models[name].get_params()))
    s = cross_val_score(pipe, d4[feat4], y_all, cv=GroupKFold(5), groups=d4.district_id, scoring="roc_auc")
    cv_auc[name] = (s.mean(), s.std())
    print(f"5-fold grouped CV AUC - {name}: {s.mean():.3f} +/- {s.std():.3f}")
rf_pipe = fitted["Random Forest (200 trees)"]
print("Random Forest confusion matrix [rows = actual Low/High]:\n", confusion_matrix(y4te, rf_pipe.predict(X4te)))
imp = pd.Series(rf_pipe[-1].feature_importances_, index=feat4).sort_values()
print("\nRandom Forest feature importance (mean decrease in Gini impurity):"); print(imp.sort_values(ascending=False).round(3).to_string())
summary.append(["4a Decision Tree (fully grown, baseline)", "High reported IMR", "Test accuracy / AUC", tab4.loc["Decision Tree (fully grown)", "Test accuracy"],
                f"AUC {tab4.loc['Decision Tree (fully grown)', 'AUC']:.3f}; train acc {tab4.loc['Decision Tree (fully grown)', 'Train accuracy']:.3f}"])
summary.append(["4b Random Forest", "High reported IMR", "Test accuracy / AUC", tab4.loc["Random Forest (200 trees)", "Test accuracy"],
                f"AUC {tab4.loc['Random Forest (200 trees)', 'AUC']:.3f}; CV AUC {cv_auc['Random Forest (200 trees)'][0]:.3f}"])

fig, ax = plt.subplots(figsize=(15, 6.5))
plot_tree(fitted["Decision Tree (depth 3, interpretable)"][-1], feature_names=feat4, class_names=["Low IMR", "High IMR"],
          filled=True, rounded=True, fontsize=8, ax=ax)
ax.set_title("Fig 9. Decision Tree (depth 3) - readable rules for high reported infant mortality")
save(fig, "fig09_decision_tree.png")
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
tab4[["Train accuracy", "Test accuracy"]].plot.bar(ax=ax[0], rot=15, color=["#bdbdbd", "#3b6ea5"])
ax[0].set_ylim(0.4, 1.02); ax[0].set_title("Train vs test accuracy (variance of a single tree)")
ax[0].set_xticklabels(["Tree\n(depth 3)", "Tree\n(full)", "Random\nForest"], rotation=0)
imp.plot.barh(ax=ax[1], color="#3b6ea5"); ax[1].set_title("Random Forest feature importance")
fig.suptitle("Fig 10. Bagging: Decision Tree vs Random Forest")
save(fig, "fig10_random_forest.png")

# =============================================================================
# UNIT IV - MODEL 5: K-MEANS CLUSTERING (unsupervised)
# Question: what natural TYPES of districts exist by service delivery, and do outcomes differ?
# =============================================================================
header("UNIT IV |  MODEL 5: K-MEANS CLUSTERING  (district types)")
feat5 = ["early_anc", "anc4", "tt", "ifa", "inst_pct", "priv_sh", "csec", "weighed", "bf1h", "obst"]   # service delivery only
Xc = df[feat5].copy()
Xc = Xc.clip(Xc.quantile(0.01), Xc.quantile(0.99), axis=1)
Xc = Xc.fillna(Xc.median())
Z = StandardScaler().fit_transform(Xc)           # K-Means uses distances, so features must be on one scale
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
    inertia.append(km.inertia_); sil.append(silhouette_score(Z, km.labels_))
print(pd.DataFrame({"k": list(ks), "inertia": np.round(inertia, 0), "silhouette": np.round(sil, 3)}).to_string(index=False))
# k=2 would only split 'better' from 'worse' districts, so we choose the best silhouette among k = 3..6
cand = {k: s for k, s in zip(ks, sil) if 3 <= k <= 6}
K = max(cand, key=cand.get)
print(f"Chosen k = {K} (highest silhouette among k = 3..6: {cand[K]:.3f})")
km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Z)
df["cluster"] = km.labels_
# random-initialisation trap: a single random start can end in a worse solution
rand_inertia = [KMeans(n_clusters=K, init="random", n_init=1, random_state=s).fit(Z).inertia_ for s in range(10)]
print(f"Random-initialisation trap (10 single random starts): inertia {min(rand_inertia):.0f} to {max(rand_inertia):.0f}; "
      f"k-means++ with n_init=10: {km.inertia_:.0f}")
prof = df.groupby("cluster")[feat5 + ["lbw", "IMR"]].median().round(1)
prof.insert(0, "n_district_years", df.cluster.value_counts().sort_index())
print("\nMedian profile of each cluster:"); print(prof.to_string())
print("\nMost common States/UTs in each cluster (share of that cluster's district-years):")
for c in range(K):
    s = df[df.cluster == c].State.value_counts(normalize=True).head(4).round(2)
    print(f"  Cluster {c}: " + ", ".join(f"{i} {v:.0%}" for i, v in s.items()))
summary.append(["5 K-Means", "District types (unsupervised)", "Silhouette", sil[list(ks).index(K)], f"k = {K}"])

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia"); ax[0].set_title("Elbow method")
ax[1].plot(list(ks), sil, "o-", color="green"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Silhouette"); ax[1].set_title("Silhouette score")
fig.suptitle("Fig 11. Choosing the number of clusters")
save(fig, "fig11_kmeans_selection.png")
zprof = df.groupby("cluster")[feat5 + ["lbw", "IMR"]].mean()
zprof = (zprof - df[feat5 + ["lbw", "IMR"]].mean()) / df[feat5 + ["lbw", "IMR"]].std()
fig, ax = plt.subplots(figsize=(10, 3 + 0.5 * K))
sns.heatmap(zprof, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
ax.set_title("Fig 12. Cluster profiles (z-score of cluster mean)\nlast two columns (lbw, IMR) are outcomes and were NOT used for clustering")
save(fig, "fig12_cluster_profiles.png")

# =============================================================================
# UNIT V - TECHNIQUE 6: PCA (dimensionality reduction - not a predictive model)
# Question: how many independent dimensions of performance exist, and can clusters be seen in 2-D?
# =============================================================================
header("UNIT V |  PCA  (same standardised indicators as K-Means)")
pca = PCA().fit(Z)
evr = pca.explained_variance_ratio_
print("Explained variance ratio:", np.round(evr, 3)); print("Cumulative:", np.round(np.cumsum(evr), 3))
n80 = int(np.argmax(np.cumsum(evr) >= 0.80)) + 1
print(f"Components needed for 80% of the variance: {n80} (of {len(feat5)} indicators)")
load = pd.DataFrame(pca.components_[:3].T, index=feat5, columns=["PC1", "PC2", "PC3"])
print("\nLoadings of the first three components:"); print(load.round(2).to_string())
pcs = pca.transform(Z)
summary.append(["6 PCA (technique)", "Structure of indicators", "Variance in PC1+PC2", evr[:2].sum(), f"{n80} PCs for 80%"])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].bar(range(1, len(evr) + 1), evr, label="each PC"); ax[0].plot(range(1, len(evr) + 1), np.cumsum(evr), "ro-", label="cumulative")
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("Explained variance ratio"); ax[0].legend(); ax[0].set_title("Scree plot")
sc = ax[1].scatter(pcs[:, 0], pcs[:, 1], c=df.cluster, cmap="tab10", s=8, alpha=0.6)
ax[1].set_xlabel(f"PC1 ({evr[0]:.0%})"); ax[1].set_ylabel(f"PC2 ({evr[1]:.0%})"); ax[1].set_title("Districts in PC space, coloured by K-Means cluster")
ax[1].legend(*sc.legend_elements(), title="Cluster")
fig.suptitle("Fig 13. Principal Component Analysis")
save(fig, "fig13_pca.png")

# =============================================================================
# SUMMARY
# =============================================================================
header("SUMMARY OF HEADLINE RESULTS (all values come from the run above)")
out = pd.DataFrame(summary, columns=["Model", "Target / purpose", "Metric", "Value", "Note"])
out["Value"] = out["Value"].round(3)
print(out.to_string(index=False))
out.to_csv(os.path.join(OUT_DIR, "results_summary.csv"), index=False)
print("\nFigures and results_summary.csv saved in:", os.path.abspath(OUT_DIR))
