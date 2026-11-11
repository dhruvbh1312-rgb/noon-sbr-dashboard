"""
Noon SBR analysis: every technique from the 9 Data Driven Growth blogs, run on the synthetic data.

  Blog 1  Know your metrics          -> monthly KPIs, retention, cohorts, COD and stock KPIs
  Blog 2  RFM segmentation           -> K-means + elbow, overall score, segments, COD / price tags
  Blog 3  CLV prediction             -> 3m features -> 6m LTV clusters, tuned classifier
  Blog 4  Churn prediction           -> churn model (Problem 2) and COD refusal-risk model (Problem 1)
  Blog 5  Next purchase day          -> 0-20 / 21-49 / 50+ day classes, model comparison + tuning
  Blog 6  Predicting sales           -> differencing, lag features, adjusted R2, LSTM (or MLP) + tuned GBM
  Blog 7  Market response models     -> conversion / order / revenue uplift, predicted vs real uptick
  Blog 8  Uplift + cost optimisation -> X-Learner, net value CATE, policy comparison
  Blog 9  A/B testing                -> stratified groups, t-test, one-way and two-way ANOVA, sample size

Model back-ends: XGBoost is used when installed, otherwise scikit-learn HistGradientBoosting.
The LSTM (Keras) is used when TensorFlow is installed, otherwise an MLP on the same lag features.

Run:  python src/analysis.py      (after python src/generate_data.py)
Outputs: results/*.csv, results/results.json, charts/*.png
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import brentq
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score, mean_absolute_error,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV, StratifiedKFold, TimeSeriesSplit,
                                     cross_val_score, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA, RES, CH = ROOT / "data", ROOT / "results", ROOT / "charts"
RES.mkdir(exist_ok=True)
CH.mkdir(exist_ok=True)

try:
    from xgboost import XGBClassifier, XGBRegressor  # noqa: F401

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False
try:
    import tensorflow as tf  # noqa: F401

    HAS_TF = True
except Exception:  # pragma: no cover
    HAS_TF = False

FAILED_COST = 25.0          # AED per failed delivery (assumption, see report)
COD_HANDLING = 3.0          # AED cash-collection cost avoided when an order is prepaid
MARGIN = 0.20               # contribution margin on order value used for profit maths
R: dict = {"backend": {"gbm": "XGBoost" if HAS_XGB else "sklearn HistGradientBoosting",
                       "sequence_model": "Keras LSTM" if HAS_TF else "sklearn MLP (LSTM stand-in)"}}
TUNING: list[dict] = []


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def gbm_classifier(**kw):
    if HAS_XGB:
        return XGBClassifier(random_state=SEED, n_jobs=2, eval_metric="logloss", verbosity=0, **kw)
    return HistGradientBoostingClassifier(random_state=SEED, **kw)


def gbm_grid():
    if HAS_XGB:
        return {"max_depth": [3, 4, 5, 6, 8], "min_child_weight": [1, 3, 5], "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "n_estimators": [150, 300, 500], "subsample": [0.7, 0.85, 1.0], "colsample_bytree": [0.7, 0.85, 1.0]}
    return {"max_depth": [3, 4, 5, 6, None], "min_samples_leaf": [10, 20, 40, 80], "learning_rate": [0.03, 0.05, 0.1, 0.2],
            "max_iter": [150, 300, 500], "l2_regularization": [0.0, 0.1, 1.0], "max_leaf_nodes": [15, 31, 63]}


def tune(name, X, y, scoring, cv, n_iter=20, base=None, grid=None):
    """Randomised hyperparameter search; logs default vs tuned CV score."""
    base = base if base is not None else gbm_classifier()
    grid = grid if grid is not None else gbm_grid()
    t0 = time.time()
    default_cv = cross_val_score(base, X, y, scoring=scoring, cv=cv, n_jobs=2).mean()
    rs = RandomizedSearchCV(base, grid, n_iter=n_iter, scoring=scoring, cv=cv, random_state=SEED, n_jobs=2, refit=True)
    rs.fit(X, y)
    TUNING.append({"model": name, "algorithm": type(base).__name__, "metric": scoring, "cv_default": round(default_cv, 4),
                   "cv_tuned": round(rs.best_score_, 4), "best_params": json.dumps(rs.best_params_, default=str),
                   "candidates_tested": n_iter, "seconds": round(time.time() - t0, 1)})
    print(f"   tuned {name}: {scoring} {default_cv:.4f} -> {rs.best_score_:.4f}")
    return rs.best_estimator_, rs


def order_cluster(df, cluster_col, target_col, ascending):
    """Blog 2's order_cluster(): renumber K-means clusters so a higher number is a better customer."""
    means = df.groupby(cluster_col)[target_col].mean().sort_values(ascending=ascending)
    mapping = {old: new for new, old in enumerate(means.index)}
    df[cluster_col] = df[cluster_col].map(mapping)
    return df


def kmeans_col(df, col, k, ascending):
    km = KMeans(n_clusters=k, n_init=10, random_state=SEED)
    df[col + "Cluster"] = km.fit_predict(df[[col]])
    return order_cluster(df, col + "Cluster", col, ascending)


def ttest_power(d, n1, ratio=1.0, alpha=0.05):
    n2 = n1 * ratio
    df = n1 + n2 - 2
    nc = d * np.sqrt(n1 * n2 / (n1 + n2))
    tc = stats.t.ppf(1 - alpha / 2, df)
    p = 1 - stats.nct.cdf(tc, df, nc) + stats.nct.cdf(-tc, df, nc)
    if not np.isfinite(p):  # very large samples: the t distribution is normal
        z = stats.norm.ppf(1 - alpha / 2)
        p = 1 - stats.norm.cdf(z - nc) + stats.norm.cdf(-z - nc)
    return p


def sample_size(d, power=0.8, alpha=0.05, ratio=1.0):
    """Same answer as statsmodels TTestIndPower().solve_power (Blog 9)."""
    return brentq(lambda n: ttest_power(abs(d), n, ratio, alpha) - power, 2, 1e8)


def ols_rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(resid @ resid), beta


def adj_r2(X, y):
    X1 = np.column_stack([np.ones(len(X)), X])
    rss, _ = ols_rss(X1, y)
    tss = float(((y - y.mean()) ** 2).sum())
    n, p = X1.shape
    r2 = 1 - rss / tss
    return 1 - (1 - r2) * (n - 1) / (n - p)


def two_way_anova(df, y, a, b):
    """Type II two-way ANOVA with interaction (equivalent to statsmodels anova_lm(typ=2))."""
    A = pd.get_dummies(df[a], drop_first=True, dtype=float).to_numpy()
    B = pd.get_dummies(df[b], drop_first=True, dtype=float).to_numpy()
    AB = np.column_stack([A[:, [i]] * B[:, [j]] for i in range(A.shape[1]) for j in range(B.shape[1])])
    one = np.ones((len(df), 1))
    Y = df[y].to_numpy(float)
    rss_full, _ = ols_rss(np.hstack([one, A, B, AB]), Y)
    rss_add, _ = ols_rss(np.hstack([one, A, B]), Y)
    rss_noA, _ = ols_rss(np.hstack([one, B]), Y)
    rss_noB, _ = ols_rss(np.hstack([one, A]), Y)
    df_res = len(df) - (1 + A.shape[1] + B.shape[1] + AB.shape[1])
    ms_res = rss_full / df_res
    rows = []
    for name, ss, dfn in [(a, rss_noA - rss_add, A.shape[1]), (b, rss_noB - rss_add, B.shape[1]),
                          (f"{a}:{b}", rss_add - rss_full, AB.shape[1])]:
        F = (ss / dfn) / ms_res
        rows.append({"term": name, "sum_sq": ss, "df": dfn, "F": F, "p_value": 1 - stats.f.cdf(F, dfn, df_res)})
    rows.append({"term": "Residual", "sum_sq": rss_full, "df": df_res, "F": np.nan, "p_value": np.nan})
    return pd.DataFrame(rows)


def savefig(name):
    plt.tight_layout()
    plt.savefig(CH / name, dpi=150)
    plt.close()


def pct(x):
    return round(float(x) * 100, 2)


# ---------------------------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------------------------
cust = pd.read_csv(DATA / "customers.csv", parse_dates=["signup_date"])
orders = pd.read_csv(DATA / "orders.csv.gz", parse_dates=["order_date"])
camp_r = pd.read_csv(DATA / "campaign_retention.csv")
camp_p = pd.read_csv(DATA / "campaign_prepaid.csv")
demand = pd.read_csv(DATA / "daily_demand.csv", parse_dates=["date"])
SNAP = orders.order_date.max()
R["data"] = {"customers": len(cust), "orders": len(orders), "period": [str(orders.order_date.min().date()), str(SNAP.date())],
             "campaign_retention_rows": len(camp_r), "campaign_prepaid_rows": len(camp_p), "daily_demand_rows": len(demand)}


# =============================================================================================
# BLOG 1: KNOW YOUR METRICS
# =============================================================================================
def blog1():
    print("Blog 1: metrics")
    o = orders.copy()
    o["ym"] = o.order_date.dt.to_period("M")
    m = o.groupby("ym").agg(revenue=("revenue_aed", "sum"), active_customers=("customer_id", "nunique"),
                            orders=("order_id", "count"), failed=("failed", "sum"),
                            cod_orders=("payment_method", lambda s: (s == "COD").sum()))
    m["growth_rate_pct"] = m.revenue.pct_change() * 100
    m["avg_revenue_per_order"] = m.revenue / m.orders
    first = o.groupby("customer_id").ym.min().rename("first_ym")
    o = o.join(first, on="customer_id")
    o["type"] = np.where(o.ym == o.first_ym, "New", "Existing")
    nc = o[o.type == "New"].groupby("ym").customer_id.nunique()
    m["new_customers"] = nc
    m["new_customer_ratio_pct"] = m.new_customers / m.active_customers * 100
    # monthly retention rate = retained from previous month / active customers this month (Blog 1)
    act = o.groupby(["customer_id", "ym"]).size().unstack(fill_value=0).gt(0)
    cols = act.columns
    ret = [np.nan]
    for i in range(1, len(cols)):
        retained = (act[cols[i]] & act[cols[i - 1]]).sum()
        ret.append(retained / act[cols[i]].sum() * 100)
    m["retention_rate_pct"] = ret
    cod = o[o.payment_method == "COD"].groupby("ym").agg(cod_failed=("failed", "sum"), cod_n=("order_id", "count"))
    m = m.join(cod)
    m["cod_share_pct"] = m.cod_n / m.orders * 100
    m["cod_failed_rate_pct"] = m.cod_failed / m.cod_n * 100
    m["delivery_success_rate_pct"] = (1 - m.failed / m.orders) * 100
    m["failed_delivery_cost_aed"] = m.failed * FAILED_COST
    m.index = m.index.astype(str)
    m.round(3).to_csv(RES / "monthly_metrics.csv")

    # cohort-based retention
    o["cohort"] = o.first_ym
    o["k"] = (o.ym - o.cohort).apply(lambda x: x.n)
    coh = o.groupby(["cohort", "k"]).customer_id.nunique().unstack()
    coh_pct = coh.div(coh[0], axis=0) * 100
    coh_pct.index = coh_pct.index.astype(str)
    coh_pct.round(2).to_csv(RES / "cohort_retention.csv")

    last12 = m.iloc[-12:]
    prev12 = m.iloc[-24:-12]
    month_k = lambda k: coh_pct[k].dropna().mean() if k in coh_pct else np.nan  # noqa: E731
    R["blog1"] = {
        "north_star": "Monthly Revenue = Active Customers x Orders per Customer x Average Revenue per Order",
        "revenue_last12_aed": round(last12.revenue.sum()), "revenue_prev12_aed": round(prev12.revenue.sum()),
        "yoy_growth_pct": round((last12.revenue.sum() / prev12.revenue.sum() - 1) * 100, 1),
        "avg_monthly_retention_last12_pct": round(last12.retention_rate_pct.mean(), 1),
        "avg_new_customer_ratio_last12_pct": round(last12.new_customer_ratio_pct.mean(), 1),
        "cohort_m1_retention_pct": round(month_k(1), 1), "cohort_m6_retention_pct": round(month_k(6), 1),
        "cohort_m12_retention_pct": round(month_k(12), 1),
        "cod_share_last12_pct": round(last12.cod_n.sum() / last12.orders.sum() * 100, 1),
        "cod_failed_rate_last12_pct": round(last12.cod_failed.sum() / last12.cod_n.sum() * 100, 2),
        "delivery_success_last12_pct": round((1 - last12.failed.sum() / last12.orders.sum()) * 100, 2),
        "failed_deliveries_last12": int(last12.failed.sum()),
        "failed_delivery_cost_last12_aed": round(last12.failed.sum() * FAILED_COST),
        "cod_failed_rate_by_country_pct": (o[o.payment_method == "COD"].groupby("country").failed.mean() * 100).round(2).to_dict(),
    }
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(m.index, m.revenue / 1e6, color="#1f5fbf")
    ax[0].set_title("Monthly revenue (AED m)")
    ax[0].set_xticks(m.index[::4])
    ax[0].tick_params(axis="x", rotation=45)
    ax[1].plot(m.index, m.cod_failed_rate_pct, color="#d0453a")
    ax[1].set_title("COD failed-delivery rate (%)")
    ax[1].set_xticks(m.index[::4])
    ax[1].tick_params(axis="x", rotation=45)
    savefig("blog1_metrics.png")


# =============================================================================================
# BLOG 2: RFM SEGMENTATION
# =============================================================================================
def rfm_table(o, snap):
    g = o.groupby("customer_id").agg(last=("order_date", "max"), Frequency=("order_id", "count"),
                                     Revenue=("revenue_aed", "sum"))
    g["Recency"] = (snap - g["last"]).dt.days
    return g.drop(columns="last")


def blog2():
    print("Blog 2: RFM")
    o = orders[orders.order_date > SNAP - pd.Timedelta(days=365)]
    g = rfm_table(o, SNAP)
    # elbow method on Recency (as in the blog)
    inertia = {k: KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(g[["Recency"]]).inertia_ for k in range(1, 11)}
    g = kmeans_col(g, "Recency", 4, ascending=False)
    g = kmeans_col(g, "Frequency", 4, ascending=True)
    g = kmeans_col(g, "Revenue", 4, ascending=True)
    g["OverallScore"] = g.RecencyCluster + g.FrequencyCluster + g.RevenueCluster
    g["Segment"] = np.select([g.OverallScore <= 2, g.OverallScore <= 4], ["Low-Value", "Mid-Value"], "High-Value")
    extra = o.groupby("customer_id").agg(discount_share=("discount_used", "mean"),
                                         cod_share=("payment_method", lambda s: (s == "COD").mean()))
    codf = o[o.payment_method == "COD"].groupby("customer_id").agg(cod_orders=("order_id", "count"), cod_failed=("failed", "sum"))
    g = g.join(extra).join(codf).fillna({"cod_orders": 0, "cod_failed": 0})
    # COD reliability score (Bayesian-smoothed success rate; prior = 4 orders at the network average)
    prior = orders[orders.payment_method == "COD"].failed.mean()
    g["cod_reliability"] = 1 - (g.cod_failed + 4 * prior) / (g.cod_orders + 4)
    g = g.join(cust.set_index("customer_id")[["country", "member"]])
    # Problem 2 behavioural segments
    g["P2_segment"] = np.select(
        [(g.Recency > 90) & (g.Segment != "Low-Value"),
         g.discount_share >= 0.6,
         (g.member == 1) | ((g.discount_share < 0.35) & (g.Frequency >= g.Frequency.median())),
         g.Segment == "High-Value"],
        ["At-Risk", "Deal Hunters", "Convenience Buyers", "Loyal Value Seekers"], "Occasional")
    g.round(4).to_csv(RES / "rfm_segments.csv")
    seg = g.groupby("Segment").agg(customers=("Recency", "size"), recency=("Recency", "mean"),
                                   frequency=("Frequency", "mean"), revenue=("Revenue", "mean"),
                                   revenue_total=("Revenue", "sum"), cod_reliability=("cod_reliability", "mean"))
    seg["revenue_share_pct"] = seg.revenue_total / seg.revenue_total.sum() * 100
    p2 = g.groupby("P2_segment").agg(customers=("Recency", "size"), revenue=("Revenue", "mean"),
                                     discount_share=("discount_share", "mean"))
    R["blog2"] = {"elbow_inertia": {k: round(v / 1e6, 2) for k, v in inertia.items()}, "k_used": 4,
                  "segments": seg.round(2).reset_index().to_dict("records"),
                  "p2_segments": p2.round(3).reset_index().to_dict("records"),
                  "customers_scored": len(g)}
    plt.figure(figsize=(5, 3.3))
    plt.plot(list(inertia), [v / 1e6 for v in inertia.values()], marker="o")
    plt.title("Elbow method: Recency inertia (millions)")
    plt.xlabel("k")
    savefig("blog2_elbow.png")
    return g


# =============================================================================================
# BLOG 3: CLV PREDICTION
# =============================================================================================
def blog3():
    print("Blog 3: CLV")
    f_start, f_end, l_end = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"), pd.Timestamp("2026-09-30")
    o3 = orders[(orders.order_date >= f_start) & (orders.order_date <= f_end)]
    o6 = orders[(orders.order_date > f_end) & (orders.order_date <= l_end)]
    g = rfm_table(o3, f_end)
    g = kmeans_col(g, "Recency", 4, False)
    g = kmeans_col(g, "Frequency", 4, True)
    g = kmeans_col(g, "Revenue", 4, True)
    g["OverallScore"] = g.RecencyCluster + g.FrequencyCluster + g.RevenueCluster
    g["Segment"] = np.select([g.OverallScore <= 2, g.OverallScore <= 4], ["Low", "Mid"], "High")
    ex = o3.groupby("customer_id").agg(discount_share=("discount_used", "mean"),
                                       cod_share=("payment_method", lambda s: (s == "COD").mean()),
                                       aov=("order_value_aed", "mean"), n_categories=("category", "nunique"))
    g = g.join(ex).join(cust.set_index("customer_id")[["member", "country"]])
    g["m6_LTV"] = o6.groupby("customer_id").revenue_aed.sum()
    g["m6_LTV"] = g.m6_LTV.fillna(0)
    g = g[g.m6_LTV < g.m6_LTV.quantile(0.99)]  # remove outliers, as in the blog
    km = KMeans(n_clusters=3, n_init=10, random_state=SEED)
    g["LTVCluster"] = km.fit_predict(g[["m6_LTV"]])
    g = order_cluster(g, "LTVCluster", "m6_LTV", True)
    clusters = g.groupby("LTVCluster").m6_LTV.agg(["count", "mean", "min", "max"]).round(1)
    X = pd.get_dummies(g.drop(columns=["m6_LTV", "LTVCluster"]), columns=["Segment", "country"], dtype=float)
    y = g.LTVCluster
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    base = gbm_classifier().fit(Xtr, ytr)
    acc_default = accuracy_score(yte, base.predict(Xte))
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best, _ = tune("CLV segment classifier (Blog 3)", Xtr, ytr, "accuracy", cv, n_iter=16)
    pred = best.predict(Xte)
    rep = classification_report(yte, pred, output_dict=True)
    R["blog3"] = {"feature_window": "2026-01-01 to 2026-03-31", "ltv_window": "2026-04-01 to 2026-09-30",
                  "customers": len(g), "ltv_clusters": clusters.reset_index().to_dict("records"),
                  "benchmark_accuracy_pct": pct(yte.value_counts(normalize=True).max()),
                  "test_accuracy_default_pct": pct(acc_default), "test_accuracy_tuned_pct": pct(accuracy_score(yte, pred)),
                  "macro_f1_tuned": round(f1_score(yte, pred, average="macro"), 3),
                  "per_class": {k: {m: round(v[m], 3) for m in ["precision", "recall", "f1-score"]} for k, v in rep.items() if k in ["0", "1", "2"]}}
    g["pred_LTVCluster"] = best.predict(X)
    g[["Recency", "Frequency", "Revenue", "OverallScore", "Segment", "m6_LTV", "LTVCluster", "pred_LTVCluster"]].to_csv(RES / "clv_predictions.csv")


# =============================================================================================
# BLOG 4: CHURN (Problem 2) and REFUSAL RISK (Problem 1)
# =============================================================================================
def customer_features(o, snap):
    g = rfm_table(o, snap)
    o = o.sort_values(["customer_id", "order_date"])
    gaps = o.groupby("customer_id").order_date.diff().dt.days
    o = o.assign(gap=gaps)
    a = o.groupby("customer_id").agg(avg_gap=("gap", "mean"), std_gap=("gap", "std"), discount_share=("discount_used", "mean"),
                                     cod_share=("payment_method", lambda s: (s == "COD").mean()), failed=("failed", "sum"),
                                     aov=("order_value_aed", "mean"), n_cat=("category", "nunique"))
    last90 = o[o.order_date > snap - pd.Timedelta(days=90)].groupby("customer_id").size()
    prev90 = o[(o.order_date <= snap - pd.Timedelta(days=90)) & (o.order_date > snap - pd.Timedelta(days=180))].groupby("customer_id").size()
    g = g.join(a)
    g["orders_last90"] = last90
    g["orders_prev90"] = prev90
    g = g.fillna({"orders_last90": 0, "orders_prev90": 0, "avg_gap": 365, "std_gap": 0})
    g["trend_90"] = g.orders_last90 - g.orders_prev90
    g["recency_over_gap"] = g.Recency / g.avg_gap.clip(lower=1)
    c = cust.set_index("customer_id")
    g = g.join(c[["member", "country", "acq_channel", "signup_date"]])
    g["tenure_days"] = (snap - g.signup_date).dt.days
    return pd.get_dummies(g.drop(columns=["signup_date"]), columns=["country", "acq_channel"], dtype=float)


def blog4_churn():
    print("Blog 4a: churn")
    snap = pd.Timestamp("2026-06-30")
    hist = orders[(orders.order_date <= snap) & (orders.order_date > snap - pd.Timedelta(days=365))]
    act = hist[hist.order_date > snap - pd.Timedelta(days=180)].customer_id.unique()
    X = customer_features(hist[hist.customer_id.isin(act)], snap)
    fut = orders[(orders.order_date > snap) & (orders.order_date <= snap + pd.Timedelta(days=90))].customer_id.unique()
    y = (~X.index.isin(fut)).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=SEED, stratify=y)
    dt = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)  # blog's baseline
    p_dt = dt.predict(Xte)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best, _ = tune("Churn model (Blog 4)", Xtr, ytr, "roc_auc", cv, n_iter=16)
    proba = best.predict_proba(Xte)[:, 1]
    # threshold chosen on training out-of-fold scores to maximise F1 (handles class imbalance)
    oof = np.zeros(len(Xtr))
    for tr, va in cv.split(Xtr, ytr):
        m = gbm_classifier(**_strip(best.get_params())).fit(Xtr.iloc[tr], ytr[tr])
        oof[va] = m.predict_proba(Xtr.iloc[va])[:, 1]
    ths = np.linspace(0.1, 0.9, 81)
    th = ths[np.argmax([f1_score(ytr, oof >= t) for t in ths])]
    p = (proba >= th).astype(int)
    order = np.argsort(-proba)
    top20 = order[: int(0.2 * len(order))]
    R["blog4_churn"] = {
        "definition": "Active Jan-Jun 2026 (snapshot 30 Jun 2026); churned = no order 1 Jul to 28 Sep 2026",
        "customers": len(X), "churn_rate_pct": pct(y.mean()),
        "decision_tree_depth3": {"precision": round(precision_score(yte, p_dt), 3), "recall": round(recall_score(yte, p_dt), 3),
                                 "f1": round(f1_score(yte, p_dt), 3), "accuracy_pct": pct(accuracy_score(yte, p_dt))},
        "tuned_gbm": {"roc_auc": round(roc_auc_score(yte, proba), 3), "threshold": round(float(th), 2),
                      "precision": round(precision_score(yte, p), 3), "recall": round(recall_score(yte, p), 3),
                      "f1": round(f1_score(yte, p), 3), "accuracy_pct": pct(accuracy_score(yte, p))},
        "majority_baseline_accuracy_pct": pct(max(yte.mean(), 1 - yte.mean())),
        "top20pct_capture_of_churners_pct": pct(yte[top20].sum() / yte.sum()),
    }
    fi = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False).head(6)
    R["blog4_churn"]["decision_tree_top_features"] = fi.round(3).to_dict()
    sc = pd.DataFrame({"churn_prob": best.predict_proba(X)[:, 1], "churned": y}, index=X.index)
    sc.round(4).to_csv(RES / "churn_scores.csv")
    return sc


def _strip(params):
    keep = gbm_grid().keys()
    return {k: v for k, v in params.items() if k in keep}


def refusal_features(o):
    o = o.sort_values(["customer_id", "order_date"]).copy()
    o["n_prev_orders"] = o.groupby("customer_id").cumcount()
    o["days_since_prev"] = o.groupby("customer_id").order_date.diff().dt.days.fillna(-1)
    o["log_value"] = np.log(o.order_value_aed)
    o["late_hour"] = ((o.order_hour >= 22) | (o.order_hour <= 2)).astype(int)
    c = cust.set_index("customer_id")[["member", "card_saved"]]
    o = o.join(c, on="customer_id")
    cod = o[o.payment_method == "COD"].copy()
    X = cod[["first_order", "prior_refusals", "n_prev_orders", "days_since_prev", "log_value", "order_hour", "late_hour",
             "discount_used", "member", "card_saved", "address_quality", "country", "category"]]
    X = pd.get_dummies(X, columns=["address_quality", "country", "category"], dtype=float)
    return cod, X


def blog4_refusal():
    print("Blog 4b: COD refusal risk")
    cod, X = refusal_features(orders)
    y = cod.failed.to_numpy()
    split = pd.Timestamp("2026-05-01")
    tr = (cod.order_date < split).to_numpy() & (cod.order_date >= pd.Timestamp("2025-01-01")).to_numpy()
    te = (cod.order_date >= split).to_numpy()
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    dt = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Xtr, ytr)
    p_dt = dt.predict_proba(Xte)[:, 1]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    # tune on a 60k sample for speed
    idx = np.random.RandomState(SEED).choice(len(Xtr), min(60_000, len(Xtr)), replace=False)
    best, _ = tune("COD refusal-risk model (Blog 4 method)", Xtr.iloc[idx], ytr[idx], "roc_auc", cv, n_iter=14)
    best.fit(Xtr, ytr)
    proba = best.predict_proba(Xte)[:, 1]
    test = cod[te].copy()
    test["risk"] = proba
    q60, q85 = np.quantile(proba, [0.60, 0.85])
    test["tier"] = np.select([proba >= q85, proba >= q60], ["High", "Medium"], "Low")
    dec = pd.qcut(proba, 10, labels=False, duplicates="drop")
    lift = test.assign(decile=10 - dec).groupby("decile").agg(orders=("failed", "size"), failed_rate=("failed", "mean"))
    lift["capture_pct"] = test.assign(decile=10 - dec).groupby("decile").failed.sum().cumsum() / test.failed.sum() * 100
    tiers = test.groupby("tier").agg(orders=("failed", "size"), failed=("failed", "sum"), failed_rate=("failed", "mean"))
    R["blog4_refusal"] = {
        "train_period": "2025-01-01 to 2026-04-30", "test_period": "2026-05-01 to 2026-10-31",
        "cod_orders_test": int(len(test)), "failed_rate_test_pct": pct(yte.mean()),
        "decision_tree_auc": round(roc_auc_score(yte, p_dt), 3), "tuned_gbm_auc": round(roc_auc_score(yte, proba), 3),
        "top10pct_failed_rate_pct": pct(lift.failed_rate.iloc[0]), "top10pct_capture_pct": round(lift.capture_pct.iloc[0], 1),
        "top30pct_capture_pct": round(lift.capture_pct.iloc[2], 1),
        "tiers": tiers.assign(failed_rate=lambda d: (d.failed_rate * 100).round(2)).reset_index().to_dict("records"),
        "lift_table": lift.round(4).reset_index().to_dict("records"),
    }
    test[["order_id", "customer_id", "order_date", "country", "city", "order_value_aed", "address_quality", "first_order",
          "prior_refusals", "failed", "risk", "tier"]].round(4).to_csv(RES / "cod_risk_scores.csv", index=False)
    plt.figure(figsize=(5.5, 3.3))
    plt.bar(lift.index.astype(str), lift.failed_rate * 100, color="#d0453a")
    plt.axhline(yte.mean() * 100, ls="--", color="k", lw=1)
    plt.title("Failed-delivery rate by risk decile (1 = riskiest)")
    plt.ylabel("%")
    savefig("blog4_refusal_deciles.png")
    return test


# =============================================================================================
# BLOG 8 (Problem 1): who should get the prepaid cashback?
# =============================================================================================
def x_learner(X, T, Y, Xpred):
    """Two-arm X-Learner (Kunzel et al.) built from gradient boosting models."""
    reg = lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250, random_state=SEED)  # noqa: E731
    clf = lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, random_state=SEED)  # noqa: E731
    m0 = clf().fit(X[T == 0], Y[T == 0])
    m1 = clf().fit(X[T == 1], Y[T == 1])
    d1 = Y[T == 1] - m0.predict_proba(X[T == 1])[:, 1]
    d0 = m1.predict_proba(X[T == 0])[:, 1] - Y[T == 0]
    t1 = reg().fit(X[T == 1], d1)
    t0 = reg().fit(X[T == 0], d0)
    g = 0.5  # randomised 50/50, so the propensity is known
    cate = g * t0.predict(Xpred) + (1 - g) * t1.predict(Xpred)
    return cate, m0.predict_proba(Xpred)[:, 1], m1.predict_proba(Xpred)[:, 1]


def blog8_prepaid():
    print("Blog 8a: prepaid cashback uplift")
    d = camp_p.copy()
    X = pd.get_dummies(d[["country", "address_quality", "card_saved", "frequency", "cod_share", "past_failed_deliveries",
                          "avg_order_value_aed"]], columns=["country", "address_quality"], dtype=float)
    T, Y = d.treatment.to_numpy(), d.prepaid_next_order.to_numpy()
    tr, te = train_test_split(np.arange(len(d)), test_size=0.4, random_state=SEED, stratify=T * 2 + Y)
    cate, p0, p1 = x_learner(X.iloc[tr], T[tr], Y[tr], X.iloc[te])
    prior = orders[orders.payment_method == "COD"].failed.mean()
    p_fail = ((d.past_failed_deliveries + 4 * prior) / (d.frequency * d.cod_share + 4)).to_numpy()[te]
    v = p_fail * FAILED_COST + COD_HANDLING  # AED saved when one COD order becomes prepaid
    s, c = 10.0, 0.10  # cashback paid on conversion, message cost
    net = cate * v - s * p1 - c
    Tt, Yt = T[te], Y[te]
    realised = Yt * v - Tt * (Yt * s + c)  # value actually created by each test customer

    def policy_value(treat):
        match = (Tt == treat)
        return float((realised * match / 0.5).mean())  # inverse propensity weighting (p = 0.5)

    rnd = np.random.RandomState(SEED).rand(len(te)) < 0.5
    pols = {"No one": np.zeros(len(te), bool), "Everyone": np.ones(len(te), bool), "Random 50%": rnd,
            "Uplift (CATE > 0)": cate > 0, "Net value CATE > 0": net > 0}
    base = policy_value(np.zeros(len(te), int))
    out = {k: {"targeted_pct": pct(v_.mean()), "value_per_customer_aed": round(policy_value(v_.astype(int)), 3),
               "incremental_vs_no_one_aed": round(policy_value(v_.astype(int)) - base, 3)} for k, v_ in pols.items()}
    ate = Yt[Tt == 1].mean() - Yt[Tt == 0].mean()
    sel = net > 0
    R["blog8_prepaid"] = {"customers": len(d), "test_customers": len(te), "ate_pct_points": pct(ate),
                          "prepaid_rate_control_pct": pct(Yt[Tt == 0].mean()), "prepaid_rate_treated_pct": pct(Yt[Tt == 1].mean()),
                          "cashback_aed": s, "message_cost_aed": c, "policies": out,
                          "card_saved_share_of_targeted_pct": pct(d.card_saved.to_numpy()[te][sel].mean()) if sel.any() else 0,
                          "avg_cate_targeted_pct_points": pct(cate[sel].mean()) if sel.any() else 0}
    return float(cate[sel].mean()) if sel.any() else 0.0, float(sel.mean())


# =============================================================================================
# Problem 1: predicted improvement + A/B pilot (Blog 9)
# =============================================================================================
def problem1_impact(test, persuadable_uplift, persuadable_share):
    print("Problem 1: impact simulation")
    CONFIRM_EFFECT = 0.30   # WhatsApp confirmation + map-pin check cuts failure probability by 30% (assumption)
    MSG_COST = 0.15
    PREPAID_FAIL = 0.012
    rs = np.random.RandomState(SEED)
    t = test.copy()
    # expected failure after the policy, per order
    exp_after = t.failed.astype(float).copy()
    med = t.tier == "Medium"
    high = t.tier == "High"
    exp_after[med] = t.failed[med] * (1 - CONFIRM_EFFECT)
    # High tier: targeted cashback converts persuadables to prepaid; everyone else gets confirmation
    switch = persuadable_uplift * persuadable_share
    exp_after[high] = t.failed[high] * ((1 - switch) * (1 - CONFIRM_EFFECT)) + switch * PREPAID_FAIL
    msgs = (med | high).sum()
    cashback = high.sum() * switch * 10.0
    months = (t.order_date.max() - t.order_date.min()).days / 30.4
    failed_before, failed_after = t.failed.sum(), exp_after.sum()
    saving = (failed_before - failed_after) * FAILED_COST - msgs * MSG_COST - cashback
    R["problem1_impact"] = {
        "assumptions": {"confirmation_effect": CONFIRM_EFFECT, "message_cost_aed": MSG_COST,
                        "prepaid_failure_rate": PREPAID_FAIL, "failed_delivery_cost_aed": FAILED_COST,
                        "high_tier_switch_to_prepaid_pct": pct(switch)},
        "months_in_test": round(months, 1), "cod_orders": int(len(t)),
        "failed_before": int(failed_before), "failed_after_expected": round(float(failed_after)),
        "failed_rate_before_pct": pct(failed_before / len(t)), "failed_rate_after_pct": pct(failed_after / len(t)),
        "relative_reduction_pct": pct(1 - failed_after / failed_before),
        "net_saving_test_period_aed": round(float(saving)), "net_saving_annualised_aed": round(float(saving) * 12 / months),
    }
    sens = []
    for eff in [0.15, 0.30, 0.45]:
        ea = t.failed.astype(float).copy()
        ea[med] = t.failed[med] * (1 - eff)
        ea[high] = t.failed[high] * ((1 - switch) * (1 - eff)) + switch * PREPAID_FAIL
        sv = (failed_before - ea.sum()) * FAILED_COST - msgs * MSG_COST - cashback
        sens.append({"confirmation_effect_pct": int(eff * 100), "failed_rate_after_pct": pct(ea.sum() / len(t)),
                     "relative_reduction_pct": pct(1 - ea.sum() / failed_before), "net_saving_annualised_aed": round(float(sv) * 12 / months)})
    R["problem1_impact"]["sensitivity"] = sens
    # ---- A/B pilot in Riyadh: stratified by risk tier, 50/50 (Blog 9) ----
    pilot = t[t.city == "Riyadh"].copy()
    pilot["group"] = "control"
    for tier, idx in pilot.groupby("tier").groups.items():
        chosen = rs.choice(idx, size=len(idx) // 2, replace=False)
        pilot.loc[chosen, "group"] = "test"
    fail_sim = pilot.failed.to_numpy().copy()
    is_test = (pilot.group == "test").to_numpy()
    m = is_test & (pilot.tier == "Medium").to_numpy() & (fail_sim == 1)
    fail_sim[m] = (rs.rand(m.sum()) > CONFIRM_EFFECT).astype(int)
    h = is_test & (pilot.tier == "High").to_numpy() & (fail_sim == 1)
    fail_sim[h] = (rs.rand(h.sum()) > (1 - (1 - switch) * (1 - CONFIRM_EFFECT))).astype(int)
    pilot["failed_pilot"] = fail_sim
    a, b = pilot.loc[pilot.group == "test", "failed_pilot"], pilot.loc[pilot.group == "control", "failed_pilot"]
    tt = stats.ttest_ind(a, b)
    p0 = b.mean()
    d = (p0 * (1 - R["problem1_impact"]["relative_reduction_pct"] / 100) - p0) / np.sqrt(p0 * (1 - p0))
    R["problem1_ab"] = {"city": "Riyadh", "orders_test": int(len(a)), "orders_control": int(len(b)),
                        "failed_rate_test_pct": pct(a.mean()), "failed_rate_control_pct": pct(b.mean()),
                        "t_statistic": round(float(tt.statistic), 3), "p_value": float(f"{tt.pvalue:.3g}"),
                        "significant": bool(tt.pvalue < 0.05), "effect_size_d": round(float(d), 4),
                        "required_sample_per_group": int(np.ceil(sample_size(d))),
                        "tier_balance": pilot.groupby(["tier", "group"]).size().unstack().to_dict("index")}


# =============================================================================================
# BLOG 7: MARKET RESPONSE MODEL (Problem 2)
# =============================================================================================
def blog7():
    print("Blog 7: market response")
    d = camp_r.copy()
    aov = d.loc[d.conversion == 1, "order_value_aed"].mean()
    base = d[d.offer == "No Offer"].conversion.mean()
    up = {}
    for off in ["Price Match", "Membership Trial"]:
        grp = d[d.offer == off]
        conv_up = grp.conversion.mean() - base
        order_up = conv_up * len(grp)
        up[off] = {"conversion_rate_pct": pct(grp.conversion.mean()), "conversion_uplift_pct_points": pct(conv_up),
                   "order_uplift": round(order_up, 1), "revenue_uplift_aed": round(order_up * aov)}
    # EDA
    d["disc_bucket"] = pd.cut(d.discount_share, [-0.01, 0.2, 0.4, 0.6, 1.0], labels=["0-20%", "20-40%", "40-60%", "60-100%"])
    eda = d.pivot_table(index="disc_bucket", columns="offer", values="conversion", aggfunc="mean", observed=False).round(3)
    X = pd.get_dummies(d[["recency_days", "frequency", "monetary_aed", "discount_share", "cod_share", "member", "country",
                          "acq_channel", "offer"]], columns=["country", "acq_channel", "offer"], dtype=float)
    y = d.conversion
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=56, stratify=y)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    best, _ = tune("Market response conversion model (Blog 7)", Xtr, ytr, "roc_auc", cv, n_iter=14)
    Xte = Xte.copy()
    Xte["proba"] = best.predict_proba(Xte)[:, 1]
    Xte["conversion"] = yte
    res = {}
    for off, col in [("Price Match", "offer_Price Match"), ("Membership Trial", "offer_Membership Trial")]:
        real = len(Xte) * (Xte[Xte[col] == 1].conversion.mean() - Xte[Xte["offer_No Offer"] == 1].conversion.mean())
        pred = len(Xte) * (Xte[Xte[col] == 1].proba.mean() - Xte[Xte["offer_No Offer"] == 1].proba.mean())
        res[off] = {"real_order_uptick": round(real, 1), "predicted_order_uptick": round(pred, 1),
                    "error_pct": round(abs(pred - real) / abs(real) * 100, 1),
                    "real_revenue_uptick_aed": round(real * aov), "predicted_revenue_uptick_aed": round(pred * aov)}
    R["blog7"] = {"customers": len(d), "avg_order_value_aed": round(aov, 1), "control_conversion_pct": pct(base),
                  "uplift": up, "test_auc": round(roc_auc_score(yte, Xte.proba), 3), "predicted_vs_real": res,
                  "conversion_by_discount_share": eda.reset_index().astype({"disc_bucket": str}).to_dict("records")}


# =============================================================================================
# BLOG 8: UPLIFT WITH COST OPTIMISATION (Problem 2)
# =============================================================================================
def blog8_retention():
    print("Blog 8b: multi-treatment uplift with costs")
    d = camp_r.copy()
    X = pd.get_dummies(d[["recency_days", "frequency", "monetary_aed", "discount_share", "cod_share", "member", "country",
                          "acq_channel"]], columns=["country", "acq_channel"], dtype=float)
    tr, te = train_test_split(np.arange(len(d)), test_size=0.4, random_state=SEED, stratify=d.offer)
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    arms = ["No Offer", "Price Match", "Membership Trial"]
    A = d.offer.to_numpy()
    Y = d.conversion.to_numpy()
    V = d.order_value_aed.to_numpy()
    # expected order value model (trained on converters)
    vm = HistGradientBoostingRegressor(max_depth=3, max_iter=250, learning_rate=0.05, random_state=SEED)
    conv_tr = tr[Y[tr] == 1]
    vm.fit(X.iloc[conv_tr], V[conv_tr])
    v_hat = vm.predict(Xte) * MARGIN  # contribution margin of an order
    ov_hat = vm.predict(Xte)
    cates, p1s = {}, {}
    for arm in arms[1:]:
        m = np.isin(A[tr], ["No Offer", arm])
        T = (A[tr][m] == arm).astype(int)
        cate, _, p1 = x_learner(Xtr[m], T, Y[tr][m], Xte)
        cates[arm], p1s[arm] = cate, p1
    cost_s = {"Price Match": np.minimum(0.10 * ov_hat, 60.0), "Membership Trial": np.full(len(te), 12.0)}
    c = 0.10
    net = {arm: cates[arm] * v_hat - cost_s[arm] * p1s[arm] - c for arm in arms[1:]}
    # policies
    cate_mat = np.column_stack([np.zeros(len(te))] + [cates[a] for a in arms[1:]])
    net_mat = np.column_stack([np.zeros(len(te))] + [net[a] for a in arms[1:]])
    At, Yt, Vt = A[te], Y[te], V[te]
    realised = Yt * Vt * MARGIN - np.where(At == "Price Match", Yt * np.minimum(0.10 * Vt, 60) + c, 0) \
        - np.where(At == "Membership Trial", Yt * 12.0 + c, 0)

    def pv(rec):
        return float((realised * (At == rec) * 3).mean())  # propensity 1/3

    ate = {a: Yt[At == a].mean() - Yt[At == "No Offer"].mean() for a in arms[1:]}
    best_ate = max(ate, key=ate.get)
    pols = {"No offer to anyone": np.full(len(te), "No Offer", dtype=object),
            "Price Match to everyone": np.full(len(te), "Price Match", dtype=object),
            "Membership Trial to everyone": np.full(len(te), "Membership Trial", dtype=object),
            "Random allocation": np.random.RandomState(SEED).choice(arms, len(te)),
            f"Best average arm ({best_ate})": np.full(len(te), best_ate, dtype=object),
            "X-Learner (max CATE)": np.array(arms, dtype=object)[cate_mat.argmax(1)],
            "Net value X-Learner": np.array(arms, dtype=object)[net_mat.argmax(1)]}
    base = pv(pols["No offer to anyone"])
    out = {}
    for k, rec in pols.items():
        val = pv(rec)
        out[k] = {"profit_per_customer_aed": round(val, 3), "incremental_vs_no_offer_aed": round(val - base, 3),
                  "share_offered_pct": pct((rec != "No Offer").mean())}
    chosen = pols["Net value X-Learner"]
    idx = np.array([arms.index(a) for a in chosen])
    exp_inc_conv = cate_mat[np.arange(len(te)), idx]
    R_extra = {"expected_incremental_orders_per_10k": round(float(exp_inc_conv.mean() * 1e4)),
               "expected_incremental_orders_per_10k_targeted": round(float(exp_inc_conv[idx > 0].mean() * 1e4)) if (idx > 0).any() else 0}
    seg = pd.DataFrame({"disc": d.discount_share.to_numpy()[te], "rec": chosen})
    seg["type"] = np.where(seg.disc >= 0.6, "Deal Hunters (discount share >= 60%)", "Others")
    mix = seg.groupby("type").rec.value_counts(normalize=True).unstack().fillna(0).round(3)
    R["blog8_retention"] = {"margin_used": MARGIN, "impression_cost_aed": c, "price_match_cost": "10% of order value, max AED 60",
                            "membership_cost_aed": 12.0, "ate_pct_points": {k: pct(v) for k, v in ate.items()},
                            "policies": out, "recommendation_mix": mix.reset_index().to_dict("records"), **R_extra}
    sc = d.iloc[te][["customer_id", "discount_share", "member", "offer", "conversion"]].copy()
    sc["cate_price_match"] = cates["Price Match"]
    sc["cate_membership"] = cates["Membership Trial"]
    sc["net_price_match"] = net["Price Match"]
    sc["net_membership"] = net["Membership Trial"]
    sc["order_value_hat"] = ov_hat
    sc["p_conv_price_match"] = p1s["Price Match"]
    sc["p_conv_membership"] = p1s["Membership Trial"]
    sc["recommended"] = chosen
    sc.round(4).to_csv(RES / "uplift_recommendations.csv", index=False)
    plt.figure(figsize=(6.5, 3.5))
    ks = list(out)
    plt.barh(ks, [out[k]["incremental_vs_no_offer_aed"] for k in ks], color=["#999"] * 5 + ["#1f5fbf", "#2e8b57"])
    plt.axvline(0, color="k", lw=0.8)
    plt.title("Incremental profit per customer vs no offer (AED)")
    savefig("blog8_policies.png")


# =============================================================================================
# BLOG 9: ANOVA on the retention campaign (Problem 2) + stratified sampling check
# =============================================================================================
def blog9_problem2(rfm):
    print("Blog 9: ANOVA and sampling")
    d = camp_r.copy()
    d["profit"] = d.order_value_aed * MARGIN - np.where(d.offer == "Price Match", np.minimum(0.1 * d.order_value_aed, 60), 0) \
        - np.where(d.offer == "Membership Trial", 12.0 * d.conversion, 0)
    d["segment"] = np.where(d.discount_share >= 0.6, "Deal Hunter", "Other")
    groups = [d.loc[d.offer == a, "conversion"] for a in ["No Offer", "Price Match", "Membership Trial"]]
    f1 = stats.f_oneway(*groups)
    pm = stats.ttest_ind(d.loc[d.offer == "Price Match", "conversion"], d.loc[d.offer == "No Offer", "conversion"])
    mt = stats.ttest_ind(d.loc[d.offer == "Membership Trial", "conversion"], d.loc[d.offer == "No Offer", "conversion"])
    aov2 = two_way_anova(d, "conversion", "segment", "offer")
    # sample size for a retention test: lift 30-day conversion by 5% relative
    p0 = d.loc[d.offer == "No Offer", "conversion"].mean()
    dd = (p0 * 1.05 - p0) / np.sqrt(p0 * (1 - p0))
    # random vs stratified split: spread of High-Value share difference over 300 draws
    seg = rfm.Segment
    diff_r, diff_s = [], []
    rs = np.random.RandomState(SEED)
    for _ in range(300):
        tmask = rs.rand(len(seg)) < 0.9
        diff_r.append((seg[tmask] == "High-Value").mean() - (seg[~tmask] == "High-Value").mean())
        ts = pd.Series(False, index=seg.index)
        for s_, idx in seg.groupby(seg).groups.items():
            ts.loc[rs.choice(idx, int(round(0.9 * len(idx))), replace=False)] = True
        diff_s.append((seg[ts] == "High-Value").mean() - (seg[~ts] == "High-Value").mean())
    R["blog9_problem2"] = {
        "one_way_anova_conversion": {"F": round(float(f1.statistic), 2), "p_value": float(f"{f1.pvalue:.3g}")},
        "t_test_price_match_vs_control": {"t": round(float(pm.statistic), 2), "p_value": float(f"{pm.pvalue:.3g}")},
        "t_test_membership_vs_control": {"t": round(float(mt.statistic), 2), "p_value": float(f"{mt.pvalue:.3g}")},
        "two_way_anova": aov2.assign(p_value=aov2.p_value.map(lambda x: float(f"{x:.3g}") if pd.notna(x) else None)).round(4).to_dict("records"),
        "sample_size_5pct_relative_lift": {"control_conversion_pct": pct(p0), "effect_size_d": round(float(dd), 4),
                                           "per_group": int(np.ceil(sample_size(dd)))},
        "sampling_check_high_value_share_gap_pct_points": {"random_split_sd": round(float(np.std(diff_r)) * 100, 3),
                                                           "stratified_split_sd": round(float(np.std(diff_s)) * 100, 3)},
        "conversion_by_segment_and_offer_pct": (d.pivot_table(index="segment", columns="offer", values="conversion") * 100).round(2).to_dict("index"),
    }


# =============================================================================================
# BLOG 5: NEXT PURCHASE DAY (Problem 3: who will buy in the peak window)
# =============================================================================================
def blog5():
    print("Blog 5: next purchase day")
    f_start, f_end, n_end = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-05-31"), pd.Timestamp("2026-08-31")
    o6 = orders[(orders.order_date >= f_start) & (orders.order_date <= f_end)]
    nxt = orders[(orders.order_date > f_end) & (orders.order_date <= n_end)]
    last = o6.groupby("customer_id").order_date.max()
    first_next = nxt.groupby("customer_id").order_date.min()
    lab = ((first_next.reindex(last.index) - last).dt.days).fillna(999)
    g = rfm_table(o6, f_end)
    g = kmeans_col(g, "Recency", 4, False)
    g = kmeans_col(g, "Frequency", 4, True)
    g = kmeans_col(g, "Revenue", 4, True)
    g["OverallScore"] = g.RecencyCluster + g.FrequencyCluster + g.RevenueCluster
    day = o6[["customer_id", "order_date"]].drop_duplicates().sort_values(["customer_id", "order_date"])
    for k in (1, 2, 3):
        day[f"T{k}"] = day.groupby("customer_id").order_date.shift(k)
        day[f"DayDiff{'' if k == 1 else k}"] = (day.order_date - day[f"T{k}"]).dt.days
    dd = day.groupby("customer_id").DayDiff.agg(["mean", "std"]).rename(columns={"mean": "DayDiffMean", "std": "DayDiffStd"})
    lastrow = day.drop_duplicates("customer_id", keep="last").set_index("customer_id")[["DayDiff", "DayDiff2", "DayDiff3"]].dropna()
    g = g.join(lastrow, how="inner").join(dd)  # only customers with > 3 purchases, as in the blog
    g = g.join(cust.set_index("customer_id")[["member"]])
    g["NextPurchaseDay"] = lab
    g["NextPurchaseDayRange"] = np.select([g.NextPurchaseDay <= 20, g.NextPurchaseDay <= 49], [2, 1], 0)
    X = g.drop(columns=["NextPurchaseDay", "NextPurchaseDayRange"]).fillna(0)
    y = g.NextPurchaseDayRange
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)
    models = {"Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
              "Naive Bayes": GaussianNB(), "Random Forest": RandomForestClassifier(300, random_state=SEED, n_jobs=2),
              "Decision Tree": DecisionTreeClassifier(random_state=SEED),
              "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
              R["backend"]["gbm"]: gbm_classifier()}
    comp = {}
    for k, m in models.items():
        s = cross_val_score(m, Xtr, ytr, cv=cv, scoring="accuracy", n_jobs=2)
        comp[k] = {"cv_accuracy_mean": round(s.mean(), 4), "cv_std": round(s.std(), 4)}
    # Grid search on the blog's two parameters (max_depth, min_child_weight analogue) plus learning rate
    if HAS_XGB:
        grid = {"max_depth": [3, 5, 7, 9], "min_child_weight": [1, 3, 5], "learning_rate": [0.05, 0.1]}
    else:
        grid = {"max_depth": [3, 5, 7, 9], "min_samples_leaf": [10, 30, 60], "learning_rate": [0.05, 0.1]}
    t0 = time.time()
    default_acc = accuracy_score(yte, gbm_classifier().fit(Xtr, ytr).predict(Xte))
    gs = GridSearchCV(gbm_classifier(), grid, cv=cv, scoring="accuracy", n_jobs=2).fit(Xtr, ytr)
    TUNING.append({"model": "Next purchase day classifier (Blog 5)", "algorithm": type(gs.best_estimator_).__name__,
                   "metric": "accuracy", "cv_default": comp[R["backend"]["gbm"]]["cv_accuracy_mean"],
                   "cv_tuned": round(gs.best_score_, 4), "best_params": json.dumps(gs.best_params_),
                   "candidates_tested": int(np.prod([len(v) for v in grid.values()])), "seconds": round(time.time() - t0, 1)})
    tuned_acc = accuracy_score(yte, gs.predict(Xte))
    R["blog5"] = {"feature_window": "2025-12-01 to 2026-05-31", "label_window": "2026-06-01 to 2026-08-31",
                  "customers": len(g), "class_share_pct": (y.value_counts(normalize=True) * 100).round(1).to_dict(),
                  "model_comparison": comp, "test_accuracy_default_pct": pct(default_acc),
                  "test_accuracy_tuned_pct": pct(tuned_acc), "best_params": gs.best_params_,
                  "corr_with_label": X.join(y).corr()["NextPurchaseDayRange"].drop("NextPurchaseDayRange").round(3).sort_values().to_dict()}
    g["pred_class"] = gs.predict(X)
    g[["Recency", "Frequency", "Revenue", "OverallScore", "NextPurchaseDay", "NextPurchaseDayRange", "pred_class"]].to_csv(RES / "next_purchase.csv")


# =============================================================================================
# BLOG 6: PREDICTING SALES (Problem 3)
# =============================================================================================
EVENTS = ["ramadan", "pre_eid", "eid", "yellow_friday", "back_to_school", "payday"]


def weekly_panel():
    d = demand.copy()
    d["week"] = d.date.dt.to_period("W-SAT").dt.start_time
    agg = {"demand_units": "sum", "planned_stock_units": "sum", "units_sold": "sum", "lost_units": "sum", "leftover_units": "sum"}
    agg.update({e: "sum" for e in EVENTS})
    w = d.groupby(["country", "category", "week"]).agg(agg).reset_index()
    w = w[(w.groupby(["country", "category"]).week.transform("count") > 0)]
    full_weeks = d.groupby("week").date.nunique()
    w = w[w.week.isin(full_weeks[full_weeks == 7].index)]
    w = w.sort_values(["country", "category", "week"]).reset_index(drop=True)
    grp = w.groupby(["country", "category"])
    w["diff"] = grp.demand_units.diff()
    for k in range(1, 13):
        w[f"lag_{k}"] = grp["diff"].shift(k)
    for e in EVENTS:
        w[f"{e}_chg"] = w[e] - grp[e].shift(1)  # change in event-days vs last week drives the jump
    w["prev_units"] = grp.demand_units.shift(1)
    w["woy"] = w.week.dt.isocalendar().week.astype(int)
    return w.dropna().reset_index(drop=True)


def blog6():
    print("Blog 6: sales forecasting")
    w = weekly_panel()
    w = pd.get_dummies(w, columns=["category", "country"], dtype=float, prefix=["cat", "cty"])
    lag_cols = [f"lag_{k}" for k in range(1, 13)]
    ev_cols = EVENTS + [f"{e}_chg" for e in EVENTS]
    dummies = [c for c in w.columns if c.startswith(("cat_", "cty_"))]
    # interaction: event change x category so that Ramadan moves grocery and Yellow Friday moves electronics
    inter = []
    for e in ["ramadan_chg", "pre_eid_chg", "eid_chg", "yellow_friday_chg", "back_to_school_chg"]:
        for c in [c for c in dummies if c.startswith("cat_")]:
            w[f"{e}*{c}"] = w[e] * w[c] * w.prev_units / 1000
            inter.append(f"{e}*{c}")
    feats = lag_cols + ev_cols + dummies + inter + ["prev_units"]
    train = w[w.week < pd.Timestamp("2026-01-01")]
    test = w[w.week >= pd.Timestamp("2026-01-01")]
    # Adjusted R2 (Blog 6) on the training data
    ar2 = {"lag_1": adj_r2(train[["lag_1"]].to_numpy(), train["diff"].to_numpy()),
           "lag_1..5": adj_r2(train[lag_cols[:5]].to_numpy(), train["diff"].to_numpy()),
           "lag_1..12": adj_r2(train[lag_cols].to_numpy(), train["diff"].to_numpy()),
           "lag_1..12 + events": adj_r2(train[lag_cols + ev_cols + inter].to_numpy(), train["diff"].to_numpy())}
    sc_x = MinMaxScaler((-1, 1)).fit(train[feats])
    sc_y = MinMaxScaler((-1, 1)).fit(train[["diff"]])
    Xtr, Xte = sc_x.transform(train[feats]), sc_x.transform(test[feats])
    ytr = sc_y.transform(train[["diff"]]).ravel()
    tscv = TimeSeriesSplit(4)
    # (a) sequence model: Keras LSTM when available, otherwise an MLP on the same lags
    if HAS_TF:
        from tensorflow import keras

        def build(units):
            m = keras.Sequential([keras.layers.Input((1, Xtr.shape[1])), keras.layers.LSTM(units), keras.layers.Dense(1)])
            m.compile(loss="mean_squared_error", optimizer="adam")
            return m
        best_u, best_s = None, np.inf
        for u in [4, 16, 32]:
            m = build(u)
            m.fit(Xtr.reshape(-1, 1, Xtr.shape[1]), ytr, epochs=60, batch_size=32, verbose=0, shuffle=False, validation_split=0.2)
            s = m.evaluate(Xtr[-200:].reshape(-1, 1, Xtr.shape[1]), ytr[-200:], verbose=0)
            if s < best_s:
                best_u, best_s, seq = u, s, m
        pred_seq = seq.predict(Xte.reshape(-1, 1, Xte.shape[1]), verbose=0).ravel()
        TUNING.append({"model": "Sales forecast LSTM (Blog 6)", "algorithm": "Keras LSTM", "metric": "val MSE",
                       "cv_default": None, "cv_tuned": round(float(best_s), 5), "best_params": json.dumps({"units": best_u}),
                       "candidates_tested": 3, "seconds": None})
    else:
        mlp, _ = tune("Sales forecast neural net (Blog 6, LSTM stand-in)", Xtr, ytr, "neg_mean_absolute_error", tscv, n_iter=12,
                      base=MLPRegressor(max_iter=800, early_stopping=True, random_state=SEED),
                      grid={"hidden_layer_sizes": [(16,), (32,), (64,), (32, 16), (64, 32)], "alpha": [1e-4, 1e-3, 1e-2, 1e-1],
                            "learning_rate_init": [1e-3, 3e-3, 1e-2], "activation": ["relu", "tanh"]})
        pred_seq = mlp.predict(Xte)
    # (b) tuned gradient boosting regressor
    gbr_base = XGBRegressor(random_state=SEED, n_jobs=2, verbosity=0) if HAS_XGB else HistGradientBoostingRegressor(random_state=SEED)
    gbr_grid = ({"max_depth": [3, 4, 6, 8], "learning_rate": [0.03, 0.05, 0.1], "n_estimators": [200, 400, 800],
                 "min_child_weight": [1, 5, 10], "subsample": [0.7, 0.85, 1.0]} if HAS_XGB else
                {"max_depth": [3, 4, 6, None], "learning_rate": [0.03, 0.05, 0.1], "max_iter": [200, 400, 800],
                 "min_samples_leaf": [5, 10, 20, 40], "l2_regularization": [0.0, 0.1, 1.0]})
    gbr, _ = tune("Sales forecast gradient boosting (Blog 6 extension)", Xtr, ytr, "neg_mean_absolute_error", tscv, n_iter=14,
                  base=gbr_base, grid=gbr_grid)
    pred_gbr = gbr.predict(Xte)

    def to_units(p):
        return test.prev_units.to_numpy() + sc_y.inverse_transform(p.reshape(-1, 1)).ravel()

    out = test[["week", "prev_units", "demand_units", "planned_stock_units", "lost_units", "leftover_units"] + EVENTS].copy()
    out["country"] = w.loc[test.index, [c for c in w.columns if c.startswith("cty_")]].idxmax(1).str[4:].to_numpy()
    out["category"] = w.loc[test.index, [c for c in w.columns if c.startswith("cat_")]].idxmax(1).str[4:].to_numpy()
    out["fc_sequence"] = to_units(pred_seq)
    out["fc_gbm"] = to_units(pred_gbr)
    out["fc_current_plan"] = out.planned_stock_units / 1.10  # the plan before its 10% buffer
    out["fc_naive_last_week"] = out.prev_units
    out["fc_ensemble"] = 0.5 * out.fc_sequence + 0.5 * out.fc_gbm
    peak = (out[["ramadan", "pre_eid", "eid", "yellow_friday"]].sum(1) > 0)

    def wape(a, f):
        return float(np.abs(a - f).sum() / a.sum() * 100)

    acc = {}
    for k in ["fc_current_plan", "fc_naive_last_week", "fc_sequence", "fc_gbm", "fc_ensemble"]:
        acc[k] = {"WAPE_all_pct": round(wape(out.demand_units, out[k]), 2),
                  "WAPE_peak_weeks_pct": round(wape(out.demand_units[peak], out[k][peak]), 2),
                  "WAPE_normal_weeks_pct": round(wape(out.demand_units[~peak], out[k][~peak]), 2),
                  "MAPE_pct": round(float((np.abs(out.demand_units - out[k]) / out.demand_units).mean() * 100), 2),
                  "MAE_units": round(mean_absolute_error(out.demand_units, out[k]), 1)}
    best_key = min(["fc_sequence", "fc_gbm", "fc_ensemble"], key=lambda k: acc[k]["WAPE_all_pct"])
    # stock simulation: forecast + safety buffer chosen on TRAIN residuals (90th percentile of under-forecast)
    tr_pred = 0.5 * (train.prev_units.to_numpy() + sc_y.inverse_transform(gbr.predict(Xtr).reshape(-1, 1)).ravel()) + \
        0.5 * (train.prev_units.to_numpy() + sc_y.inverse_transform((seq.predict(Xtr.reshape(-1, 1, Xtr.shape[1]), verbose=0).ravel() if HAS_TF else mlp.predict(Xtr)).reshape(-1, 1)).ravel())
    rel_err = (train.demand_units.to_numpy() - tr_pred) / tr_pred
    buf = float(np.clip(np.quantile(rel_err, 0.85), 0.02, 0.4))
    stock_new = out[best_key] * (1 + buf)
    sold_new = np.minimum(out.demand_units, stock_new)
    lost_new = out.demand_units - sold_new
    left_new = stock_new - sold_new
    cur = {"stockout_weeks_pct": pct((out.lost_units > 0).mean()), "lost_units_pct": pct(out.lost_units.sum() / out.demand_units.sum()),
           "leftover_units_pct": pct(out.leftover_units.sum() / out.demand_units.sum()),
           "stockout_weeks_peak_pct": pct((out.lost_units[peak] > 0).mean()),
           "lost_units_peak_pct": pct(out.lost_units[peak].sum() / out.demand_units[peak].sum())}
    new = {"stockout_weeks_pct": pct((lost_new > 0.5).mean()), "lost_units_pct": pct(lost_new.sum() / out.demand_units.sum()),
           "leftover_units_pct": pct(left_new.sum() / out.demand_units.sum()),
           "stockout_weeks_peak_pct": pct((lost_new[peak] > 0.5).mean()),
           "lost_units_peak_pct": pct(lost_new[peak].sum() / out.demand_units[peak].sum())}
    trade = []
    for b in [0.0, buf, 0.08, 0.15]:
        st = out[best_key] * (1 + b)
        so = np.minimum(out.demand_units, st)
        trade.append({"buffer_pct": round(b * 100, 1), "stockout_weeks_pct": pct(((out.demand_units - so) > 0.5).mean()),
                      "lost_units_pct": pct((out.demand_units - so).sum() / out.demand_units.sum()),
                      "leftover_units_pct": pct((st - so).sum() / out.demand_units.sum())})
    out["stock_new_plan"] = stock_new
    out["lost_new_plan"] = lost_new
    out.round(1).to_csv(RES / "forecast_weekly.csv", index=False)
    R["blog6"] = {"granularity": "weekly, 3 countries x 5 categories = 15 series", "train": "to 2025-12-31",
                  "test": "2026-01 to 2026-10 (includes Ramadan, Eid al-Fitr and Eid al-Adha 2026)",
                  "adjusted_r2": {k: round(v, 3) for k, v in ar2.items()}, "accuracy": acc, "best_model": best_key,
                  "safety_buffer_pct": round(buf * 100, 1), "current_plan": cur, "forecast_plan": new,
                  "lost_units_recovered": int(out.lost_units.sum() - lost_new.sum()), "buffer_tradeoff": trade}
    # chart: national weekly demand vs forecasts
    nat = out.groupby("week")[["demand_units", "fc_current_plan", best_key]].sum()
    plt.figure(figsize=(10, 3.6))
    plt.plot(nat.index, nat.demand_units / 1e3, label="Actual demand", color="k")
    plt.plot(nat.index, nat.fc_current_plan / 1e3, label="Current plan (last year x 1.2)", ls="--", color="#d0453a")
    plt.plot(nat.index, nat[best_key] / 1e3, label="Model forecast", color="#1f5fbf")
    plt.axvspan(pd.Timestamp("2026-02-15"), pd.Timestamp("2026-03-22"), color="#f2c14e", alpha=0.25, label="Ramadan + Eid 2026")
    plt.title("Weekly units (thousands), test period 2026")
    plt.legend(fontsize=8)
    savefig("blog6_forecast.png")


# =============================================================================================
def main():
    t0 = time.time()
    blog1()
    rfm = blog2()
    blog3()
    blog4_churn()
    test = blog4_refusal()
    upl, share = blog8_prepaid()
    problem1_impact(test, upl, share)
    blog7()
    blog8_retention()
    blog9_problem2(rfm)
    blog5()
    blog6()
    pd.DataFrame(TUNING).to_csv(RES / "hyperparameter_tuning.csv", index=False)
    R["tuning"] = TUNING
    R["runtime_seconds"] = round(time.time() - t0, 1)
    with open(RES / "results.json", "w") as f:
        json.dump(R, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"done in {R['runtime_seconds']} s")


if __name__ == "__main__":
    main()
