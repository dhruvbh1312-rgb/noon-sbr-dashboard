"""
Noon SBR dashboard (Streamlit).

Reads the pre-computed outputs in results/ (made by src/analysis.py) and adds live what-if controls.
Run locally:  streamlit run app.py
All data is SYNTHETIC and built for a university SBR exercise.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CH = ROOT / "charts"
sys.path.insert(0, str(ROOT / "src"))

st.set_page_config(page_title="Noon SBR Dashboard", page_icon="📊", layout="wide")


@st.cache_data
def load_json():
    with open(RES / "results.json") as f:
        return json.load(f)


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(RES / name, **kw)


R = load_json()

LIVE = "🔴 Live operations"
PAGES = [LIVE, "Overview", "1. Metrics (Blog 1)", "2. Segments (Blogs 2, 3)", "Problem 1: COD refusals",
         "Problem 2: Switching to rivals", "Problem 3: Demand spikes", "A/B tests (Blog 9)", "Model tuning", "About the data"]
page = st.sidebar.radio("Go to", PAGES)
st.sidebar.markdown("---")
st.sidebar.caption("Team: Harshita (CEO), Dhruv P Bhansali (Data), Abhinav Sharma (Marketing), "
                   "Shristi Murarka (CX), Sashidhar (Operations), Durga Venu (Finance)")
st.sidebar.caption(f"Models: {R['backend']['gbm']} | {R['backend']['sequence_model']}")


def aed(x):
    return f"AED {x:,.0f}"


# ------------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Starting the live data stream (trains the refusal model, about 10 seconds)...")
def get_store():
    from live_stream import LiveStore
    return LiveStore()


def live_page():
    from live_stream import CATS, CITIES
    st.title("🔴 Noon live operations")
    st.caption("A simulated real-time feed of orders, deliveries, customer reviews, competitor prices and stock, timed in Dubai time. "
               "Models and alerts run on every refresh and detect events from the data alone.")
    store = get_store()
    c = st.columns([1, 1, 1, 2])
    every = c[0].selectbox("Refresh every", ["5s", "10s", "30s"], index=0)
    window = c[1].selectbox("Window", [15, 60, 180], index=1, format_func=lambda m: f"Last {m} min")
    paused = c[2].toggle("Pause", value=False)
    with c[3].expander("Demo: trigger a real-world event"):
        e1, e2 = st.columns(2)
        kind = e1.selectbox("Event", ["flash_sale", "courier_issue", "rival_drop"],
                            format_func={"flash_sale": "Flash sale", "courier_issue": "Courier problem",
                                         "rival_drop": "Rival price drop"}.get)
        target = e2.selectbox("Where", list(CITIES) if kind == "courier_issue" else CATS)
        if st.button("Trigger now", type="primary"):
            store.inject(kind, target, minutes=30)
            st.toast(f"Triggered for 30 minutes. Watch the alerts as the data comes in.")

    @st.fragment(run_every=None if paused else every)
    def feed():
        store.advance()
        snap = store.snapshot(window)
        k = snap["kpi"]
        st.markdown(f"**Last update:** {snap['t']:%d %b %Y, %H:%M:%S} Dubai time  |  {k['pending']:,} deliveries on the road")
        m = st.columns(8)
        m[0].metric("Orders", f"{k['orders']:,}", f"{k['orders_per_min']:.1f} / min")
        m[1].metric("Revenue", f"AED {k['revenue'] / 1000:,.0f}k")
        m[2].metric("COD share", f"{k['cod_share'] * 100:.0f}%")
        m[3].metric("COD failed rate", "n/a" if np.isnan(k["cod_failed_rate"]) else f"{k['cod_failed_rate'] * 100:.1f}%")
        m[4].metric("Avg rating", "n/a" if np.isnan(k["avg_stars"]) else f"{k['avg_stars']:.2f} ★")
        m[5].metric("Net sentiment", "n/a" if np.isnan(k["net_sentiment"]) else f"{k['net_sentiment']:+.0f}")
        m[6].metric("Fake reviews caught", k["spam_caught"])
        m[7].metric("Orders lost to stock-outs", k["lost_stockout"])

        st.subheader("Alerts")
        if snap["alerts"]:
            for title, msg, prob in snap["alerts"][:8]:
                (st.error if title in ("Refusals rising", "Stock-out risk") else st.warning)(f"**{title}** ({prob}): {msg}")
        else:
            st.success("No anomalies in the last 30 minutes.")

        t1, t2, t3, t4 = st.tabs(["Orders and deliveries", "Customer reviews", "Stock and competitor prices", "Data quality and model health"])
        o, ow = snap["orders"], snap["orders_w"]
        with t1:
            a, b = st.columns(2)
            if len(ow):
                per_min = ow.set_index("ts").groupby("category").resample("1min").size().unstack(0).fillna(0)
                a.markdown("**Orders per minute by category**")
                a.line_chart(per_min)
            res = snap["resolved"]
            rw = res[res.resolve_ts >= snap["t"] - timedelta(minutes=window)]
            if len(rw):
                b.markdown("**COD failed-delivery rate by city (%)**")
                b.bar_chart((rw.assign(f=rw.status != "delivered").groupby("city").f.mean() * 100).round(1))
            st.markdown("**Newest COD orders, scored by the refusal-risk model**")
            cod = o[o.payment_method == "COD"].tail(12).iloc[::-1]
            if len(cod):
                show = cod[["ts", "city", "category", "order_value_aed", "first_order", "address_quality", "risk", "status"]].copy()
                show["risk tier"] = np.select([show.risk >= 0.40, show.risk >= 0.20], ["High: confirm + prepaid nudge", "Medium: WhatsApp confirm"], "Low")
                show["ts"] = show.ts.dt.strftime("%H:%M:%S")
                st.dataframe(show, hide_index=True, use_container_width=True,
                             column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f")})
        with t2:
            rv = snap["reviews_w"]
            if len(rv):
                a, b = st.columns(2)
                real = rv[~rv.is_spam_flag]
                trend = real.set_index("ts").resample("5min").sentiment.apply(lambda s: (s == "Negative").mean() * 100 if len(s) else np.nan)
                a.markdown("**Negative reviews (% of genuine reviews, 5-min buckets)**")
                a.line_chart(trend)
                b.markdown("**What customers talk about**")
                b.bar_chart(real.groupby(["topic", "sentiment"]).size().unstack(fill_value=0))
                st.markdown("**Latest reviews** (raw text as written, with what the analyser concluded)")
                feedv = rv.sort_values("ts", ascending=False).head(15)
                feedv = feedv.assign(time=feedv.ts.dt.strftime("%H:%M"),
                                     flag=np.where(feedv.is_spam_flag, "🚫 fake/spam", ""),
                                     mood=feedv.sentiment.map({"Positive": "🟢", "Neutral": "⚪", "Negative": "🔴"}) + " " + feedv.sentiment)
                st.dataframe(feedv[["time", "city", "category", "stars", "text", "mood", "topic", "flag"]], hide_index=True, use_container_width=True)
        with t3:
            a, b = st.columns(2)
            stock = pd.Series(snap["stock"]).rename("units").reset_index()
            stock.columns = ["country", "category", "units"]
            a.markdown("**Units on hand (refilled hourly: forecast x 4 h x 1.08 buffer)**")
            a.bar_chart(stock.pivot(index="category", columns="country", values="units").round(0))
            pr = snap["prices"]
            if len(pr):
                b.markdown("**Competitor price index (100 = our price)**")
                b.line_chart(pr[pr.ts >= snap["t"] - timedelta(minutes=window)].set_index("ts")[CATS])
            if snap["restocks"]:
                st.markdown("**Latest replenishments**")
                st.dataframe(pd.DataFrame(snap["restocks"][-8:]).iloc[::-1], hide_index=True, use_container_width=True)
        with t4:
            q, mo = snap["quality"], snap["monitor"]
            a = st.columns(4)
            a[0].metric("Raw order records received", f"{q['raw_rows']:,}")
            a[1].metric("Missing city filled", q["missing_city"])
            a[2].metric("Negative values fixed", q["negative_value"])
            a[3].metric("Duplicate records removed", q["duplicates"])
            b = st.columns(4)
            b[0].metric("Refusal model live ROC-AUC", "n/a" if np.isnan(mo["risk_auc"]) else f"{mo['risk_auc']:.3f}")
            b[1].metric("Sentiment accuracy", "n/a" if np.isnan(mo["sentiment_acc"]) else f"{mo['sentiment_acc'] * 100:.1f}%")
            b[2].metric("Fake reviews caught (recall)", "n/a" if np.isnan(mo["spam_recall"]) else f"{mo['spam_recall'] * 100:.0f}%")
            b[3].metric("Flags that were truly fake (precision)", "n/a" if np.isnan(mo["spam_precision"]) else f"{mo['spam_precision'] * 100:.0f}%")
            st.caption("Accuracy figures compare the analysers with the simulator's hidden truth, which a real company would get from manual audits.")
            with st.expander("Answer key: events actually running (hidden from the alert engine)"):
                act = [s for s in snap["shocks"] if s.end >= snap["t"] - timedelta(minutes=window)]
                st.write([f"{s.label()} {s.start:%H:%M} to {s.end:%H:%M}{' (manual)' if s.manual else ''}" for s in act] or "None")

    feed()


if page == LIVE:
    live_page()

# ------------------------------------------------------------------------------------------------
elif page == "Overview":
    st.title("Noon: Data-Driven Growth Dashboard")
    st.caption("SBR group project. All figures come from a synthetic dataset built to behave like a Middle East marketplace.")
    p1, b6, b8 = R["problem1_impact"], R["blog6"], R["blog8_retention"]["policies"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Problem 1: COD failed-delivery rate", f"{p1['failed_rate_after_pct']}%",
              f"{p1['failed_rate_after_pct'] - p1['failed_rate_before_pct']:.1f} pts vs today", delta_color="inverse")
    c2.metric("Problem 2: profit per customer, targeted offers vs blanket price match",
              aed(b8["Net value X-Learner"]["profit_per_customer_aed"]),
              f"{b8['Net value X-Learner']['profit_per_customer_aed'] - b8['Price Match to everyone']['profit_per_customer_aed']:.2f} AED")
    c3.metric("Problem 3: forecast error (WAPE) in peak weeks",
              f"{b6['accuracy'][b6['best_model']]['WAPE_peak_weeks_pct']}%",
              f"{b6['accuracy'][b6['best_model']]['WAPE_peak_weeks_pct'] - b6['accuracy']['fc_current_plan']['WAPE_peak_weeks_pct']:.1f} pts",
              delta_color="inverse")
    st.subheader("How the 9 blogs map to the three problems")
    st.dataframe(pd.DataFrame({
        "Problem": ["1. COD refusals", "2. Customers switching to rivals", "3. Ramadan / Yellow Friday spikes"],
        "Blogs used": ["1, 2, 4, 8, 9", "1, 2, 3, 4, 7, 8, 9", "1, 2, 5, 6, 9"],
        "Main technique": ["Refusal-risk model + risk tiers + targeted prepaid cashback",
                           "Churn model + CLV + net-value uplift targeting",
                           "Event-aware demand forecast + safety buffer"]}), hide_index=True, use_container_width=True)
    m = load_csv("monthly_metrics.csv", index_col=0)
    st.subheader("North Star: monthly revenue (AED)")
    st.line_chart(m[["revenue"]])

# ------------------------------------------------------------------------------------------------
elif page == "1. Metrics (Blog 1)":
    st.title("Know your metrics")
    b1 = R["blog1"]
    st.info(b1["north_star"])
    c = st.columns(4)
    c[0].metric("Revenue, last 12 months", aed(b1["revenue_last12_aed"]), f"{b1['yoy_growth_pct']}% YoY")
    c[1].metric("Monthly retention (avg, 12m)", f"{b1['avg_monthly_retention_last12_pct']}%")
    c[2].metric("New customer ratio (avg, 12m)", f"{b1['avg_new_customer_ratio_last12_pct']}%")
    c[3].metric("Delivery success rate", f"{b1['delivery_success_last12_pct']}%")
    m = load_csv("monthly_metrics.csv", index_col=0)
    metric = st.selectbox("Metric", ["revenue", "growth_rate_pct", "active_customers", "orders", "avg_revenue_per_order",
                                     "new_customer_ratio_pct", "retention_rate_pct", "cod_share_pct", "cod_failed_rate_pct",
                                     "delivery_success_rate_pct", "failed_delivery_cost_aed"])
    st.line_chart(m[[metric]])
    st.subheader("Cohort retention (% of each joining month still buying k months later)")
    coh = load_csv("cohort_retention.csv", index_col=0)
    st.dataframe(coh.iloc[:, :13].style.background_gradient(cmap="Blues", axis=None).format("{:.0f}", na_rep=""),
                 use_container_width=True)

# ------------------------------------------------------------------------------------------------
elif page == "2. Segments (Blogs 2, 3)":
    st.title("RFM segmentation and lifetime value")
    rfm = load_csv("rfm_segments.csv", index_col=0)
    seg = pd.DataFrame(R["blog2"]["segments"]).set_index("Segment")
    st.dataframe(seg.round(1), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.subheader("Elbow method (Recency)")
    el = pd.Series(R["blog2"]["elbow_inertia"]).rename("inertia (millions)")
    el.index = el.index.astype(int)
    c1.line_chart(el)
    c2.subheader("Problem 2 behaviour segments")
    p2 = pd.DataFrame(R["blog2"]["p2_segments"]).set_index("P2_segment")
    c2.bar_chart(p2["customers"])
    country = st.multiselect("Country", sorted(rfm.country.unique()), default=sorted(rfm.country.unique()))
    sub = rfm[rfm.country.isin(country)]
    st.scatter_chart(sub.sample(min(3000, len(sub)), random_state=1), x="Frequency", y="Revenue", color="Segment")
    st.subheader("CLV prediction (Blog 3)")
    b3 = R["blog3"]
    c = st.columns(3)
    c[0].metric("Benchmark accuracy (largest class)", f"{b3['benchmark_accuracy_pct']}%")
    c[1].metric("Default model", f"{b3['test_accuracy_default_pct']}%")
    c[2].metric("Tuned model", f"{b3['test_accuracy_tuned_pct']}%")
    st.dataframe(pd.DataFrame(b3["ltv_clusters"]), hide_index=True)

# ------------------------------------------------------------------------------------------------
elif page == "Problem 1: COD refusals":
    st.title("Problem 1: Cash on delivery refusals")
    b4 = R["blog4_refusal"]
    c = st.columns(4)
    c[0].metric("COD failed rate (test period)", f"{b4['failed_rate_test_pct']}%")
    c[1].metric("Refusal model ROC-AUC", b4["tuned_gbm_auc"], f"{b4['tuned_gbm_auc'] - b4['decision_tree_auc']:.3f} vs tree")
    c[2].metric("Failures caught in riskiest 10%", f"{b4['top10pct_capture_pct']}%")
    c[3].metric("Failures caught in riskiest 30%", f"{b4['top30pct_capture_pct']}%")
    lift = pd.DataFrame(b4["lift_table"]).set_index("decile")
    st.subheader("Failed-delivery rate by risk decile (1 = riskiest)")
    st.bar_chart(lift["failed_rate"] * 100)

    st.subheader("What-if: risk-tier policy")
    sc = load_csv("cod_risk_scores.csv", parse_dates=["order_date"])
    col = st.columns(4)
    hi = col[0].slider("High tier = riskiest x%", 5, 40, 15)
    md = col[1].slider("Medium tier = next y%", 5, 50, 25)
    eff = col[2].slider("Confirmation cuts failures by (%)", 0, 60, 30) / 100
    cost = col[3].number_input("Cost per failed delivery (AED)", 5.0, 100.0, 25.0)
    city = st.multiselect("Cities", sorted(sc.city.unique()), default=sorted(sc.city.unique()))
    s = sc[sc.city.isin(city)].copy()
    q_hi, q_md = np.quantile(s.risk, 1 - hi / 100), np.quantile(s.risk, 1 - (hi + md) / 100)
    s["tier"] = np.select([s.risk >= q_hi, s.risk >= q_md], ["High", "Medium"], "Low")
    switch = R["problem1_impact"]["assumptions"]["high_tier_switch_to_prepaid_pct"] / 100
    after = s.failed.astype(float)
    after = np.where(s.tier == "Medium", s.failed * (1 - eff), after)
    after = np.where(s.tier == "High", s.failed * (1 - switch) * (1 - eff) + switch * 0.012, after)
    msgs = (s.tier != "Low").sum()
    months = max((s.order_date.max() - s.order_date.min()).days / 30.4, 1)
    saving = ((s.failed.sum() - after.sum()) * cost - msgs * 0.15) * 12 / months
    k = st.columns(3)
    k[0].metric("Failed rate today", f"{s.failed.mean() * 100:.2f}%")
    k[1].metric("Failed rate with policy", f"{after.mean() * 100:.2f}%", f"{(after.mean() - s.failed.mean()) * 100:.2f} pts",
                delta_color="inverse")
    k[2].metric("Net saving per year", aed(saving))
    st.dataframe(s.groupby("tier").agg(orders=("failed", "size"), failed_rate=("failed", "mean")).assign(
        failed_rate=lambda d: (d.failed_rate * 100).round(2)), use_container_width=True)
    st.subheader("Who should get the AED 10 prepaid cashback? (Blog 8)")
    pol = pd.DataFrame(R["blog8_prepaid"]["policies"]).T
    st.dataframe(pol, use_container_width=True)
    st.caption("Blanket cashback loses money because many customers would prepay anyway. Net-value targeting keeps the offer "
               "only where the saved delivery cost beats the cashback.")

# ------------------------------------------------------------------------------------------------
elif page == "Problem 2: Switching to rivals":
    st.title("Problem 2: Price-comparing customers switching to rivals")
    ch = R["blog4_churn"]
    c = st.columns(4)
    c[0].metric("Churn rate (90 days)", f"{ch['churn_rate_pct']}%")
    c[1].metric("Churn model ROC-AUC", ch["tuned_gbm"]["roc_auc"])
    c[2].metric("Recall / precision", f"{ch['tuned_gbm']['recall']} / {ch['tuned_gbm']['precision']}")
    c[3].metric("Churners caught in top 20%", f"{ch['top20pct_capture_of_churners_pct']}%")
    st.subheader("Market response (Blog 7): uplift by offer")
    up = pd.DataFrame(R["blog7"]["uplift"]).T
    st.dataframe(up, use_container_width=True)
    st.subheader("Policy comparison (Blog 8): incremental profit per customer vs no offer")
    pol = pd.DataFrame(R["blog8_retention"]["policies"]).T
    st.bar_chart(pol["incremental_vs_no_offer_aed"])
    st.subheader("What-if: change the offer costs")
    u = load_csv("uplift_recommendations.csv")
    col = st.columns(3)
    margin = col[0].slider("Contribution margin (%)", 5, 40, 20) / 100
    pm_cost = col[1].slider("Price match discount (% of order)", 2, 25, 10) / 100
    mem_cost = col[2].slider("Membership trial cost per converter (AED)", 0, 40, 12)
    v = u.order_value_hat
    net_pm = u.cate_price_match * v * margin - np.minimum(pm_cost * v, 60) * u.p_conv_price_match - 0.1
    net_mt = u.cate_membership * v * margin - mem_cost * u.p_conv_membership - 0.1
    best = np.select([(net_pm > 0) & (net_pm >= net_mt), (net_mt > 0) & (net_mt > net_pm)], ["Price Match", "Membership Trial"], "No Offer")
    exp = np.where(best == "Price Match", net_pm, np.where(best == "Membership Trial", net_mt, 0))
    k = st.columns(3)
    k[0].metric("Customers offered something", f"{(best != 'No Offer').mean() * 100:.1f}%")
    k[1].metric("Expected extra profit per 10,000 customers", aed(exp.mean() * 1e4))
    k[2].metric("Deal hunters offered price match", f"{((best == 'Price Match') & (u.discount_share >= 0.6)).sum() / max((u.discount_share >= 0.6).sum(), 1) * 100:.1f}%")
    st.bar_chart(pd.Series(best).value_counts())

# ------------------------------------------------------------------------------------------------
elif page == "Problem 3: Demand spikes":
    st.title("Problem 3: Demand spikes in Ramadan and Yellow Friday")
    b6 = R["blog6"]
    acc = pd.DataFrame(b6["accuracy"]).T
    acc.index = acc.index.str.replace("fc_", "")
    st.dataframe(acc, use_container_width=True)
    st.caption(f"Adjusted R2: {b6['adjusted_r2']}")
    f = load_csv("forecast_weekly.csv", parse_dates=["week"])
    col = st.columns(3)
    ctry = col[0].selectbox("Country", ["All"] + sorted(f.country.unique()))
    cat = col[1].selectbox("Category", ["All"] + sorted(f.category.unique()))
    buf = col[2].slider("Safety buffer on forecast (%)", 0, 30, int(round(b6["safety_buffer_pct"]))) / 100
    s = f.copy()
    if ctry != "All":
        s = s[s.country == ctry]
    if cat != "All":
        s = s[s.category == cat]
    best = b6["best_model"]
    nat = s.groupby("week")[["demand_units", "fc_current_plan", best]].sum().rename(
        columns={"demand_units": "Actual", "fc_current_plan": "Current plan", best: "Model forecast"})
    st.line_chart(nat)
    stock = s[best] * (1 + buf)
    lost = (s.demand_units - np.minimum(s.demand_units, stock)).clip(lower=0)
    left = (stock - np.minimum(s.demand_units, stock)).clip(lower=0)
    k = st.columns(4)
    k[0].metric("Stock-out weeks, current plan", f"{(s.lost_units > 0).mean() * 100:.1f}%")
    k[1].metric("Stock-out weeks, forecast plan", f"{(lost > 0.5).mean() * 100:.1f}%")
    k[2].metric("Lost sales, current vs forecast", f"{s.lost_units.sum() / s.demand_units.sum() * 100:.2f}% -> {lost.sum() / s.demand_units.sum() * 100:.2f}%")
    k[3].metric("Leftover stock, current vs forecast", f"{s.leftover_units.sum() / s.demand_units.sum() * 100:.1f}% -> {left.sum() / s.demand_units.sum() * 100:.1f}%")
    st.subheader("Next purchase day (Blog 5)")
    b5 = R["blog5"]
    st.dataframe(pd.DataFrame(b5["model_comparison"]).T, use_container_width=True)
    st.write(f"Tuned model test accuracy: **{b5['test_accuracy_tuned_pct']}%** (default {b5['test_accuracy_default_pct']}%)")

# ------------------------------------------------------------------------------------------------
elif page == "A/B tests (Blog 9)":
    st.title("Experiment design and results")
    ab = R["problem1_ab"]
    st.subheader("Problem 1 pilot: Riyadh, stratified by risk tier")
    c = st.columns(4)
    c[0].metric("Failed rate, test", f"{ab['failed_rate_test_pct']}%")
    c[1].metric("Failed rate, control", f"{ab['failed_rate_control_pct']}%")
    c[2].metric("t-statistic", ab["t_statistic"])
    c[3].metric("p-value", f"{ab['p_value']:.2g}")
    st.subheader("Problem 2 campaign: three arms")
    b9 = R["blog9_problem2"]
    st.write(f"One-way ANOVA on conversion: F = {b9['one_way_anova_conversion']['F']}, p = {b9['one_way_anova_conversion']['p_value']:.2g}")
    st.dataframe(pd.DataFrame(b9["two_way_anova"]), hide_index=True)
    st.dataframe(pd.DataFrame(b9["conversion_by_segment_and_offer_pct"]).T, use_container_width=True)
    st.subheader("Sample size calculator")
    col = st.columns(3)
    base = col[0].number_input("Baseline rate (%)", 1.0, 99.0, float(b9["sample_size_5pct_relative_lift"]["control_conversion_pct"])) / 100
    lift = col[1].number_input("Relative change to detect (%)", 1.0, 100.0, 5.0) / 100
    power = col[2].selectbox("Power", [0.8, 0.9], 0)
    from scipy import stats
    d = abs(base * lift) / np.sqrt(base * (1 - base))
    n = ((stats.norm.ppf(0.975) + stats.norm.ppf(power)) / d) ** 2 * 2
    st.metric("Customers needed per group", f"{int(np.ceil(n)):,}")

# ------------------------------------------------------------------------------------------------
elif page == "Model tuning":
    st.title("Hyperparameter tuning")
    t = load_csv("hyperparameter_tuning.csv")
    st.dataframe(t, hide_index=True, use_container_width=True)
    st.caption("Randomised or grid search with stratified K-fold (classifiers) or time-series split (forecasting).")

# ------------------------------------------------------------------------------------------------
else:
    st.title("About the data")
    st.write(R["data"])
    st.markdown(
        "- **customers**: 30,000 synthetic customers in the UAE, KSA and Egypt\n"
        "- **orders**: every order from Jan 2024 to Oct 2026 with payment method and delivery outcome\n"
        "- **campaign_retention**: randomised 3-arm test (No Offer / Price Match / Membership Trial)\n"
        "- **campaign_prepaid**: randomised AED 10 cashback-for-prepaid test\n"
        "- **daily_demand**: daily units by country and category with Ramadan, Eid and Yellow Friday flags\n\n"
        "Regenerate everything with `python src/generate_data.py` then `python src/analysis.py`.")
    for img in sorted(CH.glob("*.png")):
        st.image(str(img), caption=img.stem)
