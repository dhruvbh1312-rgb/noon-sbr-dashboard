"""
Noon SBR dashboard (Streamlit).

Reads the pre-computed outputs in results/ (made by src/analysis.py), adds what-if controls,
and runs a live operations page on a real-time simulated stream (src/live_stream.py).
Run locally:  streamlit run app.py
All data is SYNTHETIC and built for a university SBR exercise.
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
CH = ROOT / "charts"
sys.path.insert(0, str(ROOT / "src"))
import ui  # noqa: E402

st.set_page_config(page_title="Noon SBR Dashboard", page_icon="🟡", layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)


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
st.sidebar.markdown('<p class="side-brand">Noon growth room</p><p class="side-sub">SBR group project, Dubai</p>',
                    unsafe_allow_html=True)
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("Team: Harshita (CEO), Dhruv P Bhansali (Data), Abhinav Sharma (Marketing), "
                   "Shristi Murarka (CX), Sashidhar (Operations), Durga Venu (Finance)")
st.sidebar.caption(f"Models: {R['backend']['gbm']} | {R['backend']['sequence_model']}")


def aed(x):
    return f"AED {x:,.0f}"


def chart(c):
    st.altair_chart(c, use_container_width=True)


def nz(x, fmt, na="n/a"):
    return na if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, fmt)


# ================================================================================================
# LIVE OPERATIONS
# ================================================================================================
@st.cache_resource(show_spinner="Starting the live data stream (trains the refusal model, about 10 seconds)...")
def get_store():
    from live_stream import LiveStore
    return LiveStore()


ALERT_STYLE = {"Refusals rising": ("critical", "🚚"), "Stock-out risk": ("critical", "📦"),
               "Demand spike": ("", "📈"), "Price complaints": ("serious", "🏷️")}


def live_page():
    from live_stream import CATS, CITIES
    store = get_store()
    c = st.columns([1, 1, 0.7, 2.3])
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
            st.toast("Event triggered for 30 minutes. Alerts appear as the data comes in.")

    @st.fragment(run_every=None if paused else every)
    def feed():
        store.advance()
        snap = store.snapshot(window)
        k, t = snap["kpi"], snap["t"]
        res = snap["resolved"]
        base = res[(res.resolve_ts < t - timedelta(minutes=30)) & (res.resolve_ts >= t - timedelta(hours=3))]
        normal_fail = (base.status != "delivered").mean() if len(base) else np.nan

        # ---- hero
        st.markdown(f"""
<div class="hero">
  <div>
    <span class="live"><i></i>{'Paused' if paused else 'Live'}</span>
    <h1>Noon live operations</h1>
    <p>Orders, deliveries, customer reviews, rival prices and stock, streaming in real time. The alerts below are found
    by the models from the data alone.</p>
  </div>
  <div class="clock"><b>{t:%H:%M:%S}</b><span>Dubai, {t:%a %d %b %Y}. {k['pending']:,} deliveries on the road</span></div>
</div>""", unsafe_allow_html=True)

        # ---- newest customer voices
        rv = snap["reviews"]
        if len(rv):
            latest = rv.sort_values("ts", ascending=False).head(4)
            cards = []
            for r in latest.itertuples(index=False):
                cls = "spam" if r.is_spam_flag else r.sentiment
                label = "Flagged as fake" if r.is_spam_flag else r.sentiment
                cards.append(f'<div class="voice {cls}"><small>{r.ts:%H:%M} in {ui.esc(r.city)}, {ui.esc(r.category)}</small>'
                             f'<q dir="auto">{ui.esc(r.text)}</q><small>{ui.stars(r.stars)} &nbsp; {label}, {ui.esc(r.topic)}</small></div>')
            st.markdown(f'<div class="voices">{"".join(cards)}</div>', unsafe_allow_html=True)

        # ---- KPI groups by problem
        fail = k["cod_failed_rate"]
        fail_cls = "bad" if (not np.isnan(fail) and not np.isnan(normal_fail) and fail > normal_fail + 0.03) else ""
        g1, g2, g3 = st.columns(3)
        g1.markdown(f"""<div class="group"><h4>Deliveries <span>(Problem 1)</span></h4><div class="kpis">
<div class="kpi"><b class="{fail_cls}">{nz(fail * 100 if not np.isnan(fail) else np.nan, '.1f')}%</b><span>COD failed, normal {nz(normal_fail * 100 if not np.isnan(normal_fail) else np.nan, '.0f')}%</span></div>
<div class="kpi"><b>{k['cod_share'] * 100:.0f}%</b><span>paid cash on delivery</span></div>
<div class="kpi"><b>{k['high_risk_share'] * 100:.1f}%</b><span>orders flagged high risk</span></div></div></div>""",
                    unsafe_allow_html=True)
        ns = k["net_sentiment"]
        g2.markdown(f"""<div class="group"><h4>Customers <span>(Problem 2)</span></h4><div class="kpis">
<div class="kpi"><b>{nz(k['avg_stars'], '.2f')} <span class="stars">★</span></b><span>average rating</span></div>
<div class="kpi"><b class="{'good' if not np.isnan(ns) and ns > 0 else 'bad' if not np.isnan(ns) and ns < -10 else ''}">{nz(ns, '+.0f')}</b><span>net sentiment (positive minus negative %)</span></div>
<div class="kpi"><b>{k['spam_caught']}</b><span>fake reviews caught</span></div></div></div>""", unsafe_allow_html=True)
        g3.markdown(f"""<div class="group"><h4>Demand <span>(Problem 3)</span></h4><div class="kpis">
<div class="kpi"><b>{k['orders_per_min']:.1f}</b><span>orders a minute ({k['orders']:,} in window)</span></div>
<div class="kpi"><b>AED {k['revenue'] / 1000:,.0f}k</b><span>revenue in window</span></div>
<div class="kpi"><b class="{'bad' if k['lost_stockout'] else ''}">{k['lost_stockout']}</b><span>orders lost to stock-outs</span></div></div></div>""",
                    unsafe_allow_html=True)
        st.write("")

        # ---- alerts + orders chart
        left, right = st.columns([1, 1.6])
        with left:
            st.markdown("### Alerts")
            if snap["alerts"]:
                html = []
                for title, msg, prob in snap["alerts"][:6]:
                    cls, icon = ALERT_STYLE.get(title, ("", "⚠️"))
                    html.append(f'<div class="alert {cls}"><b>{icon} {ui.esc(title)}</b><span class="tag">{prob}</span>'
                                f'<small>{ui.esc(msg)}</small></div>')
                st.markdown("".join(html), unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert calm"><b>✅ All clear</b><small>No anomalies in the last 30 minutes.</small></div>',
                            unsafe_allow_html=True)
        o, ow = snap["orders"], snap["orders_w"]
        with right:
            st.markdown("### Orders per minute")
            if len(ow):
                pm = ow.set_index("ts").groupby("category").resample("1min").size().rename("orders").reset_index()
                chart(ui.lines_by_category(pm, "ts", "orders", "orders per minute", height=250))

        t1, t2, t3, t4 = st.tabs(["🚚 Deliveries", "💬 Customer reviews", "📦 Stock and rival prices", "🧪 Data quality and models"])
        with t1:
            a, b = st.columns([1, 1.3])
            rw = res[res.resolve_ts >= t - timedelta(minutes=window)]
            if len(rw):
                cf = rw.assign(f=rw.status != "delivered").groupby("city").f.mean().mul(100).round(1).rename("failed %").reset_index()
                with a:
                    st.markdown("**COD failed-delivery rate by city**")
                    nf = float(normal_fail * 100) if not np.isnan(normal_fail) else float(cf["failed %"].mean())
                    chart(ui.highlight_bars(cf, "city", "failed %", "% of COD deliveries failed", threshold=nf + 5,
                                            rule=nf, height=270))
                    st.caption("Red bars are more than 5 points above the normal rate (dashed line).")
            with b:
                st.markdown("**Newest COD orders, scored by the refusal-risk model**")
                cod = o[o.payment_method == "COD"].tail(12).iloc[::-1]
                if len(cod):
                    show = cod[["ts", "city", "category", "order_value_aed", "first_order", "address_quality", "risk", "status"]].copy()
                    show["action"] = np.select([show.risk >= 0.40, show.risk >= 0.20],
                                               ["🔴 confirm + prepaid nudge", "🟠 WhatsApp confirm"], "🟢 none")
                    show["ts"] = show.ts.dt.strftime("%H:%M:%S")
                    st.dataframe(show, hide_index=True, use_container_width=True, height=300,
                                 column_config={"risk": st.column_config.ProgressColumn("risk", min_value=0, max_value=1, format="%.2f"),
                                                "order_value_aed": st.column_config.NumberColumn("value (AED)", format="%.0f")})
        with t2:
            rvw = snap["reviews_w"]
            if len(rvw):
                real = rvw[~rvw.is_spam_flag]
                a, b = st.columns(2)
                with a:
                    st.markdown("**Negative reviews over time (% of genuine reviews)**")
                    tr = real.set_index("ts").resample("5min").sentiment.apply(
                        lambda s: (s == "Negative").mean() * 100 if len(s) else np.nan).rename("negative %").reset_index()
                    chart(ui.single_line(tr.dropna(), "ts", "negative %", "% negative", color=ui.CRITICAL, height=240, fmt=".0f"))
                with b:
                    st.markdown("**What customers talk about**")
                    ts_ = real.groupby(["topic", "sentiment"]).size().rename("n").reset_index()
                    ts_["order"] = ts_.sentiment.map({s: i for i, s in enumerate(ui.SENT)})
                    chart(ui.stacked_sentiment(ts_, "topic", height=240))
                st.markdown("**Latest reviews, exactly as written, with what the analyser concluded**")
                rows = []
                for r in rvw.sort_values("ts", ascending=False).head(14).itertuples(index=False):
                    pill = '<span class="pill spam">fake</span>' if r.is_spam_flag else f'<span class="pill {r.sentiment}">{r.sentiment}</span>'
                    rows.append(f'<div class="row"><time>{r.ts:%H:%M}</time><div><div class="txt" dir="auto">{ui.esc(r.text)}</div>'
                                f'<div class="meta">{ui.stars(r.stars)} &nbsp;{ui.esc(r.city)}, {ui.esc(r.category)}, topic: {ui.esc(r.topic)}, '
                                f'{ui.esc(r.language)}</div></div>{pill}</div>')
                st.markdown(f'<div class="feed">{"".join(rows)}</div>', unsafe_allow_html=True)
        with t3:
            stock = pd.Series(snap["stock"]).rename("units").reset_index()
            stock.columns = ["country", "category", "units"]
            st.markdown("**Units on hand by country** (refilled hourly: forecast x 4 hours x 1.08 buffer)")
            chart(ui.bars_by_category(stock, "category", "units", "units", facet="country", height=190))
            pr = snap["prices"]
            if len(pr):
                st.markdown("**Rival price index by category** (100 = our price; below 100 means the rival is cheaper)")
                pl = pr[pr.ts >= t - timedelta(minutes=window)].melt(id_vars="ts", value_vars=CATS, var_name="category", value_name="index")
                chart(ui.lines_by_category(pl, "ts", "index", "rival price index", height=240, rule_y=100))
        with t4:
            q, mo = snap["quality"], snap["monitor"]
            a = st.columns(4)
            a[0].metric("Raw order records received", f"{q['raw_rows']:,}")
            a[1].metric("Missing city filled", q["missing_city"])
            a[2].metric("Negative values fixed", q["negative_value"])
            a[3].metric("Duplicate records removed", q["duplicates"])
            b = st.columns(4)
            b[0].metric("Refusal model live ROC-AUC", nz(mo["risk_auc"], ".3f"))
            b[1].metric("Sentiment accuracy", nz(mo["sentiment_acc"] * 100 if not np.isnan(mo["sentiment_acc"]) else np.nan, ".1f") + "%")
            b[2].metric("Fake reviews caught (recall)", nz(mo["spam_recall"] * 100 if not np.isnan(mo["spam_recall"]) else np.nan, ".0f") + "%")
            b[3].metric("Flags that were truly fake (precision)", nz(mo["spam_precision"] * 100 if not np.isnan(mo["spam_precision"]) else np.nan, ".0f") + "%")
            st.caption("Accuracy compares the analysers with the simulator's hidden truth, which a real company would get from manual audits.")
            with st.expander("Answer key: events actually running (hidden from the alert engine)"):
                act = [s for s in snap["shocks"] if s.end >= t - timedelta(minutes=window)]
                st.write([f"{s.label()} {s.start:%H:%M} to {s.end:%H:%M}{' (manual)' if s.manual else ''}" for s in act] or "None")

    feed()


if page == LIVE:
    live_page()

# ================================================================================================
elif page == "Overview":
    p1, b6, b8 = R["problem1_impact"], R["blog6"], R["blog8_retention"]["policies"]
    st.markdown("""<div class="banner"><h1>Noon: data-driven growth</h1>
<p>Three problems, the nine Data Driven Growth techniques, and what we predict each solution will change.
All figures come from a synthetic dataset that behaves like a Gulf marketplace.</p></div>""", unsafe_allow_html=True)
    acc = b6["accuracy"]
    best = b6["best_model"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"""<div class="result"><h3>🚚 Cash on delivery refusals</h3>
<div class="ba"><span class="before">{p1['failed_rate_before_pct']:.1f}%</span><span class="after">{p1['failed_rate_after_pct']:.1f}%</span></div>
<p>COD orders that fail. Risk tiers plus WhatsApp confirmation cut failures by {p1['relative_reduction_pct']:.0f}%.</p></div>""",
                unsafe_allow_html=True)
    c2.markdown(f"""<div class="result"><h3>🏷️ Customers switching to rivals</h3>
<div class="ba"><span class="before">AED {b8['Price Match to everyone']['incremental_vs_no_offer_aed']:+.2f}</span><span class="after">AED {b8['Net value X-Learner']['incremental_vs_no_offer_aed']:+.2f}</span></div>
<p>Profit per customer from retention offers: blanket price match vs net-value targeting of {b8['Net value X-Learner']['share_offered_pct']:.0f}% of customers.</p></div>""",
                unsafe_allow_html=True)
    c3.markdown(f"""<div class="result"><h3>📦 Ramadan and Yellow Friday spikes</h3>
<div class="ba"><span class="before">{acc['fc_current_plan']['WAPE_peak_weeks_pct']:.1f}%</span><span class="after">{acc[best]['WAPE_peak_weeks_pct']:.1f}%</span></div>
<p>Forecast error in peak weeks: "last year x 1.2" vs the event-aware forecast.</p></div>""", unsafe_allow_html=True)
    st.write("")
    m = load_csv("monthly_metrics.csv", index_col=0).reset_index().rename(columns={"index": "month", "ym": "month"})
    m["month"] = pd.to_datetime(m.iloc[:, 0])
    a, b = st.columns([1.6, 1])
    with a:
        st.markdown("### North Star: monthly revenue")
        chart(ui.single_line(m, "month", "revenue", "AED", color="#2a78d6", height=260))
    with b:
        st.markdown("### Blogs used per problem")
        st.dataframe(pd.DataFrame({
            "Problem": ["1. COD refusals", "2. Switching to rivals", "3. Demand spikes"],
            "Blogs": ["1, 2, 4, 8, 9", "1, 2, 3, 4, 7, 8, 9", "1, 2, 5, 6, 9"]}), hide_index=True, use_container_width=True)
        st.caption("Open 🔴 Live operations to watch the techniques run on streaming data.")

# ================================================================================================
elif page == "1. Metrics (Blog 1)":
    st.title("Know your metrics")
    b1 = R["blog1"]
    st.info(b1["north_star"])
    c = st.columns(4)
    c[0].metric("Revenue, last 12 months", aed(b1["revenue_last12_aed"]), f"{b1['yoy_growth_pct']}% YoY")
    c[1].metric("Monthly retention (avg, 12m)", f"{b1['avg_monthly_retention_last12_pct']}%")
    c[2].metric("New customer ratio (avg, 12m)", f"{b1['avg_new_customer_ratio_last12_pct']}%")
    c[3].metric("Delivery success rate", f"{b1['delivery_success_last12_pct']}%")
    m = load_csv("monthly_metrics.csv", index_col=0).reset_index()
    m["month"] = pd.to_datetime(m.iloc[:, 0])
    names = {"revenue": "Revenue (AED)", "growth_rate_pct": "Revenue growth (%)", "active_customers": "Active customers",
             "orders": "Orders", "avg_revenue_per_order": "Average revenue per order (AED)", "new_customer_ratio_pct": "New customer ratio (%)",
             "retention_rate_pct": "Monthly retention (%)", "cod_share_pct": "COD share (%)", "cod_failed_rate_pct": "COD failed rate (%)",
             "delivery_success_rate_pct": "Delivery success (%)", "failed_delivery_cost_aed": "Failed delivery cost (AED)"}
    metric = st.selectbox("Metric", list(names), format_func=names.get)
    color = ui.CRITICAL if metric in ("cod_failed_rate_pct", "failed_delivery_cost_aed") else "#2a78d6"
    chart(ui.single_line(m.dropna(subset=[metric]), "month", metric, names[metric], color=color, height=280, fmt=",.1f"))
    st.markdown("### Cohort retention")
    st.caption("% of each joining month still buying k months later. Darker = more customers kept.")
    coh = load_csv("cohort_retention.csv", index_col=0)
    st.dataframe(coh.iloc[:, :13].style.background_gradient(cmap="Blues", axis=None).format("{:.0f}", na_rep=""),
                 use_container_width=True)

# ================================================================================================
elif page == "2. Segments (Blogs 2, 3)":
    st.title("RFM segmentation and lifetime value")
    rfm = load_csv("rfm_segments.csv", index_col=0)
    seg = pd.DataFrame(R["blog2"]["segments"])
    seg_colors = {"High-Value": "#2a78d6", "Mid-Value": "#1baf7a", "Low-Value": "#B9B5D6"}
    c = st.columns(3)
    for col, (_, r) in zip(c, seg.set_index("Segment").loc[list(seg_colors)].iterrows()):
        col.metric(f"{r.name}: {int(r.customers):,} customers", f"{r.revenue_share_pct:.1f}% of revenue",
                   f"avg {r.frequency:.0f} orders, AED {r.revenue:,.0f}")
    a, b = st.columns(2)
    with a:
        st.markdown("**Elbow method on Recency** (the bend at k = 3 to 4 picks the cluster count)")
        el = pd.DataFrame({"k": [int(x) for x in R["blog2"]["elbow_inertia"]], "inertia": list(R["blog2"]["elbow_inertia"].values())})
        line = alt.Chart(el).mark_line(color="#2a78d6", strokeWidth=2).encode(x=alt.X("k:O", title="k"), y=alt.Y("inertia:Q", title="inertia (millions)"))
        pts = alt.Chart(el).mark_circle(size=70, color="#2a78d6").encode(x="k:O", y="inertia:Q", tooltip=["k", alt.Tooltip("inertia:Q", format=".1f")])
        hi = alt.Chart(el[el.k == 4]).mark_circle(size=180, color=ui.SAFFRON).encode(x="k:O", y="inertia:Q")
        chart(ui.finish(alt.layer(line, pts, hi), 260))
    with b:
        st.markdown("**Problem 2 behaviour segments**")
        p2 = pd.DataFrame(R["blog2"]["p2_segments"])
        chart(ui.highlight_bars(p2, "P2_segment", "customers", "customers", threshold=None, height=260, fmt=",.0f"))
    country = st.multiselect("Country", sorted(rfm.country.unique()), default=sorted(rfm.country.unique()))
    sub = rfm[rfm.country.isin(country)].sample(min(3000, len(rfm[rfm.country.isin(country)])), random_state=1)
    sc = alt.Chart(sub).mark_circle(size=28, opacity=0.6).encode(
        x=alt.X("Frequency:Q", scale=alt.Scale(type="log"), title="orders (log scale)"),
        y=alt.Y("Revenue:Q", scale=alt.Scale(type="symlog"), title="revenue AED (log scale)"),
        color=alt.Color("Segment:N", scale=alt.Scale(domain=list(seg_colors), range=list(seg_colors.values())), title=None),
        tooltip=["Segment", "Recency", "Frequency", alt.Tooltip("Revenue:Q", format=",.0f"), "country"])
    st.markdown("**Every customer by orders and revenue, coloured by RFM segment**")
    chart(ui.finish(sc, 340))
    st.markdown("### CLV prediction (Blog 3)")
    b3 = R["blog3"]
    c = st.columns(3)
    c[0].metric("Benchmark accuracy (largest class)", f"{b3['benchmark_accuracy_pct']}%")
    c[1].metric("Default model", f"{b3['test_accuracy_default_pct']}%")
    c[2].metric("Tuned model", f"{b3['test_accuracy_tuned_pct']}%", f"+{b3['test_accuracy_tuned_pct'] - b3['benchmark_accuracy_pct']:.1f} pts vs benchmark")
    st.dataframe(pd.DataFrame(b3["ltv_clusters"]), hide_index=True)

# ================================================================================================
elif page == "Problem 1: COD refusals":
    st.title("🚚 Problem 1: cash on delivery refusals")
    b4 = R["blog4_refusal"]
    c = st.columns(4)
    c[0].metric("COD failed rate (test period)", f"{b4['failed_rate_test_pct']}%")
    c[1].metric("Refusal model ROC-AUC", b4["tuned_gbm_auc"], f"+{b4['tuned_gbm_auc'] - b4['decision_tree_auc']:.3f} vs tree")
    c[2].metric("Failures caught in riskiest 10%", f"{b4['top10pct_capture_pct']}%")
    c[3].metric("Failures caught in riskiest 30%", f"{b4['top30pct_capture_pct']}%")
    lift = pd.DataFrame(b4["lift_table"])
    lift["failed %"] = (lift.failed_rate * 100).round(1)
    lift["decile"] = lift.decile.astype(str)
    st.markdown("**Failed-delivery rate by risk decile** (1 = riskiest 10% of orders; red = above the average, dashed)")
    chart(ui.highlight_bars(lift, "decile", "failed %", "% failed", threshold=b4["failed_rate_test_pct"], horizontal=False,
                            rule=b4["failed_rate_test_pct"], height=260))

    st.markdown("### What-if: risk-tier policy")
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
    after = np.where(s.tier == "Medium", s.failed * (1 - eff), s.failed.astype(float))
    after = np.where(s.tier == "High", s.failed * (1 - switch) * (1 - eff) + switch * 0.012, after)
    msgs = (s.tier != "Low").sum()
    months = max((s.order_date.max() - s.order_date.min()).days / 30.4, 1)
    saving = ((s.failed.sum() - after.sum()) * cost - msgs * 0.15) * 12 / months
    k = st.columns(3)
    k[0].metric("Failed rate today", f"{s.failed.mean() * 100:.2f}%")
    k[1].metric("Failed rate with policy", f"{after.mean() * 100:.2f}%", f"{(after.mean() - s.failed.mean()) * 100:.2f} pts",
                delta_color="inverse")
    k[2].metric("Net saving per year", aed(saving))
    tiers = s.groupby("tier").agg(orders=("failed", "size"), failed=("failed", "mean")).reset_index()
    tiers["failed %"] = (tiers.failed * 100).round(1)
    tier_colors = {"Low": ui.GOOD, "Medium": ui.SERIOUS, "High": ui.CRITICAL}
    chart(ui.colored_bars(tiers, "tier", "failed %", "% of orders failing", tier_colors, height=200, fmt=".1f"))
    st.markdown("### Who should get the AED 10 prepaid cashback? (Blog 8)")
    pol = pd.DataFrame(R["blog8_prepaid"]["policies"]).T.reset_index().rename(columns={"index": "policy"})
    chart(ui.diverging_bars(pol, "policy", "incremental_vs_no_one_aed", "AED per customer vs no cashback", height=220))
    st.caption("Blanket cashback loses money because many customers would prepay anyway. Net-value targeting keeps the offer "
               "only where the saved delivery cost beats the cashback.")

# ================================================================================================
elif page == "Problem 2: Switching to rivals":
    st.title("🏷️ Problem 2: customers switching to rivals")
    ch = R["blog4_churn"]
    c = st.columns(4)
    c[0].metric("Churn rate (90 days)", f"{ch['churn_rate_pct']}%")
    c[1].metric("Churn model ROC-AUC", ch["tuned_gbm"]["roc_auc"])
    c[2].metric("Recall / precision", f"{ch['tuned_gbm']['recall']} / {ch['tuned_gbm']['precision']}")
    c[3].metric("Churners caught in top 20%", f"{ch['top20pct_capture_of_churners_pct']}%")
    a, b = st.columns(2)
    with a:
        st.markdown("**Market response (Blog 7): conversion by offer**")
        conv = pd.DataFrame({"offer": ["No Offer", "Price Match", "Membership Trial"],
                             "conversion %": [R["blog7"]["control_conversion_pct"]] +
                             [R["blog7"]["uplift"][o]["conversion_rate_pct"] for o in ["Price Match", "Membership Trial"]]})
        chart(ui.colored_bars(conv, "offer", "conversion %", "% ordering within 30 days", ui.OFFER_COLORS, height=250, fmt=".1f"))
        up = pd.DataFrame(R["blog7"]["uplift"]).T
        st.dataframe(up, use_container_width=True)
    with b:
        st.markdown("**Policy comparison (Blog 8): extra profit per customer vs no offer**")
        pol = pd.DataFrame(R["blog8_retention"]["policies"]).T.reset_index().rename(columns={"index": "policy"})
        chart(ui.diverging_bars(pol, "policy", "incremental_vs_no_offer_aed", "AED per customer", height=320))
    st.markdown("### What-if: change the offer costs")
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
    k[2].metric("Deal hunters offered price match",
                f"{((best == 'Price Match') & (u.discount_share >= 0.6)).sum() / max((u.discount_share >= 0.6).sum(), 1) * 100:.1f}%")
    mix = pd.Series(best).value_counts().rename_axis("offer").rename("customers").reset_index()
    chart(ui.colored_bars(mix, "offer", "customers", "customers", ui.OFFER_COLORS, height=220))

# ================================================================================================
elif page == "Problem 3: Demand spikes":
    st.title("📦 Problem 3: Ramadan and Yellow Friday demand spikes")
    b6 = R["blog6"]
    acc = pd.DataFrame(b6["accuracy"]).T
    names = {"fc_current_plan": "Current plan (last year x 1.2)", "fc_naive_last_week": "Last week (naive)",
             "fc_sequence": "Neural network (LSTM stand-in)", "fc_gbm": "Gradient boosting", "fc_ensemble": "50/50 average"}
    acc.index = acc.index.map(names)
    a, b = st.columns([1, 1.2])
    with a:
        st.markdown("**Forecast error in peak weeks (WAPE, lower is better)**")
        w = acc["WAPE_peak_weeks_pct"].rename("WAPE %").reset_index().rename(columns={"index": "forecast"})
        chart(ui.highlight_bars(w, "forecast", "WAPE %", "% error in peak weeks", threshold=20, hi_color=ui.CRITICAL,
                                lo_color="#2a78d6", height=230, sort="x"))
    with b:
        st.dataframe(acc, use_container_width=True)
        st.caption(f"Adjusted R² of weekly changes: lags only {b6['adjusted_r2']['lag_1..12']:.2f}, lags + events {b6['adjusted_r2']['lag_1..12 + events']:.2f}.")
    f = load_csv("forecast_weekly.csv", parse_dates=["week"])
    col = st.columns(3)
    ctry = col[0].selectbox("Country", ["All"] + sorted(f.country.unique()))
    cat = col[1].selectbox("Category", ["All"] + sorted(f.category.unique()))
    buf = col[2].slider("Safety buffer on forecast (%)", 0, 30, 8) / 100
    s = f.copy()
    if ctry != "All":
        s = s[s.country == ctry]
    if cat != "All":
        s = s[s.category == cat]
    best = b6["best_model"]
    nat = s.groupby("week")[["demand_units", "fc_current_plan", best]].sum().rename(
        columns={"demand_units": "Actual demand", "fc_current_plan": "Current plan", best: "Model forecast"})
    long = nat.reset_index().melt(id_vars="week", var_name="series", value_name="units")
    st.markdown("**Weekly units in 2026** (Ramadan ran 18 Feb to 19 Mar; Eid al-Adha around 27 May)")
    chart(ui.forecast_chart(long, height=300))
    stock = s[best] * (1 + buf)
    lost = (s.demand_units - np.minimum(s.demand_units, stock)).clip(lower=0)
    left = (stock - np.minimum(s.demand_units, stock)).clip(lower=0)
    k = st.columns(4)
    k[0].metric("Stock-out weeks, current plan", f"{(s.lost_units > 0).mean() * 100:.1f}%")
    k[1].metric("Stock-out weeks, forecast plan", f"{(lost > 0.5).mean() * 100:.1f}%")
    k[2].metric("Lost sales", f"{lost.sum() / s.demand_units.sum() * 100:.2f}%",
                f"{(lost.sum() - s.lost_units.sum()) / s.demand_units.sum() * 100:.2f} pts vs current", delta_color="inverse")
    k[3].metric("Leftover stock", f"{left.sum() / s.demand_units.sum() * 100:.1f}%",
                f"{(left.sum() - s.leftover_units.sum()) / s.demand_units.sum() * 100:.1f} pts vs current", delta_color="inverse")
    st.markdown("### Next purchase day (Blog 5)")
    b5 = R["blog5"]
    mc = pd.DataFrame(b5["model_comparison"]).T.reset_index().rename(columns={"index": "model"})
    mc["accuracy %"] = (mc.cv_accuracy_mean * 100).round(1)
    chart(ui.highlight_bars(mc, "model", "accuracy %", "cross-validated accuracy (%)", threshold=66, hi_color=ui.GOOD,
                            lo_color=ui.MUTED_BAR, height=220))
    st.write(f"Tuned model test accuracy: **{b5['test_accuracy_tuned_pct']}%** (default {b5['test_accuracy_default_pct']}%)")

# ================================================================================================
elif page == "A/B tests (Blog 9)":
    st.title("🧪 Experiment design and results")
    ab = R["problem1_ab"]
    st.markdown("### Problem 1 pilot: Riyadh, stratified by risk tier")
    c = st.columns(4)
    c[0].metric("Failed rate, test", f"{ab['failed_rate_test_pct']}%", f"{ab['failed_rate_test_pct'] - ab['failed_rate_control_pct']:.2f} pts",
                delta_color="inverse")
    c[1].metric("Failed rate, control", f"{ab['failed_rate_control_pct']}%")
    c[2].metric("t-statistic", ab["t_statistic"])
    c[3].metric("p-value", f"{ab['p_value']:.2g}", "significant" if ab["significant"] else "not significant",
                delta_color="normal" if ab["significant"] else "off")
    st.markdown("### Problem 2 campaign: three arms")
    b9 = R["blog9_problem2"]
    st.write(f"One-way ANOVA on conversion: **F = {b9['one_way_anova_conversion']['F']}**, p = {b9['one_way_anova_conversion']['p_value']:.2g}")
    a, b = st.columns([1.3, 1])
    with a:
        cs = pd.DataFrame(b9["conversion_by_segment_and_offer_pct"]).T.reset_index().rename(columns={"index": "segment"})
        cs = cs.melt(id_vars="segment", var_name="offer", value_name="conversion %")
        gb = alt.Chart(cs).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("offer:N", title=None, sort=list(ui.OFFER_COLORS), axis=alt.Axis(labels=False, ticks=False)),
            y=alt.Y("conversion %:Q", title="% converting"),
            color=alt.Color("offer:N", scale=alt.Scale(domain=list(ui.OFFER_COLORS), range=list(ui.OFFER_COLORS.values())), title=None),
            column=alt.Column("segment:N", title=None), tooltip=["segment", "offer", alt.Tooltip("conversion %:Q", format=".1f")]
        ).properties(width=150, height=220)
        st.markdown("**Conversion by segment and offer** (the interaction term in the two-way ANOVA)")
        st.altair_chart(gb.configure(font="Readex Pro", background=ui.CARD).configure_view(stroke=None)
                        .configure_legend(orient="top"), use_container_width=False)
    with b:
        st.dataframe(pd.DataFrame(b9["two_way_anova"]), hide_index=True)
    st.markdown("### Sample size calculator")
    col = st.columns(3)
    base = col[0].number_input("Baseline rate (%)", 1.0, 99.0, float(b9["sample_size_5pct_relative_lift"]["control_conversion_pct"])) / 100
    lift = col[1].number_input("Relative change to detect (%)", 1.0, 100.0, 5.0) / 100
    power = col[2].selectbox("Power", [0.8, 0.9], 0)
    from scipy import stats
    d = abs(base * lift) / np.sqrt(base * (1 - base))
    n = ((stats.norm.ppf(0.975) + stats.norm.ppf(power)) / d) ** 2 * 2
    st.metric("Customers needed per group", f"{int(np.ceil(n)):,}")

# ================================================================================================
elif page == "Model tuning":
    st.title("Hyperparameter tuning")
    t = load_csv("hyperparameter_tuning.csv")
    t["gain"] = (t.cv_tuned.abs() - t.cv_default.abs()).abs()
    st.dataframe(t, hide_index=True, use_container_width=True)
    st.caption("Randomised or grid search with stratified K-fold (classifiers) or time-series split (forecasting).")

# ================================================================================================
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
