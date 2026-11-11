"""
Synthetic data generator for the Noon SBR project.

All data is SYNTHETIC. It is built to behave like a Middle East e-commerce
business (UAE, KSA, Egypt) so that the techniques from the 9 "Data Driven
Growth" blogs can be demonstrated end to end. No real Noon data is used.

Files written to ../data:
  customers.csv          one row per customer (30,000)
  orders.csv.gz          one row per order, Jan 2024 to Oct 2026
  campaign_retention.csv randomised 3-arm campaign (Problem 2): price-match / membership trial / no offer
  campaign_prepaid.csv   randomised cashback-for-prepaid test (Problem 1)
  daily_demand.csv       daily units by country x category with the current (naive) stock plan (Problem 3)

Run:  python src/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2026-10-31")
DAYS = pd.date_range(START, END, freq="D")
N_DAYS = len(DAYS)

# ---------------------------------------------------------------------------
# Calendar: Ramadan moves about 11 days earlier every year, which is exactly
# why "same period last year" planning fails.
# ---------------------------------------------------------------------------
RAMADAN = [("2024-03-11", "2024-04-09"), ("2025-03-01", "2025-03-29"), ("2026-02-18", "2026-03-19")]
EID_FITR = ["2024-04-10", "2025-03-30", "2026-03-20"]
EID_ADHA = ["2024-06-16", "2025-06-06", "2026-05-27"]
YELLOW_FRIDAY = ["2024-11-29", "2025-11-28"]  # Yellow Friday 2026 is after the data ends
BACK_TO_SCHOOL = [("2024-08-15", "2024-09-05"), ("2025-08-14", "2025-09-04"), ("2026-08-13", "2026-09-03")]


def build_calendar() -> pd.DataFrame:
    cal = pd.DataFrame({"date": DAYS})
    cal["ramadan"] = 0
    cal["pre_eid"] = 0  # last 10 days before an Eid: gifting / fashion / electronics rush
    cal["eid"] = 0
    cal["yellow_friday"] = 0
    cal["back_to_school"] = 0
    for s, e in RAMADAN:
        cal.loc[cal.date.between(s, e), "ramadan"] = 1
    for d in EID_FITR + EID_ADHA:
        d = pd.Timestamp(d)
        cal.loc[cal.date.between(d - pd.Timedelta(days=10), d - pd.Timedelta(days=1)), "pre_eid"] = 1
        cal.loc[cal.date.between(d, d + pd.Timedelta(days=2)), "eid"] = 1
    for d in YELLOW_FRIDAY:
        d = pd.Timestamp(d)
        # sale window: 7 days before Yellow Friday to the Monday after
        cal.loc[cal.date.between(d - pd.Timedelta(days=7), d + pd.Timedelta(days=3)), "yellow_friday"] = 1
    for s, e in BACK_TO_SCHOOL:
        cal.loc[cal.date.between(s, e), "back_to_school"] = 1
    cal["payday"] = cal.date.dt.day.isin([25, 26, 27, 28, 1]).astype(int)
    cal["weekend"] = cal.date.dt.dayofweek.isin([4, 5]).astype(int)  # Fri / Sat
    years = (cal.date - START).dt.days / 365.25
    cal["trend"] = 1 + 0.22 * years
    return cal


CAL = build_calendar()


def demand_multiplier(cal: pd.DataFrame) -> np.ndarray:
    m = np.ones(len(cal))
    m *= 1 + 0.30 * cal.ramadan
    m *= 1 + 0.80 * cal.pre_eid
    m *= 1 - 0.25 * cal.eid  # people are busy on Eid itself
    m *= 1 + 2.20 * cal.yellow_friday
    m *= 1 + 0.30 * cal.back_to_school
    m *= 1 + 0.12 * cal.payday
    m *= 1 + 0.08 * cal.weekend
    return (m * cal.trend).to_numpy()


SEASON = demand_multiplier(CAL)
SEASON_MAX = SEASON.max()

# ---------------------------------------------------------------------------
# Customers
# ---------------------------------------------------------------------------
N_CUST = 30_000
COUNTRIES = np.array(["UAE", "KSA", "EGY"])
CITIES = {"UAE": ["Dubai", "Abu Dhabi", "Sharjah"], "KSA": ["Riyadh", "Jeddah", "Dammam"], "EGY": ["Cairo", "Alexandria"]}
COD_BASE = {"UAE": 0.30, "KSA": 0.50, "EGY": 0.72}


def make_customers() -> pd.DataFrame:
    country = rng.choice(COUNTRIES, N_CUST, p=[0.45, 0.40, 0.15])
    city = np.array([rng.choice(CITIES[c]) for c in country])
    # sign-ups grow over time
    signup_idx = np.minimum((rng.beta(1.3, 1.0, N_CUST) * (N_DAYS - 30)).astype(int), N_DAYS - 31)
    signup = DAYS[signup_idx]
    channel = rng.choice(["Organic", "Paid Social", "Search", "Referral"], N_CUST, p=[0.35, 0.30, 0.25, 0.10])
    engagement = rng.lognormal(mean=0.0, sigma=0.65, size=N_CUST)  # ~1 order / month on average
    price_sens = rng.beta(2.0, 2.5, N_CUST)  # 0 = loyal, 1 = pure deal hunter
    member = (rng.random(N_CUST) < 0.18 * (1.3 - price_sens)).astype(int)
    cod_prop = np.clip(np.array([COD_BASE[c] for c in country]) + rng.normal(0, 0.18, N_CUST), 0.02, 0.98)
    addr_q = rng.choice(["good", "medium", "poor"], N_CUST, p=[0.62, 0.26, 0.12])
    refusal_tendency = rng.normal(0, 1, N_CUST)
    card_saved = (rng.random(N_CUST) < (0.85 - cod_prop * 0.8)).astype(int)
    # lifetime before the customer drifts away to a rival: shorter for deal hunters, longer for members
    life_mean = 650 - 420 * price_sens + 300 * member
    lifetime = rng.exponential(life_mean)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, N_CUST + 1),
            "country": country,
            "city": city,
            "signup_date": signup,
            "acq_channel": channel,
            "member": member,
            "address_quality": addr_q,
            "card_saved": card_saved,
            # latent fields (kept for transparency; models never see these)
            "_engagement": engagement.round(4),
            "_price_sensitivity": price_sens.round(4),
            "_cod_propensity": cod_prop.round(4),
            "_refusal_tendency": refusal_tendency.round(4),
            "_lifetime_days": lifetime.round(1),
        }
    )


CATS = np.array(["Electronics", "Fashion", "Beauty", "Grocery", "Home"])
CAT_P = np.array([0.20, 0.27, 0.16, 0.22, 0.15])
CAT_MEAN_AED = {"Electronics": 620, "Fashion": 190, "Beauty": 125, "Grocery": 160, "Home": 260}
FAILED_DELIVERY_COST_AED = 25.0  # two trips + reverse handling + blocked stock (assumption)


def make_orders(cust: pd.DataFrame) -> pd.DataFrame:
    rows = []
    day_index = {d: i for i, d in enumerate(DAYS)}
    yf_all = CAL.yellow_friday.to_numpy()
    for cid_, sd, life, eng, ps in zip(
        cust.customer_id, cust.signup_date, cust._lifetime_days, cust._engagement, cust._price_sensitivity
    ):
        s = day_index[sd]
        e = min(N_DAYS - 1, s + int(life))
        span = e - s + 1
        lam = eng / 30.0
        n_cand = rng.poisson(lam * SEASON_MAX * span)
        cand = s + rng.integers(0, span, n_cand)
        # deal hunters respond more to Yellow Friday
        mult = SEASON[cand].copy()
        mult[yf_all[cand] == 1] *= 0.6 + 0.9 * ps
        keep = rng.random(n_cand) < np.minimum(mult / SEASON_MAX, 1)
        days = np.sort(cand[keep])
        if len(days) == 0:
            days = np.array([s])  # every customer places at least the sign-up order
        for k, di in enumerate(days):
            rows.append((cid_, di, k == 0, 0))
    o = pd.DataFrame(rows, columns=["customer_id", "day_idx", "first_order", "prior_refusals_tmp"])
    o = o.merge(cust, on="customer_id", how="left")
    n = len(o)
    o["order_date"] = DAYS[o.day_idx.to_numpy()]
    cal = CAL.set_index("date")
    ram = cal.loc[o.order_date, "ramadan"].to_numpy()
    yfw = cal.loc[o.order_date, "yellow_friday"].to_numpy()
    pre_eid = cal.loc[o.order_date, "pre_eid"].to_numpy()

    # category mix shifts by season
    p = np.tile(CAT_P, (n, 1))
    p[ram == 1] *= np.array([0.9, 1.0, 1.0, 1.7, 1.1])
    p[pre_eid == 1] *= np.array([1.4, 1.6, 1.3, 0.8, 1.0])
    p[yfw == 1] *= np.array([1.8, 1.2, 1.1, 0.6, 1.2])
    p /= p.sum(1, keepdims=True)
    u = rng.random(n)[:, None]
    o["category"] = CATS[(u > p.cumsum(1)).sum(1)]
    mean_val = o.category.map(CAT_MEAN_AED).to_numpy()
    o["order_value_aed"] = np.round(rng.lognormal(np.log(mean_val) - 0.18, 0.6), 2)

    disc_p = np.clip(0.10 + 0.55 * o._price_sensitivity + 0.30 * yfw, 0, 0.97)
    o["discount_used"] = (rng.random(n) < disc_p).astype(int)

    # payment method: COD slowly declines over time
    t = (o.order_date - START).dt.days / 1000
    cod_p = np.clip(o._cod_propensity - 0.06 * t, 0.01, 0.99)
    o["payment_method"] = np.where(rng.random(n) < cod_p, "COD", "Prepaid")

    hour = rng.choice(np.arange(24), n, p=np.r_[np.full(7, 0.015), np.full(10, 0.05), np.full(7, 0.0564)][:24] / np.r_[np.full(7, 0.015), np.full(10, 0.05), np.full(7, 0.0564)][:24].sum())
    o["order_hour"] = hour

    # delivery outcome, sequential per customer because past refusals matter
    o = o.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
    addr = o.address_quality.map({"good": 0.0, "medium": 0.45, "poor": 1.15}).to_numpy()
    ctry = o.country.map({"UAE": 0.0, "KSA": 0.25, "EGY": 0.45}).to_numpy()
    late = ((o.order_hour >= 22) | (o.order_hour <= 2)).astype(int).to_numpy()
    base_logit = (
        -3.45
        + 1.05 * o.first_order.to_numpy()
        + addr
        + 0.55 * np.log(o.order_value_aed.to_numpy() / 200)
        + 0.45 * late
        + 1.25 * o._refusal_tendency.to_numpy()
        + ctry
        + 0.35 * yfw
        - 0.4 * o.member.to_numpy()
    )
    is_cod = (o.payment_method == "COD").to_numpy()
    outcome = np.array(["delivered"] * len(o), dtype=object)
    prior = np.zeros(len(o), dtype=int)
    cid = o.customer_id.to_numpy()
    u = rng.random(len(o))
    u2 = rng.random(len(o))
    count = 0
    last = -1
    for i in range(len(o)):
        if cid[i] != last:
            count = 0
            last = cid[i]
        prior[i] = count
        lg = base_logit[i] + 0.55 * min(count, 3)
        p_fail = 1 / (1 + np.exp(-lg)) if is_cod[i] else 0.012  # prepaid orders rarely fail
        if u[i] < p_fail:
            outcome[i] = "not_reachable" if u2[i] < 0.35 + 0.25 * (addr[i] > 1) else "refused"
            if is_cod[i]:
                count += 1
    o["prior_refusals"] = prior
    o["delivery_outcome"] = outcome
    o["failed"] = (o.delivery_outcome != "delivered").astype(int)
    o["delivery_cost_loss_aed"] = o.failed * FAILED_DELIVERY_COST_AED
    o["revenue_aed"] = np.where(o.failed == 1, 0.0, o.order_value_aed)
    o.insert(0, "order_id", np.arange(1, len(o) + 1))
    keep = [
        "order_id", "customer_id", "order_date", "country", "city", "category", "order_value_aed",
        "revenue_aed", "discount_used", "payment_method", "order_hour", "first_order", "prior_refusals",
        "address_quality", "delivery_outcome", "failed", "delivery_cost_loss_aed",
    ]
    o["first_order"] = o.first_order.astype(int)
    return o[keep].sort_values("order_date").reset_index(drop=True)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def make_campaign_retention(cust: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Problem 2: randomised campaign, Oct 2026, 3 arms with equal allocation."""
    cut = pd.Timestamp("2026-10-01")
    hist = orders[orders.order_date < cut]
    g = hist.groupby("customer_id").agg(
        last=("order_date", "max"), freq=("order_id", "count"), monetary=("revenue_aed", "sum"),
        disc_share=("discount_used", "mean"), cod_share=("payment_method", lambda s: (s == "COD").mean()),
    )
    g = g[g["last"] >= cut - pd.Timedelta(days=240)]
    g = g.join(cust.set_index("customer_id"))
    g = g.sample(n=min(24_000, len(g)), random_state=SEED)
    n = len(g)
    rec = (cut - g["last"]).dt.days.to_numpy()
    eng = g._engagement.to_numpy()
    ps = g._price_sensitivity.to_numpy()
    # baseline: will the customer order in the next 30 days without any offer?
    base = -1.1 - 0.012 * rec + 0.55 * np.log1p(g.freq.to_numpy()) + 0.35 * np.log(eng) - 0.6 * ps + 0.3 * g.member.to_numpy()
    arm = rng.choice(["No Offer", "Price Match", "Membership Trial"], n)
    # heterogeneous treatment effects on the logit scale
    eff_pm = 1.35 * ps - 0.35 - 0.45 * (1 - ps) * (g.member.to_numpy())  # deal hunters respond; some loyal members are "sleeping dogs"
    eff_mem = 0.95 * (1 - ps) * np.minimum(eng, 2.5) / 2.5 + 0.25 * (g.member.to_numpy() == 0) - 0.15
    lg = base + np.where(arm == "Price Match", eff_pm, 0) + np.where(arm == "Membership Trial", eff_mem, 0)
    conv = (rng.random(n) < sigmoid(lg)).astype(int)
    avg_val = (g.monetary / g.freq).to_numpy()
    val = np.where(conv == 1, np.round(rng.lognormal(np.log(np.clip(avg_val, 60, 2500)) - 0.1, 0.45), 2), 0.0)
    out = pd.DataFrame(
        {
            "customer_id": g.index.to_numpy(),
            "recency_days": rec,
            "frequency": g.freq.to_numpy(),
            "monetary_aed": g.monetary.round(2).to_numpy(),
            "discount_share": g.disc_share.round(3).to_numpy(),
            "cod_share": g.cod_share.round(3).to_numpy(),
            "member": g.member.to_numpy(),
            "country": g.country.to_numpy(),
            "acq_channel": g.acq_channel.to_numpy(),
            "offer": arm,
            "conversion": conv,
            "order_value_aed": val,
        }
    )
    return out.reset_index(drop=True)


def make_campaign_prepaid(cust: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Problem 1: randomised test. Treatment = AED 10 cashback if the next order is prepaid."""
    cut = pd.Timestamp("2026-09-01")
    hist = orders[orders.order_date < cut]
    g = hist.groupby("customer_id").agg(
        last=("order_date", "max"), freq=("order_id", "count"),
        cod_share=("payment_method", lambda s: (s == "COD").mean()),
        refusals=("failed", "sum"), avg_value=("order_value_aed", "mean"),
    )
    g = g[(g.cod_share >= 0.5) & (g["last"] >= cut - pd.Timedelta(days=300))]
    g = g.join(cust.set_index("customer_id"))
    g = g.sample(n=min(14_000, len(g)), random_state=SEED + 1)
    n = len(g)
    treat = rng.integers(0, 2, n)
    base = -1.2 - 2.0 * (g.cod_share.to_numpy() - 0.5) + 0.8 * g.card_saved.to_numpy() - 0.3 * (g.country.to_numpy() == "EGY")
    eff = 1.25 * g.card_saved.to_numpy() + 0.25 - 0.35 * (g.country.to_numpy() == "EGY")
    p = sigmoid(base + treat * eff)
    prepaid_next = (rng.random(n) < p).astype(int)
    out = pd.DataFrame(
        {
            "customer_id": g.index.to_numpy(),
            "country": g.country.to_numpy(),
            "address_quality": g.address_quality.to_numpy(),
            "card_saved": g.card_saved.to_numpy(),
            "frequency": g.freq.to_numpy(),
            "cod_share": g.cod_share.round(3).to_numpy(),
            "past_failed_deliveries": g.refusals.to_numpy(),
            "avg_order_value_aed": g.avg_value.round(2).to_numpy(),
            "treatment": treat,
            "prepaid_next_order": prepaid_next,
        }
    )
    return out.reset_index(drop=True)


def make_daily_demand() -> pd.DataFrame:
    """Problem 3: catalogue-level daily units by country x category, plus the CURRENT stock plan.

    Current plan = same day last year x 1.20 growth x 1.10 buffer (what the team does today).
    Before one year of history exists, the plan uses the 28-day trailing mean x 1.10.
    """
    base_units = {"UAE": 9000, "KSA": 11000, "EGY": 5000}
    cat_share = {"Electronics": 0.18, "Fashion": 0.26, "Beauty": 0.16, "Grocery": 0.26, "Home": 0.14}
    # category-specific event sensitivity
    sens = {
        "Electronics": dict(ramadan=0.05, pre_eid=0.9, eid=-0.3, yf=3.2, bts=0.5),
        "Fashion": dict(ramadan=0.15, pre_eid=1.4, eid=-0.3, yf=2.0, bts=0.4),
        "Beauty": dict(ramadan=0.15, pre_eid=0.9, eid=-0.2, yf=1.8, bts=0.1),
        "Grocery": dict(ramadan=0.75, pre_eid=0.35, eid=-0.25, yf=0.6, bts=0.15),
        "Home": dict(ramadan=0.30, pre_eid=0.6, eid=-0.2, yf=1.9, bts=0.3),
    }
    out = []
    for c, b in base_units.items():
        for cat, sh in cat_share.items():
            s = sens[cat]
            mu = (
                b * sh * CAL.trend
                * (1 + s["ramadan"] * CAL.ramadan)
                * (1 + s["pre_eid"] * CAL.pre_eid)
                * (1 + s["eid"] * CAL.eid)
                * (1 + s["yf"] * CAL.yellow_friday)
                * (1 + s["bts"] * CAL.back_to_school)
                * (1 + 0.10 * CAL.payday)
                * (1 + 0.07 * CAL.weekend)
                * (1 + 0.04 * np.sin(2 * np.pi * CAL.date.dt.dayofyear / 365.25))
            )
            units = rng.poisson(mu * rng.lognormal(0, 0.08, len(mu)))
            df = pd.DataFrame({"date": CAL.date, "country": c, "category": cat, "demand_units": units})
            ly = pd.Series(units).shift(364)
            trailing = pd.Series(units).rolling(28, min_periods=7).mean().shift(1)
            plan = np.where(ly.notna(), ly * 1.20 * 1.10, trailing * 1.10)
            df["planned_stock_units"] = np.round(pd.Series(plan).bfill().to_numpy())
            out.append(df)
    d = pd.concat(out, ignore_index=True)
    d["units_sold"] = np.minimum(d.demand_units, d.planned_stock_units).astype(int)
    d["lost_units"] = (d.demand_units - d.units_sold).astype(int)
    d["stockout"] = (d.lost_units > 0).astype(int)
    d["leftover_units"] = (d.planned_stock_units - d.units_sold).clip(lower=0).astype(int)
    d = d.merge(CAL[["date", "ramadan", "pre_eid", "eid", "yellow_friday", "back_to_school", "payday", "weekend"]], on="date")
    return d


def write_live_seed(orders: pd.DataFrame):
    """Small files the live dashboard needs, so the 10 MB order history does not have to be deployed."""
    g = orders.groupby("customer_id")
    pd.DataFrame({"n_prev": g.size(), "prior_ref": g.failed.sum()}).to_csv(DATA / "customer_history.csv")
    h = orders.sort_values(["customer_id", "order_date"]).copy()
    h["n_prev_orders"] = h.groupby("customer_id").cumcount()
    cod = h[h.payment_method == "COD"].sort_values("order_date").tail(120_000)
    cols = ["customer_id", "order_date", "first_order", "prior_refusals", "n_prev_orders", "order_value_aed", "order_hour",
            "discount_used", "address_quality", "country", "category", "failed"]
    cod[cols].to_csv(DATA / "cod_training_sample.csv.gz", index=False, compression="gzip")


def main():
    cust = make_customers()
    orders = make_orders(cust)
    camp_r = make_campaign_retention(cust, orders)
    camp_p = make_campaign_prepaid(cust, orders)
    demand = make_daily_demand()

    cust.to_csv(DATA / "customers.csv", index=False)
    orders.to_csv(DATA / "orders.csv.gz", index=False, compression="gzip")
    camp_r.to_csv(DATA / "campaign_retention.csv", index=False)
    camp_p.to_csv(DATA / "campaign_prepaid.csv", index=False)
    demand.to_csv(DATA / "daily_demand.csv", index=False)
    CAL.to_csv(DATA / "calendar.csv", index=False)
    write_live_seed(orders)

    print(f"customers        {len(cust):>9,}")
    print(f"orders           {len(orders):>9,}  ({orders.order_date.min().date()} to {orders.order_date.max().date()})")
    cod = orders[orders.payment_method == "COD"]
    print(f"COD share        {len(cod) / len(orders):>9.1%}   COD failed rate {cod.failed.mean():.1%}")
    print(f"campaign_ret     {len(camp_r):>9,}  conv by arm: {camp_r.groupby('offer').conversion.mean().round(3).to_dict()}")
    print(f"campaign_prepaid {len(camp_p):>9,}  prepaid by arm: {camp_p.groupby('treatment').prepaid_next_order.mean().round(3).to_dict()}")
    print(f"daily_demand     {len(demand):>9,}  stockout days {demand.stockout.mean():.1%}")


if __name__ == "__main__":
    main()
