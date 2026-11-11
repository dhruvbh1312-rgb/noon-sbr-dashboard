"""
Live data engine for the Noon SBR dashboard.

Simulates the event stream a marketplace backend would send, in REAL time (Dubai time, UTC+4):

  * orders        new orders every second, more in the evening, with messy records (missing city,
                  negative values, duplicates) that the cleaning step has to catch
  * deliveries    COD orders resolve 15 to 60 minutes later as delivered / refused / not reachable
  * reviews       noisy customer reviews: English, Arabic, Arabizi, emojis, typos, sarcasm,
                  one-word reviews, spam and copy-paste fakes
  * competitor    a rival price index per category (random walk with sudden drops)
  * stock         units on hand per country x category (about 4 hours of demand), refilled every hour from a forecast + 8% buffer
  * shocks        random real-world events (flash sale, courier problem in a city, rival price drop)
                  that the dashboard must DETECT from the data, never told about directly

On top of the stream it runs the analytics live:
  * the COD refusal-risk model (trained on the historical synthetic data) scores every new COD order
  * a rule-based multilingual sentiment + topic + spam detector reads every review
  * anomaly detection compares the last 15 to 30 minutes with the previous 2 hours and raises alerts
  * model monitoring: live ROC-AUC of the risk model and accuracy of the sentiment detector

Nothing here needs an API key or an external service, so it runs the same on Streamlit Community Cloud.
"""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DUBAI = timezone(timedelta(hours=4))

CITIES = {"Dubai": "UAE", "Abu Dhabi": "UAE", "Sharjah": "UAE", "Riyadh": "KSA", "Jeddah": "KSA", "Dammam": "KSA",
          "Cairo": "EGY", "Alexandria": "EGY"}
CATS = ["Electronics", "Fashion", "Beauty", "Grocery", "Home"]
CAT_P = np.array([0.20, 0.27, 0.16, 0.22, 0.15])
CAT_MEAN = {"Electronics": 620, "Fashion": 190, "Beauty": 125, "Grocery": 160, "Home": 260}
BASE_ORDERS_PER_SEC = 0.35          # about 1,250 orders an hour at the daily average
HOUR_CURVE = np.array([0.45, 0.30, 0.20, 0.15, 0.12, 0.15, 0.30, 0.55, 0.75, 0.85, 0.95, 1.00,
                       1.05, 1.05, 1.00, 1.00, 1.05, 1.15, 1.30, 1.50, 1.70, 1.75, 1.45, 0.90])
KEEP_HOURS = 24
BACKFILL_HOURS = 3
MAX_CATCHUP_HOURS = 6
STOCK_HOURS = 4                 # dark-store style: about 4 hours of demand on hand, refilled hourly
FEATURES = ["first_order", "prior_refusals", "n_prev_orders", "log_value", "late_hour", "discount_used", "member",
            "card_saved", "addr_medium", "addr_poor", "cty_KSA", "cty_UAE", "cat_Electronics", "cat_Fashion", "cat_Grocery",
            "cat_Home"]


def now_dubai() -> datetime:
    return datetime.now(DUBAI).replace(tzinfo=None)


# =================================================================================================
# Review generator (ground truth) and analyser (what the dashboard can see)
# =================================================================================================
EN = {
    ("delivery", "pos"): ["Delivered in {h} hours, super fast", "Driver was polite and on time", "Arrived a day before the promised date",
                          "fast delivery as always", "Courier called before coming, very professional"],
    ("delivery", "neg"): ["Order came {d} days late", "Courier never called me and marked me not reachable",
                          "Driver did not come up to my building", "Still waiting for my order", "delivery was very slow",
                          "They cancelled my delivery without telling me"],
    ("quality", "pos"): ["Exactly as described, great quality", "Original product, very happy", "Excellent quality for the price",
                         "Works perfectly, recommended"],
    ("quality", "neg"): ["Product arrived damaged", "Looks fake, not original", "Stopped working after 2 days", "Wrong size sent",
                         "Box was opened and item was used"],
    ("price", "pos"): ["Best price I found anywhere", "Great deal in the sale", "Cheaper than the mall"],
    ("price", "neg"): ["Same item is cheaper on Amazon", "Price went up right after I bought it", "Not worth the money",
                       "Found it {p} AED cheaper on another app"],
    ("service", "pos"): ["Customer service solved it in minutes", "Easy return, refund came fast"],
    ("service", "neg"): ["Support chat never replied", "Return request rejected for no reason", "Refund still not received"],
}
NEUTRAL = ["Ok", "Product is fine", "Average experience", "As expected", "Delivered", "fine", "It is ok for the price"]
ARABIZI = {
    "pos": ["wallah mumtaz, el delivery sari3", "7elw kteer, shukran noon", "el montag original w 7elw", "tamam kollo zay ma howa"],
    "neg": ["el mandoob ma etasal 5ales", "el montag 5arban", "ghaly kteer, fe arkhas", "ta25eer kteer fe el tawseel", "mesh original"],
}
ARABIC = {
    "pos": ["ممتاز والتوصيل سريع", "المنتج أصلي وجودة عالية", "شكرا نون خدمة رائعة"],
    "neg": ["التوصيل متأخر جدا", "المنتج وصل مكسور", "السعر أغلى من أمازون", "المندوب ما اتصل"],
    "neu": ["عادي", "المنتج مقبول"],
}
ADDONS = [", ordered {item}", " ({item})", ", bought it for my {who}", ". Order number {n} this month", ", {item} for the house",
          ", second time buying", ", fast checkout"]
ITEMS = ["a phone case", "headphones", "an abaya", "perfume", "rice and oil", "a kettle", "sneakers", "a lipstick", "a charger",
         "kids clothes", "a mixer", "dates"]
WHO = ["wife", "mother", "son", "brother", "office", "friend"]
SARCASM = ["Great, only {d} days late 👏", "Love waiting a week for a phone case 🙃", "Wow amazing, the box was empty",
           "Fantastic service, nobody answers the phone 🙃", "Perfect, wrong colour again 👏"]
SPAM = ["Use code SAVE50 at www.cheap-deals.xyz for 50% off!!!", "BEST SELLER!!! 5 STARS!!! BUY NOW!!!",
        "Visit my store for cheaper prices bit.ly/xx12", "good good good good good"]
EMO_POS, EMO_NEG = ["😍", "🔥", "👍", "❤️", "💯"], ["😡", "👎", "💔", "😤"]

POS_W = {"great", "excellent", "perfect", "fast", "super", "happy", "love", "best", "amazing", "recommended", "original",
         "polite", "professional", "cheaper", "deal", "fantastic", "solved", "easy", "wonderful", "good",
         "mumtaz", "7elw", "sari3", "shukran", "tamam", "ممتاز", "سريع", "أصلي", "رائعة", "شكرا", "عالية"}
NEG_W = {"late", "never", "damaged", "fake", "wrong", "slow", "waiting", "cancelled", "worst", "broken", "rejected", "used",
         "opened", "stopped", "expensive", "not", "empty", "nobody", "refund", "worth",
         "5arban", "ghaly", "ta25eer", "mesh", "متأخر", "مكسور", "أغلى", "ما"}
NEG_PHRASES = ["cheaper on", "cheaper on another", "never replied", "not reachable", "did not", "not worth", "went up", "ma etasal",
               "still not", "not original", "after 2 days"]
SARC_CUES = ["👏", "🙃", "wow amazing", "love waiting", "only"]
TOPIC_KW = {
    "delivery": ["deliver", "courier", "driver", "arrived", "late", "waiting", "mandoob", "tawseel", "التوصيل", "المندوب", "days"],
    "quality": ["quality", "works", "recommended", "damaged", "fake", "original", "working", "size", "colour", "box", "used", "montag", "المنتج", "مكسور"],
    "price": ["price", "cheaper", "aed", "deal", "worth", "ghaly", "arkhas", "السعر", "أغلى", "amazon"],
    "service": ["service", "support", "refund", "return", "phone", "answers", "خدمة"],
}
URL_RE = re.compile(r"(www\.|https?://|bit\.ly|code\s+[A-Z0-9]{4,})", re.I)


def add_typos(text: str, rng: np.random.Generator) -> str:
    chars = list(text)
    for _ in range(rng.integers(1, 3)):
        if len(chars) > 4:
            i = rng.integers(1, len(chars) - 1)
            if rng.random() < 0.5:
                chars[i], chars[i - 1] = chars[i - 1], chars[i]
            else:
                del chars[i]
    return "".join(chars)


def make_review(rng, label, topic, ctx) -> tuple[str, str, bool]:
    """Return (text, language, is_spam). label in {pos, neu, neg} is the hidden truth."""
    r = rng.random()
    if label == "neu":
        if r < 0.15:
            return rng.choice(ARABIC["neu"]), "Arabic", False
        return str(rng.choice(NEUTRAL)), "English", False
    if label == "neg" and r < 0.10:
        return str(rng.choice(SARCASM)).format(d=rng.integers(3, 9)), "English (sarcasm)", False
    if r < 0.22:
        return str(rng.choice(ARABIZI[label])), "Arabizi", False
    if r < 0.32:
        return str(rng.choice(ARABIC[label])), "Arabic", False
    t = str(rng.choice(EN[(topic, label)])).format(h=rng.integers(2, 20), d=rng.integers(2, 8), p=rng.integers(15, 120))
    if label == "neg" and topic == "price" and ctx.get("rival_drop"):
        t = f"Same item is {rng.integers(20, 150)} AED cheaper on Amazon"
    lang = "English"
    if rng.random() < 0.55:
        t = f"{t}{rng.choice(ADDONS)}".format(item=rng.choice(ITEMS), who=rng.choice(WHO), n=rng.integers(2, 6))
    if rng.random() < 0.25:
        t = add_typos(t, rng)
    if rng.random() < 0.2:
        t += " " + str(rng.choice(EMO_POS if label == "pos" else EMO_NEG))
    if rng.random() < 0.08:
        t = t.upper() + "!!!"
    if rng.random() < 0.12:
        t = t + ". " + str(rng.choice(ARABIZI[label]))
        lang = "Mixed"
    return t, lang, False


def analyse_review(text: str, seen: set, stars: int = 3, verified: bool = True) -> dict:
    """Rule-based multilingual sentiment, topic and spam detection (what Noon would run on raw text)."""
    t = str(text).strip()
    low = t.lower()
    words = re.findall(r"[\w؀-ۿ']+", low)
    spam = bool(URL_RE.search(t)) or (sum(c.isupper() for c in t) > 0.6 * max(1, sum(c.isalpha() for c in t)) and "5 STARS" in t) \
        or (len(set(words)) <= 2 and len(words) >= 4) or (not verified and (low in seen or stars == 5))
    score = sum(w in POS_W for w in words) - sum(w in NEG_W for w in words)
    score -= 2 * sum(p in low for p in NEG_PHRASES)
    score += sum(e in t for e in EMO_POS) - sum(e in t for e in EMO_NEG)
    if any(c in low for c in SARC_CUES) and any(k in low for k in ["late", "waiting", "empty", "nobody", "wrong", "days"]):
        score = -abs(score) - 2  # sarcasm: positive words describing a bad fact
    if "not " in low and score > 0 and any(w in low for w in ["good", "great", "original", "worth"]):
        score = -score
    label = "Positive" if score >= 1 else "Negative" if score <= -1 else "Neutral"
    topics = [k for k, kws in TOPIC_KW.items() if any(kw.lower() in low for kw in kws)]
    return {"sentiment": label, "sent_score": score, "topic": topics[0] if topics else "general", "is_spam_flag": spam}


# =================================================================================================
# Live store
# =================================================================================================
@dataclass
class Shock:
    kind: str
    target: str
    start: datetime
    end: datetime
    manual: bool = False

    def active(self, t: datetime) -> bool:
        return self.start <= t <= self.end

    def label(self) -> str:
        names = {"flash_sale": f"Flash sale on {self.target}", "courier_issue": f"Courier problem in {self.target}",
                 "rival_drop": f"Rival price drop on {self.target}"}
        return names[self.kind]


@dataclass
class LiveStore:
    seed: int = 7
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed + int(now_dubai().timestamp()) % 100000)
        cust = pd.read_csv(DATA / "customers.csv")
        # compact seed files (made from orders.csv.gz) keep the deployed app small
        hist_stats = pd.read_csv(DATA / "customer_history.csv", index_col="customer_id")
        self.cust = cust.set_index("customer_id").join(hist_stats)
        self.cust = self.cust.fillna({"n_prev": 0, "prior_ref": 0})
        w = 0.4 + self.cust._engagement.clip(upper=4)
        self.cust_ids = self.cust.index.to_numpy()
        self.cust_p = (w / w.sum()).to_numpy()
        self.model = self._train_model(pd.read_csv(DATA / "cod_training_sample.csv.gz"))
        self.orders = pd.DataFrame()
        self.reviews = pd.DataFrame()
        self.prices = pd.DataFrame()
        self.restock_log = deque(maxlen=200)
        self.quality = {"raw_rows": 0, "missing_city": 0, "negative_value": 0, "duplicates": 0}
        self.seen_reviews: set = set()
        self.shocks: list[Shock] = []
        self.price_idx = {c: 100.0 for c in CATS}
        self.order_seq = 10_000_000
        start = now_dubai() - timedelta(hours=BACKFILL_HOURS)
        self.stock = {(co, ca): self._expected_rate(co, ca) * 3600 * STOCK_HOURS for co in ["UAE", "KSA", "EGY"] for ca in CATS}
        self.last_restock = start
        self.last_ts = start
        self.advance(now_dubai())

    # ------------------------------------------------------------------ model
    def _features(self, df: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        X["first_order"] = df.first_order.astype(int)
        X["prior_refusals"] = df.prior_refusals.clip(upper=10)
        X["n_prev_orders"] = df.n_prev_orders.clip(upper=200)
        X["log_value"] = np.log(df.order_value_aed.clip(lower=1))
        X["late_hour"] = ((df.order_hour >= 22) | (df.order_hour <= 2)).astype(int)
        X["discount_used"] = df.discount_used.astype(int)
        X["member"] = df.member.astype(int)
        X["card_saved"] = df.card_saved.astype(int)
        X["addr_medium"] = (df.address_quality == "medium").astype(int)
        X["addr_poor"] = (df.address_quality == "poor").astype(int)
        X["cty_KSA"] = (df.country == "KSA").astype(int)
        X["cty_UAE"] = (df.country == "UAE").astype(int)
        for c in ["Electronics", "Fashion", "Grocery", "Home"]:
            X[f"cat_{c}"] = (df.category == c).astype(int)
        return X[FEATURES]

    def _train_model(self, h: pd.DataFrame):
        """Refusal-risk model on the latest 120k historical COD orders (same features as the live orders)."""
        h = h.join(self.cust[["member", "card_saved"]], on="customer_id")
        m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=40,
                                           l2_regularization=1.0, random_state=42)
        m.fit(self._features(h), h.failed)
        return m

    # ------------------------------------------------------------------ helpers
    def _expected_rate(self, country=None, category=None, hour=None):
        share = {"UAE": 0.45, "KSA": 0.40, "EGY": 0.15}
        r = BASE_ORDERS_PER_SEC * (HOUR_CURVE[hour] if hour is not None else 1.0)
        if country:
            r *= share[country]
        if category:
            r *= CAT_P[CATS.index(category)]
        return r

    def _active(self, t, kind):
        return [s for s in self.shocks if s.kind == kind and s.active(t)]

    def inject(self, kind: str, target: str | None = None, minutes: int = 30):
        with self.lock:
            t = now_dubai()
            if target is None:
                target = self.rng.choice(list(CITIES) if kind == "courier_issue" else CATS)
            self.shocks.append(Shock(kind, str(target), t, t + timedelta(minutes=minutes), manual=True))

    def _maybe_random_shock(self, t0, t1):
        minutes = (t1 - t0).total_seconds() / 60
        if self.rng.random() < 1 - np.exp(-minutes / 35):  # about one shock every 35 minutes
            kind = self.rng.choice(["flash_sale", "courier_issue", "rival_drop"], p=[0.35, 0.35, 0.30])
            target = self.rng.choice(list(CITIES)) if kind == "courier_issue" else self.rng.choice(CATS)
            st = t0 + timedelta(seconds=float(self.rng.uniform(0, max(minutes * 60, 1))))
            dur = {"flash_sale": 20, "courier_issue": 40, "rival_drop": 50}[kind]
            self.shocks.append(Shock(kind, str(target), st, st + timedelta(minutes=dur)))

    # ------------------------------------------------------------------ main loop
    def advance(self, t1: datetime | None = None):
        """Generate every event between the last update and t1 (default: now, Dubai time)."""
        with self.lock:
            t1 = t1 or now_dubai()
            t0 = max(self.last_ts, t1 - timedelta(hours=MAX_CATCHUP_HOURS))
            while t0 < t1:  # one-minute steps keep shocks, stock and prices consistent
                step_end = min(t0 + timedelta(minutes=1), t1)
                self._maybe_random_shock(t0, step_end)
                self._step(t0, step_end)
                t0 = step_end
            self.last_ts = t1
            cut = t1 - timedelta(hours=KEEP_HOURS)
            if len(self.orders):
                self.orders = self.orders[self.orders.ts >= cut]
            if len(self.reviews):
                self.reviews = self.reviews[self.reviews.ts >= cut]
            if len(self.prices):
                self.prices = self.prices[self.prices.ts >= cut]
            self.shocks = [s for s in self.shocks if s.end >= cut]

    def _step(self, t0: datetime, t1: datetime):
        rng = self.rng
        secs = (t1 - t0).total_seconds()
        hour = t0.hour
        # competitor price index random walk, with sharp drops during a rival price drop
        for c in CATS:
            drop = self._active(t0, "rival_drop")
            target = 100 - (12 if any(s.target == c for s in drop) else 0)
            self.price_idx[c] += 0.15 * (target - self.price_idx[c]) + rng.normal(0, 0.25)
        self.prices = pd.concat([self.prices, pd.DataFrame([{"ts": t0, **{c: round(v, 2) for c, v in self.price_idx.items()}}])],
                                ignore_index=True)
        # ---- orders
        lam = self._expected_rate(hour=hour) * secs
        flash = self._active(t0, "flash_sale")
        n = rng.poisson(lam * (1 + 0.8 * len(flash)))
        if n:
            self._make_orders(n, t0, t1, flash)
        # ---- hourly replenishment: forecast (last hour's rate) x 4h x 1.08 buffer (Problem 3 recommendation)
        if (t1 - self.last_restock) >= timedelta(hours=1):
            recent = self.orders[self.orders.ts >= t1 - timedelta(hours=1)] if len(self.orders) else pd.DataFrame()
            for key in self.stock:
                rate = len(recent[(recent.country == key[0]) & (recent.category == key[1])]) / 3600 if len(recent) else self._expected_rate(*key)
                target = max(rate, self._expected_rate(*key) * 0.5) * 3600 * STOCK_HOURS * 1.08
                if self.stock[key] < target:
                    self.restock_log.append({"ts": t1, "country": key[0], "category": key[1], "units": round(target - self.stock[key])})
                    self.stock[key] = target
            self.last_restock = t1
        # ---- deliveries resolve, then some customers write reviews
        self._resolve_and_review(t1)

    def _make_orders(self, n, t0, t1, flash):
        rng = self.rng
        secs = (t1 - t0).total_seconds()
        ts = [t0 + timedelta(seconds=float(s)) for s in np.sort(rng.uniform(0, secs, n))]
        cids = rng.choice(self.cust_ids, n, p=self.cust_p)
        c = self.cust.loc[cids]
        p = np.tile(CAT_P, (n, 1))
        for s in flash:
            p[:, CATS.index(s.target)] *= 4
        for s in self._active(t0, "rival_drop"):
            p[:, CATS.index(s.target)] *= 0.75  # customers buy that category elsewhere
        p /= p.sum(1, keepdims=True)
        cat = np.array(CATS)[(rng.random((n, 1)) > p.cumsum(1)).sum(1)]
        val = np.round(rng.lognormal(np.log([CAT_MEAN[x] for x in cat]) - 0.18, 0.6), 2)
        ps = c._price_sensitivity.to_numpy()
        disc = rng.random(n) < np.clip(0.10 + 0.55 * ps + 0.3 * np.isin(cat, [s.target for s in flash]), 0, 0.97)
        cod = rng.random(n) < (c._cod_propensity.to_numpy() - 0.06)
        df = pd.DataFrame({
            "order_id": np.arange(self.order_seq, self.order_seq + n), "ts": ts, "customer_id": cids,
            "city": c.city.to_numpy(), "country": c.country.to_numpy(), "category": cat, "order_value_aed": val,
            "discount_used": disc.astype(int), "payment_method": np.where(cod, "COD", "Prepaid"),
            "order_hour": [x.hour for x in ts], "first_order": (c.n_prev.to_numpy() == 0).astype(int),
            "n_prev_orders": c.n_prev.to_numpy(), "prior_refusals": c.prior_ref.to_numpy(), "member": c.member.to_numpy(),
            "card_saved": c.card_saved.to_numpy(), "address_quality": c.address_quality.to_numpy(),
        })
        self.order_seq += n
        # stock: an order for an empty shelf is lost demand
        lost = np.zeros(n, dtype=bool)
        for i, (co, ca) in enumerate(zip(df.country, df.category)):
            units = 1 + (rng.random() < 0.3)
            if self.stock[(co, ca)] >= units:
                self.stock[(co, ca)] -= units
            else:
                lost[i] = True
        df["lost_stockout"] = lost.astype(int)
        # live refusal-risk score from the trained model
        df["risk"] = np.where(df.payment_method == "COD", self.model.predict_proba(self._features(df))[:, 1], np.nan)
        # true outcome (hidden until it resolves) uses the generator's behaviour + any courier shock
        addr = df.address_quality.map({"good": 0.0, "medium": 0.45, "poor": 1.15}).to_numpy()
        ctry = df.country.map({"UAE": 0.0, "KSA": 0.25, "EGY": 0.45}).to_numpy()
        late = ((df.order_hour >= 22) | (df.order_hour <= 2)).astype(int).to_numpy()
        lg = (-3.45 + 1.05 * df.first_order + addr + 0.55 * np.log(df.order_value_aed / 200) + 0.45 * late
              + 1.25 * c._refusal_tendency.to_numpy() + ctry - 0.4 * df.member + 0.55 * np.minimum(df.prior_refusals, 3)).to_numpy()
        hit = np.isin(df.city, [s.target for s in self._active(t0, "courier_issue")])
        p_fail = np.where(cod, 1 / (1 + np.exp(-lg)), 0.012)
        fail = rng.random(n) < p_fail
        df["true_outcome"] = np.where(fail, np.where(rng.random(n) < 0.4, "not_reachable", "refused"), "delivered")
        df["delay_hours"] = np.round(rng.gamma(2.0, 6.0, n) * (1 + 1.5 * hit), 1)
        df["resolve_ts"] = [x + timedelta(minutes=float(m)) for x, m in zip(ts, rng.uniform(15, 60, n))]
        df["status"] = "pending"
        df.loc[lost.astype(bool), "status"] = "lost_stockout"
        df["courier_hit"] = hit.astype(int)
        # update customer history so repeat refusers become visible to the model
        self.cust.loc[cids, "n_prev"] += 1
        # ---- messy raw data: what the cleaning step must catch
        raw = df.copy()
        q = self.quality
        m1 = rng.random(n) < 0.01
        raw.loc[m1, "city"] = None
        m2 = rng.random(n) < 0.005
        raw.loc[m2, "order_value_aed"] = -raw.loc[m2, "order_value_aed"]
        dup = raw[rng.random(n) < 0.004]
        raw = pd.concat([raw, dup], ignore_index=True)
        q["raw_rows"] += len(raw)
        q["missing_city"] += int(raw.city.isna().sum())
        q["negative_value"] += int((raw.order_value_aed < 0).sum())
        q["duplicates"] += int(raw.duplicated("order_id").sum())
        clean = raw.drop_duplicates("order_id")
        clean = clean.assign(city=clean.city.fillna(clean.customer_id.map(self.cust.city)),
                             order_value_aed=clean.order_value_aed.abs())
        self.orders = pd.concat([self.orders, clean], ignore_index=True)

    def _resolve_and_review(self, t):
        if not len(self.orders):
            return
        o = self.orders
        due = (o.status == "pending") & (o.resolve_ts <= t)
        if not due.any():
            return
        # a courier problem hits every COD delivery that is out on the road in that city right now
        hit_cities = [s.target for s in self._active(t, "courier_issue")]
        if hit_cities:
            extra = due & o.city.isin(hit_cities) & (o.payment_method == "COD") & (o.true_outcome == "delivered")
            flip = extra & (self.rng.random(len(o)) < 0.30)
            o.loc[flip, "true_outcome"] = "not_reachable"
            o.loc[due & o.city.isin(hit_cities), "delay_hours"] *= 2.5
            o.loc[due & o.city.isin(hit_cities), "courier_hit"] = 1
        o.loc[due, "status"] = o.loc[due, "true_outcome"]
        failed_ids = o.loc[due & (o.true_outcome != "delivered"), "customer_id"]
        self.cust.loc[failed_ids, "prior_ref"] += 1
        rng = self.rng
        done = o[due]
        writes = rng.random(len(done)) < np.where(done.true_outcome == "delivered", 0.12, 0.35)
        rows = []
        rival_cats = {s.target for s in self._active(t, "rival_drop")}
        for r in done[writes].itertuples(index=False):
            ctx = {"rival_drop": r.category in rival_cats}
            if r.true_outcome != "delivered":
                label, topic = "neg", "delivery"
            else:
                bad_delivery = r.delay_hours > 30
                bad_quality = rng.random() < 0.10
                price_issue = rng.random() < (0.35 if ctx["rival_drop"] else 0.06)
                if bad_delivery:
                    label, topic = "neg", "delivery"
                elif bad_quality:
                    label, topic = "neg", "quality"
                elif price_issue:
                    label, topic = "neg", "price"
                else:
                    label = rng.choice(["pos", "neu"], p=[0.72, 0.28])
                    topic = str(rng.choice(["delivery", "quality", "price", "service"], p=[0.4, 0.35, 0.15, 0.10]))
                    if label == "neg" or (label == "pos" and rng.random() < 0.05):
                        topic = "service"
            text, lang, _ = make_review(rng, label, topic, ctx)
            stars = {"pos": rng.choice([4, 5], p=[0.3, 0.7]), "neu": 3, "neg": rng.choice([1, 2], p=[0.7, 0.3])}[label]
            if rng.random() < 0.06:  # rating and text disagree (people mis-tap stars)
                stars = 6 - stars
            rows.append({"ts": r.resolve_ts + timedelta(minutes=float(rng.uniform(1, 30))), "order_id": r.order_id,
                         "customer_id": r.customer_id, "city": r.city, "category": r.category, "stars": int(stars),
                         "text": text, "language": lang, "true_label": label, "true_topic": topic, "is_spam": False,
                         "verified": True})
        # spam / fake reviews arrive independently of real orders
        for _ in range(rng.poisson(0.02 * len(done) + 0.05)):
            if len(self.reviews) and rng.random() < 0.3:
                text = str(self.reviews.text.iloc[int(rng.integers(0, len(self.reviews)))])  # copy-paste fake
            else:
                text = str(rng.choice(SPAM))
            verified = bool(rng.random() < 0.15)  # some fakers buy a cheap item first to look genuine
            rows.append({"ts": t, "order_id": None, "customer_id": None, "city": str(rng.choice(list(CITIES))),
                         "category": str(rng.choice(CATS)), "stars": 5, "text": text, "language": "Spam", "true_label": "spam",
                         "true_topic": "spam", "is_spam": True, "verified": verified})
        if rows:
            rv = pd.DataFrame(rows)
            an = [analyse_review(x, self.seen_reviews, st, v) for x, st, v in zip(rv.text, rv.stars, rv.verified)]
            for x in rv.text:
                self.seen_reviews.add(str(x).lower())
            rv = pd.concat([rv, pd.DataFrame(an)], axis=1)
            self.reviews = pd.concat([self.reviews, rv], ignore_index=True)

    # =============================================================================================
    # Read side: KPIs, alerts, monitoring (called by the dashboard)
    # =============================================================================================
    def snapshot(self, window_min: int = 60) -> dict:
        with self.lock:
            t = self.last_ts
            o = self.orders.copy()
            rv = self.reviews[self.reviews.ts <= t].copy() if len(self.reviews) else pd.DataFrame()
            pr = self.prices.copy()
            stock = dict(self.stock)
            shocks = list(self.shocks)
            quality = dict(self.quality)
            restocks = list(self.restock_log)
        w0 = t - timedelta(minutes=window_min)
        ow = o[o.ts >= w0]
        res = o[o.status.isin(["delivered", "refused", "not_reachable"]) & (o.payment_method == "COD")]
        res_w = res[res.resolve_ts >= w0]
        rvw = rv[rv.ts >= w0] if len(rv) else rv
        real = rvw[~rvw.is_spam_flag] if len(rvw) else rvw
        k = {
            "now": t, "orders": len(ow), "revenue": float(ow[ow.status != "lost_stockout"].order_value_aed.sum()),
            "orders_per_min": len(ow) / window_min, "cod_share": float((ow.payment_method == "COD").mean()) if len(ow) else 0,
            "cod_failed_rate": float((res_w.status != "delivered").mean()) if len(res_w) else np.nan,
            "high_risk_share": float((ow.risk >= 0.40).mean()) if len(ow) else 0,
            "reviews": len(rvw), "avg_stars": float(real.stars.mean()) if len(real) else np.nan,
            "net_sentiment": float(((real.sentiment == "Positive").mean() - (real.sentiment == "Negative").mean()) * 100) if len(real) else np.nan,
            "spam_caught": int(rvw.is_spam_flag.sum()) if len(rvw) else 0,
            "lost_stockout": int(ow.lost_stockout.sum()), "pending": int((o.status == "pending").sum()),
        }
        return {"t": t, "kpi": k, "orders": o, "orders_w": ow, "resolved": res, "reviews": rv, "reviews_w": rvw, "prices": pr,
                "stock": stock, "shocks": shocks, "quality": quality, "alerts": self._alerts(t, o, res, rv, stock),
                "monitor": self._monitor(res, rv), "restocks": restocks}

    def _alerts(self, t, o, res, rv, stock):
        alerts = []
        recent, base = o[o.ts >= t - timedelta(minutes=15)], o[(o.ts < t - timedelta(minutes=15)) & (o.ts >= t - timedelta(hours=2))]
        if len(base) > 100:
            for c in CATS:
                r_now = (recent.category == c).sum() / 15
                r_base = (base.category == c).sum() / max((t - timedelta(minutes=15) - max(base.ts.min(), t - timedelta(hours=2))).total_seconds() / 60, 1)
                if r_base > 0 and r_now > 2.0 * r_base and (recent.category == c).sum() >= 15:
                    alerts.append(("Demand spike", f"{c}: {r_now:.1f} orders/min vs {r_base:.1f} normal. Check stock and delivery capacity.", "Problem 3"))
        r30 = res[res.resolve_ts >= t - timedelta(minutes=30)]
        b = res[(res.resolve_ts < t - timedelta(minutes=30)) & (res.resolve_ts >= t - timedelta(hours=3))]
        if len(b) > 80:
            p0 = (b.status != "delivered").mean()
            for city, g in r30.groupby("city"):
                if len(g) >= 15:
                    p = (g.status != "delivered").mean()
                    z = (p - p0) / np.sqrt(p0 * (1 - p0) / len(g))
                    if z > 2.5:
                        alerts.append(("Refusals rising", f"{city}: {p:.0%} of COD deliveries failed in the last 30 min vs {p0:.0%} normal. Check courier partner.", "Problem 1"))
        if len(rv):
            real = rv[~rv.is_spam_flag]
            rn = real[real.ts >= t - timedelta(minutes=30)]
            rb = real[(real.ts < t - timedelta(minutes=30)) & (real.ts >= t - timedelta(hours=3))]
            if len(rb) > 30:
                for c, g in rn.groupby("category"):
                    pr_now = ((g.topic == "price") & (g.sentiment == "Negative")).mean()
                    bb = rb[rb.category == c]
                    pr_base = ((bb.topic == "price") & (bb.sentiment == "Negative")).mean() if len(bb) else 0
                    if len(g) >= 8 and pr_now > max(0.15, 2.5 * pr_base):
                        alerts.append(("Price complaints", f"{c}: {pr_now:.0%} of recent reviews say it is cheaper elsewhere. Possible rival price drop: churn risk.", "Problem 2"))
        for (co, ca), units in stock.items():
            need = self._expected_rate(co, ca, t.hour) * 3600 * 2
            if units < need:
                alerts.append(("Stock-out risk", f"{ca} in {co}: {units:.0f} units left, under 2 hours of demand.", "Problem 3"))
        return alerts

    def _monitor(self, res, rv):
        out = {"risk_auc": np.nan, "resolved_cod": len(res), "sentiment_acc": np.nan, "spam_recall": np.nan, "spam_precision": np.nan}
        if len(res) > 200 and res.status.nunique() > 1:
            out["risk_auc"] = float(roc_auc_score(res.status != "delivered", res.risk))
        if len(rv) > 50:
            real = rv[rv.true_label != "spam"]
            m = {"pos": "Positive", "neu": "Neutral", "neg": "Negative"}
            out["sentiment_acc"] = float((real.true_label.map(m) == real.sentiment).mean())
            if rv.is_spam.any():
                out["spam_recall"] = float(rv[rv.is_spam].is_spam_flag.mean())
            if rv.is_spam_flag.any():
                out["spam_precision"] = float(rv[rv.is_spam_flag].is_spam.mean())
        return out


if __name__ == "__main__":  # terminal demo: python src/live_stream.py
    import time

    store = LiveStore()
    print("Live stream started. Press Ctrl+C to stop.\n")
    while True:
        store.advance()
        s = store.snapshot(15)
        k = s["kpi"]
        print(f"{s['t']:%H:%M:%S}  orders(15m)={k['orders']:>4}  COD failed={k['cod_failed_rate'] if k['cod_failed_rate'] == k['cod_failed_rate'] else 0:.1%}  "
              f"reviews={k['reviews']:>3}  net sentiment={k['net_sentiment'] if k['net_sentiment'] == k['net_sentiment'] else 0:+.0f}")
        for title, msg, prob in s["alerts"][:3]:
            print(f"   ALERT [{prob}] {title}: {msg}")
        time.sleep(5)
