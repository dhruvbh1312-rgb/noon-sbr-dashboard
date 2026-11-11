"""Visual system for the Noon SBR dashboard: colours, CSS, HTML blocks and Altair chart helpers."""
from __future__ import annotations

import html

import altair as alt
import pandas as pd

# ---------------------------------------------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------------------------------------------
INK = "#1E1B4B"          # deep indigo: header, sidebar, primary text
INK_2 = "#5B5875"        # secondary text
LINE = "#E6E3F2"         # hairlines and card borders
PAGE = "#F5F4FA"         # page background
CARD = "#FFFFFF"
SAFFRON = "#F5A524"      # live pulse, stars
MUTED_BAR = "#B9B5D6"    # bars that are not being highlighted

GOOD, WARNING, SERIOUS, CRITICAL = "#0ca30c", "#fab219", "#ec835a", "#d03b3b"
NEUTRAL = "#B8B5C9"

CATS = ["Electronics", "Fashion", "Beauty", "Grocery", "Home"]
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # validated, fixed order
SENT = ["Positive", "Neutral", "Negative"]
SENT_COLORS = [GOOD, NEUTRAL, CRITICAL]
OFFER_COLORS = {"No Offer": NEUTRAL, "Price Match": "#eb6834", "Membership Trial": "#6250d6"}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Readex+Pro:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {{
  font-family: 'Readex Pro', system-ui, sans-serif !important;
}}
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
h1 {{ font-weight: 700; }}
h2, h3 {{ font-weight: 600; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {INK}; }}
[data-testid="stSidebar"] * {{ color: #E9E7FB !important; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: 6px 10px; border-radius: 10px; margin-bottom: 2px;
}}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,0.07); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: rgba(245,165,36,0.18); box-shadow: inset 3px 0 0 {SAFFRON};
}}
.side-brand {{ font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 2px; }}
.side-sub {{ font-size: .8rem; opacity: .75; margin-bottom: 14px; }}

/* native metrics become cards */
[data-testid="stMetric"] {{
  background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 14px 16px;
}}
[data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: .85rem; }}
[data-testid="stMetricValue"] {{ color: {INK}; font-weight: 600; font-variant-numeric: tabular-nums; }}

/* tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ border-radius: 10px 10px 0 0; padding: 8px 14px; color: {INK_2}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; font-weight: 600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background: {SAFFRON}; height: 3px; }}

/* hero (live page) */
.hero {{
  background: {INK}; color: #fff; border-radius: 22px; padding: 22px 26px; margin-bottom: 14px;
  display: flex; justify-content: space-between; align-items: flex-end; gap: 18px; flex-wrap: wrap;
}}
.hero h1 {{ color: #fff; margin: 4px 0 2px; font-size: 2.1rem; line-height: 1.1; }}
.hero p {{ color: #CFCBF2; margin: 0; max-width: 60ch; font-size: .95rem; }}
.live {{ display: inline-flex; align-items: center; gap: 8px; font-weight: 600; color: {SAFFRON}; font-size: .9rem; }}
.live i {{ width: 10px; height: 10px; border-radius: 50%; background: {SAFFRON}; display: inline-block;
  box-shadow: 0 0 0 0 rgba(245,165,36,.7); animation: pulse 1.6s infinite; }}
@keyframes pulse {{ 0% {{ box-shadow: 0 0 0 0 rgba(245,165,36,.7); }} 70% {{ box-shadow: 0 0 0 12px rgba(245,165,36,0); }}
  100% {{ box-shadow: 0 0 0 0 rgba(245,165,36,0); }} }}
@media (prefers-reduced-motion: reduce) {{ .live i {{ animation: none; }} }}
.clock {{ text-align: right; }}
.clock b {{ display: block; font-size: 2.6rem; font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1; }}
.clock span {{ color: #CFCBF2; font-size: .85rem; }}

/* review strip */
.voices {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 10px; margin-bottom: 16px; }}
.voice {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; padding: 12px 14px; border-top: 4px solid {NEUTRAL}; }}
.voice.Positive {{ border-top-color: {GOOD}; }} .voice.Negative {{ border-top-color: {CRITICAL}; }}
.voice.spam {{ border-top-color: {INK_2}; opacity: .8; }}
.voice q {{ display: block; color: {INK}; font-size: .98rem; line-height: 1.45; quotes: none; margin: 4px 0 8px; }}
.voice small {{ color: {INK_2}; font-size: .78rem; }}
.stars {{ color: {SAFFRON}; letter-spacing: 1px; }}

/* kpi groups */
.group {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; padding: 14px 16px 12px; height: 100%; }}
.group h4 {{ margin: 0 0 10px; font-size: .95rem; color: {INK}; font-weight: 600; }}
.group h4 span {{ color: {INK_2}; font-weight: 400; }}
.kpis {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }}
.kpi b {{ display: block; font-size: 1.55rem; font-weight: 600; color: {INK}; font-variant-numeric: tabular-nums; line-height: 1.15; }}
.kpi b.bad {{ color: {CRITICAL}; }} .kpi b.good {{ color: {GOOD}; }}
.kpi span {{ color: {INK_2}; font-size: .78rem; }}

/* alerts */
.alert {{ background: {CARD}; border: 1px solid {LINE}; border-left: 6px solid {WARNING}; border-radius: 14px;
  padding: 10px 14px; margin-bottom: 8px; }}
.alert.critical {{ border-left-color: {CRITICAL}; }} .alert.serious {{ border-left-color: {SERIOUS}; }}
.alert.calm {{ border-left-color: {GOOD}; }}
.alert b {{ color: {INK}; }} .alert small {{ color: {INK_2}; display: block; margin-top: 2px; }}
.tag {{ display: inline-block; font-size: .72rem; padding: 2px 8px; border-radius: 999px; background: #EEEAF9; color: {INK};
  margin-left: 6px; vertical-align: 2px; }}

/* review feed */
.feed {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 16px; padding: 4px 14px; }}
.row {{ display: grid; grid-template-columns: 56px 1fr auto; gap: 10px; padding: 10px 0; border-bottom: 1px solid {LINE}; align-items: start; }}
.row:last-child {{ border-bottom: none; }}
.row time {{ color: {INK_2}; font-size: .8rem; font-variant-numeric: tabular-nums; padding-top: 2px; }}
.row .txt {{ color: {INK}; line-height: 1.45; }}
.row .meta {{ color: {INK_2}; font-size: .78rem; }}
.pill {{ display: inline-block; font-size: .75rem; padding: 2px 9px; border-radius: 999px; color: #fff; white-space: nowrap; }}
.pill.Positive {{ background: {GOOD}; }} .pill.Neutral {{ background: #8F8AA8; }} .pill.Negative {{ background: {CRITICAL}; }}
.pill.spam {{ background: {INK}; }}

/* overview result cards */
.result {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 20px; padding: 18px 20px; height: 100%; }}
.result h3 {{ font-size: 1.02rem; margin: 0 0 12px; }}
.result .ba {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.result .before {{ color: {INK_2}; font-size: 1.3rem; text-decoration: line-through; text-decoration-color: {CRITICAL}; }}
.result .after {{ color: {INK}; font-size: 2.3rem; font-weight: 700; font-variant-numeric: tabular-nums; }}
.result p {{ color: {INK_2}; margin: 8px 0 0; font-size: .9rem; }}
.banner {{ background: {INK}; color: #fff; border-radius: 22px; padding: 24px 28px; margin-bottom: 16px; }}
.banner h1 {{ color: #fff; margin: 0 0 6px; font-size: 2.2rem; }}
.banner p {{ color: #CFCBF2; margin: 0; max-width: 70ch; }}
</style>
"""


def esc(x) -> str:
    return html.escape(str(x))


def stars(n: int) -> str:
    n = int(n)
    return f'<span class="stars" aria-label="{n} out of 5 stars">{"★" * n}</span><span style="color:#D8D5EA">{"★" * (5 - n)}</span>'


# ---------------------------------------------------------------------------------------------------------------
# Altair helpers
# ---------------------------------------------------------------------------------------------------------------
def finish(ch, height=260):
    return (ch.properties(height=height)
            .configure(font="Readex Pro", background=CARD)
            .configure_view(stroke=None)
            .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor="#EFEDF7", domainColor=LINE, tickColor=LINE,
                            labelFontSize=11, titleFontSize=11, titleFontWeight=400, labelLimit=260)
            .configure_legend(labelColor=INK, titleColor=INK_2, orient="top", labelFontSize=11, titleFontSize=11,
                              symbolType="circle")
            .configure_title(color=INK, fontSize=13, anchor="start", fontWeight=600))


def cat_scale():
    return alt.Scale(domain=CATS, range=CAT_COLORS)


def lines_by_category(df: pd.DataFrame, x: str, y: str, y_title: str, x_type: str = "T", height=260, rule_y=None, y_domain=None):
    yscale = alt.Scale(domain=y_domain, zero=False, clamp=True) if y_domain else alt.Undefined
    base = alt.Chart(df).mark_line(strokeWidth=2, interpolate="monotone").encode(
        x=alt.X(f"{x}:{x_type}", title=None),
        y=alt.Y(f"{y}:Q", title=y_title, scale=yscale),
        color=alt.Color("category:N", scale=cat_scale(), title=None),
        tooltip=[alt.Tooltip(f"{x}:{x_type}", title="time" if x_type == "T" else x), "category:N", alt.Tooltip(f"{y}:Q", format=",.1f")],
    )
    if rule_y is not None:
        base = alt.layer(alt.Chart(pd.DataFrame({"y": [rule_y]})).mark_rule(color=INK_2, strokeDash=[4, 4]).encode(y="y:Q"), base)
    return finish(base, height)


def single_line(df, x, y, y_title, color="#2a78d6", x_type="T", height=240, area=True, fmt=",.0f"):
    enc = dict(x=alt.X(f"{x}:{x_type}", title=None), y=alt.Y(f"{y}:Q", title=y_title),
               tooltip=[alt.Tooltip(f"{x}:{x_type}"), alt.Tooltip(f"{y}:Q", format=fmt)])
    line = alt.Chart(df).mark_line(strokeWidth=2.2, color=color, interpolate="monotone").encode(**enc)
    if area:
        fill = alt.Chart(df).mark_area(color=color, opacity=0.12, interpolate="monotone").encode(x=enc["x"], y=enc["y"])
        return finish(alt.layer(fill, line), height)
    return finish(line, height)


def highlight_bars(df, cat, val, val_title, threshold=None, horizontal=True, height=260, fmt=",.1f",
                   hi_color=CRITICAL, lo_color=MUTED_BAR, rule=None, rule_label=None, sort="-x"):
    color = (alt.condition(f"datum['{val}'] > {threshold}", alt.value(hi_color), alt.value(lo_color))
             if threshold is not None else alt.value("#2a78d6"))
    if horizontal:
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
            y=alt.Y(f"{cat}:N", title=None, sort=sort), x=alt.X(f"{val}:Q", title=val_title), color=color,
            tooltip=[f"{cat}:N", alt.Tooltip(f"{val}:Q", format=fmt)])
    else:
        bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, width={"band": 0.7}).encode(
            x=alt.X(f"{cat}:N", title=None, sort=None), y=alt.Y(f"{val}:Q", title=val_title), color=color,
            tooltip=[f"{cat}:N", alt.Tooltip(f"{val}:Q", format=fmt)])
    layers = [bars]
    if rule is not None:
        enc = {"x": "v:Q"} if horizontal else {"y": "v:Q"}
        layers.append(alt.Chart(pd.DataFrame({"v": [rule]})).mark_rule(color=INK, strokeDash=[4, 4], strokeWidth=1.5).encode(**enc))
        if rule_label:
            t = alt.Chart(pd.DataFrame({"v": [rule], "t": [rule_label]})).mark_text(
                align="left", dx=4, dy=-6 if horizontal else -6, color=INK, fontSize=11)
            layers.append(t.encode(**enc, text="t:N") if not horizontal else t.encode(x="v:Q", y=alt.value(0), text="t:N"))
    return finish(alt.layer(*layers), height)


def diverging_bars(df, cat, val, val_title, height=260, fmt="+,.2f"):
    lo, hi = float(min(df[val].min(), 0)), float(max(df[val].max(), 0))
    pad = (hi - lo) * 0.18 or 1
    xs = alt.Scale(domain=[lo - pad, hi + pad], nice=False)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height={"band": 0.7}).encode(
        y=alt.Y(f"{cat}:N", title=None, sort="-x"), x=alt.X(f"{val}:Q", title=val_title, scale=xs),
        color=alt.condition(f"datum['{val}'] >= 0", alt.value(GOOD), alt.value(CRITICAL)),
        tooltip=[f"{cat}:N", alt.Tooltip(f"{val}:Q", format=fmt)])
    pos = alt.Chart(df).transform_filter(f"datum['{val}'] >= 0").mark_text(align="left", dx=6, color=INK, fontSize=11).encode(
        y=alt.Y(f"{cat}:N", sort="-x"), x=alt.X(f"{val}:Q", scale=xs), text=alt.Text(f"{val}:Q", format=fmt))
    neg = alt.Chart(df).transform_filter(f"datum['{val}'] < 0").mark_text(align="right", dx=-6, color=INK, fontSize=11).encode(
        y=alt.Y(f"{cat}:N", sort="-x"), x=alt.X(f"{val}:Q", scale=xs), text=alt.Text(f"{val}:Q", format=fmt))
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(color=INK_2).encode(x="z:Q")
    return finish(alt.layer(bars, pos, neg, zero), height)


def stacked_sentiment(df, cat, height=260):
    """df columns: cat, sentiment, n"""
    ch = alt.Chart(df).mark_bar(stroke=CARD, strokeWidth=2, height={"band": 0.7}).encode(
        y=alt.Y(f"{cat}:N", title=None, sort="-x", axis=alt.Axis(labelOverlap=False)), x=alt.X("n:Q", title="reviews", stack=True),
        color=alt.Color("sentiment:N", scale=alt.Scale(domain=SENT, range=SENT_COLORS), title=None),
        order=alt.Order("order:Q"), tooltip=[f"{cat}:N", "sentiment:N", "n:Q"])
    return finish(ch, height)


def bars_by_category(df, x, y, y_title, facet=None, height=200, fmt=",.0f"):
    base = alt.Chart(df).mark_bar(cornerRadiusEnd=4, width={"band": 0.75}).encode(
        x=alt.X(f"{x}:N", title=None, sort=CATS, axis=alt.Axis(labelAngle=-35, labelOverlap=False)), y=alt.Y(f"{y}:Q", title=y_title),
        color=alt.Color("category:N", scale=cat_scale(), legend=None),
        tooltip=[f"{x}:N", alt.Tooltip(f"{y}:Q", format=fmt)] + ([f"{facet}:N"] if facet else []))
    if facet:
        ch = base.properties(width=170, height=height).facet(column=alt.Column(f"{facet}:N", title=None,
                                                                                 header=alt.Header(labelColor=INK, labelFontSize=13, labelFontWeight=600, labelOrient="top")))
        return (ch.configure(font="Readex Pro", background=CARD).configure_view(stroke=None)
                .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor="#EFEDF7", domainColor=LINE))
    return finish(base, height)


def colored_bars(df, cat, val, val_title, colors: dict, height=220, fmt=",.0f"):
    ch = alt.Chart(df).mark_bar(cornerRadiusEnd=4, width={"band": 0.6}).encode(
        x=alt.X(f"{cat}:N", title=None, sort=list(colors), axis=alt.Axis(labelAngle=0)), y=alt.Y(f"{val}:Q", title=val_title),
        color=alt.Color(f"{cat}:N", scale=alt.Scale(domain=list(colors), range=list(colors.values())), legend=None),
        tooltip=[f"{cat}:N", alt.Tooltip(f"{val}:Q", format=fmt)])
    return finish(ch, height)


def forecast_chart(df, height=300):
    """df: week, series, units"""
    dom = ["Actual demand", "Current plan", "Model forecast"]
    rng = [INK, CRITICAL, "#2a78d6"]
    ch = alt.Chart(df).mark_line(strokeWidth=2.2, interpolate="monotone").encode(
        x=alt.X("week:T", title=None), y=alt.Y("units:Q", title="units per week"),
        color=alt.Color("series:N", scale=alt.Scale(domain=dom, range=rng), title=None),
        strokeDash=alt.StrokeDash("series:N", scale=alt.Scale(domain=dom, range=[[1, 0], [6, 4], [1, 0]]), legend=None),
        tooltip=[alt.Tooltip("week:T"), "series:N", alt.Tooltip("units:Q", format=",.0f")])
    return finish(ch, height)
