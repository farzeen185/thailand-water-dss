"""Thailand Water Storage Decision Support System (Streamlit).

Earthquake-corrected GRACE storage, climate-driven storage (Models A and B) and the
residual (human water use + model error) for Thailand and its 22 river basins.
Author: Farzeen Khan, Asian Institute of Technology (2026).
"""
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit import config as st_config

DATA_ZIP = Path(__file__).parent / "dss_data.zip"

def read_json(name):
    with zipfile.ZipFile(DATA_ZIP) as z:
        return json.loads(z.read(name).decode("utf-8"))

st.set_page_config(page_title="Thailand Water Storage DSS", page_icon="💧", layout="wide")

# Always light, so the page looks the same on a phone set to dark mode.
try:
    if st_config.get_option("theme.base") != "light":
        for opt, val in (("theme.base", "light"), ("theme.primaryColor", "#0f7c8a"), ("theme.backgroundColor", "#f3f7fa"),
                         ("theme.secondaryBackgroundColor", "#e6eff4"), ("theme.textColor", "#16324a")):
            st_config.set_option(opt, val)
        st.rerun()
except st.errors.Error:
    pass
except (KeyError, AttributeError, RuntimeError):
    pass

FONT = '"Times New Roman", Tinos, Times, serif'
NAVY, TEAL, INK, MUTED = "#0b2e4a", "#0f7c8a", "#16324a", "#55697a"

def html(s):
    st.markdown(" ".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)

# ------------------------------------------------------------------ style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Tinos:wght@400;700&display=swap');
.stApp {background: #f3f7fa;}
.stApp, .stApp p, .stApp li, .stApp label, .stApp input, .stApp button, .stApp h1, .stApp h2, .stApp h3, .stApp td, .stApp th,
.stApp div[data-baseweb="select"] div, [data-testid="stMarkdownContainer"] * {font-family: "Times New Roman", Tinos, Times, serif !important;}
.stApp p, .stApp li, .stApp label {font-size: 1.04rem;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.1rem; padding-bottom: 2.4rem; max-width: 1380px;}
h3 {color: #0b2e4a; font-weight: 700; letter-spacing: -.1px;}

.hero {position: relative; overflow: hidden; border-radius: 20px 20px 0 0; padding: 38px 42px 72px; color: #fff; margin-bottom: 18px;
       background: linear-gradient(120deg, #0b2e4a 0%, #0d5570 58%, #0f7c8a 100%);}
.hero .eye {font-size: .92rem; letter-spacing: 2.2px; text-transform: uppercase; color: #9fe0ea;}
.hero h1 {color: #fff; font-size: 2.75rem; line-height: 1.12; margin: 8px 0 10px; padding: 0; font-weight: 700;}
.hero .sub {font-size: 1.24rem; line-height: 1.42; color: #dcedf3; max-width: 800px;}
.hero .chips {margin-top: 18px; display: flex; flex-wrap: wrap; gap: 8px; position: relative; z-index: 2;}
.hero .chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.26); border-radius: 999px; padding: 4px 14px; font-size: .96rem;}
.hero svg {position: absolute; left: 0; right: 0; bottom: -1px; width: 100%; height: 62px; display: block;}

.stTabs [data-baseweb="tab-list"] {gap: 6px; flex-wrap: wrap; border: 0; margin-bottom: 8px;}
.stTabs [data-baseweb="tab-border"], .stTabs [data-baseweb="tab-highlight"] {display: none;}
.stTabs button[role="tab"] {background: #fff; border: 1px solid #d6e2ea; border-radius: 999px; padding: 7px 20px; height: auto;}
.stTabs button[role="tab"] p {font-size: 1.08rem; color: #16324a;}
.stTabs button[role="tab"][aria-selected="true"] {background: #0b2e4a; border-color: #0b2e4a;}
.stTabs button[role="tab"][aria-selected="true"] p {color: #fff; font-weight: 700;}

[class*="st-key-card"] {background: #fff; border: 1px solid #e0e9ef !important; border-radius: 16px; box-shadow: 0 2px 10px rgba(11,46,74,.06); padding: 16px 18px;}

.kpis {display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 4px 0 16px;}
.kpi {background: #fff; border: 1px solid #e0e9ef; border-radius: 14px; padding: 16px 18px 14px; box-shadow: 0 2px 10px rgba(11,46,74,.06); border-top: 4px solid var(--a, #0f7c8a);}
.kpi .v {font-size: 2rem; font-weight: 700; color: #0b2e4a; line-height: 1.1;}
.kpi .l {font-size: 1rem; color: #55697a; margin-top: 6px; line-height: 1.25;}

.callout {background: #fff; border-left: 5px solid #0f7c8a; border-radius: 0 14px 14px 0; padding: 15px 20px; margin: 4px 0 16px; font-size: 1.12rem; line-height: 1.45; color: #16324a; box-shadow: 0 2px 10px rgba(11,46,74,.05);}
.callout.warn {border-left-color: #c77700;}
.sec {font-size: 1.5rem; font-weight: 700; color: #0b2e4a; margin: 14px 0 2px;}
.secsub {font-size: 1.05rem; color: #55697a; margin-bottom: 10px;}
.note {font-size: 1rem; color: #55697a;}

.tag {display: inline-block; padding: 3px 13px; border-radius: 999px; color: #fff; font-weight: 700; font-size: .95rem;}
.tags {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 4px;}

.top {display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 14px; margin-bottom: 18px;}
.tc {background: #fff; border: 1px solid #e0e9ef; border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(11,46,74,.06);}
.tc .n {font-size: .95rem; color: #c62828; font-weight: 700;}
.tc .b {font-size: 1.2rem; font-weight: 700; color: #0b2e4a; line-height: 1.2; min-height: 2.4em; margin-top: 2px;}
.tc .bar {height: 7px; border-radius: 4px; background: #edf2f6; margin: 8px 0 7px; overflow: hidden;}
.tc .bar i {display: block; height: 100%; background: linear-gradient(90deg, #f0a35e, #c62828);}
.tc .m {font-size: .98rem; color: #55697a; line-height: 1.3;}

.acts {display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; margin: 12px 0 18px;}
.act {background: #fff; border: 1px solid #e0e9ef; border-top: 4px solid var(--a); border-radius: 14px; padding: 14px 16px; box-shadow: 0 2px 10px rgba(11,46,74,.06);}
.act .h {font-size: 1.16rem; font-weight: 700; color: #0b2e4a;}
.act .t {font-size: 1.02rem; color: #16324a; line-height: 1.35; margin: 5px 0 8px;}
.act .bs {font-size: .96rem; color: #55697a; line-height: 1.3;}

.steps {display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 14px; margin: 8px 0 18px;}
.step {background: #fff; border: 1px solid #e0e9ef; border-radius: 14px; padding: 15px 18px; display: flex; gap: 14px; box-shadow: 0 2px 10px rgba(11,46,74,.06);}
.step .c {flex: 0 0 36px; height: 36px; border-radius: 50%; background: #0f7c8a; color: #fff; font-weight: 700; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;}
.step .h {font-weight: 700; color: #0b2e4a; font-size: 1.14rem;}
.step .t {color: #16324a; font-size: 1.02rem; line-height: 1.36; margin-top: 3px;}

.foot {margin-top: 22px; padding-top: 14px; border-top: 1px solid #d6e2ea; color: #55697a; font-size: .98rem; line-height: 1.4;}

@media (max-width: 640px) {
  .block-container {padding-top: 3.2rem;}
  .hero {padding: 24px 20px 58px; border-radius: 16px 16px 0 0;}
  .hero h1 {font-size: 1.9rem;}
  .hero .sub {font-size: 1.08rem;}
  .kpis {grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px;}
  .kpi {padding: 12px 12px 10px;}
  .kpi .v {font-size: 1.45rem;}
  .kpi .l {font-size: .92rem;}
  .steps {grid-template-columns: 1fr;}
  [class*="st-key-card"] {padding: 10px 8px;}
}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ data
@st.cache_data
def load():
    return read_json("results.json"), read_json("basins.geojson"), read_json("points.json")

@st.cache_data
def series(rid):
    return read_json(f"ts/{rid}.json")

R, GEO, PTS = load()
META, REG = R["meta"], R["regions"]
BAS = [k for k in META["order"] if k.isdigit()]
for f in GEO["features"]:
    f["id"] = str(f["properties"].get("region", f["properties"].get("id", ""))).zfill(2)
NAME = {k: v["name"] for k, v in REG.items()}
TARG = ["CSR_seismic_adj", "GSFC_seismic_adj", "GravIS", "GravIS_GFZ", "G3P"]
TLAB = {"CSR_seismic_adj": "CSR (corrected)", "GSFC_seismic_adj": "GSFC (corrected)", "GravIS": "COST-G GravIS",
        "GravIS_GFZ": "GFZ GravIS", "G3P": "G3P", "GRACE_ensemble": "Ensemble of 5"}
PCOL = {"CSR": "#0072B2", "GSFC": "#E69F00", "CSR_seismic_adj": "#0072B2", "GSFC_seismic_adj": "#E69F00", "GravIS": "#009E73",
        "GravIS_GFZ": "#CC79A7", "G3P": "#8C564B", "GRACE_ensemble": "#000000"}
CLASS_COL = {"High": "#c62828", "Medium": "#e08e0b", "Low": "#2e7d32", "Uncertain": "#8a97a3"}
ACTIONS = {"High": "Introduce monthly groundwater-level monitoring, meter and license large abstractions, review dry-season irrigation demand, assess managed aquifer recharge.",
           "Medium": "Verify the residual with wells and reservoir records, apply dry-season demand management, include in the next groundwater survey.",
           "Low": "Storage change is small or mostly climate-driven: maintain the existing observation network.",
           "Uncertain": "Affected by the 2004 earthquake signal or model limits: collect ground observations before deciding."}

GAP = ("2017-07-01", "2018-05-31")
QUAKE = "2004-12-26"

def cont(values, idx):
    """Continuous monthly line: gaps of up to 3 months filled; only the 2017-18 satellite gap stays open."""
    y = pd.Series(values, index=idx, dtype=float).interpolate(limit=3, limit_area="inside")
    y[(y.index >= GAP[0]) & (y.index <= GAP[1])] = np.nan
    return y

def fmt(v, pattern="{:+.1f}", suffix=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return pattern.format(v).replace("-", "−") + suffix

def kpis(items):
    """items: (value, label, accent colour)"""
    cells = "".join(f'<div class="kpi" style="--a:{a}"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l, a in items)
    html(f'<div class="kpis">{cells}</div>')

def basemap(z, locs, colorscale, zmin, zmax, title, hover, height=560, line="white"):
    fig = go.Figure(go.Choropleth(geojson=GEO, locations=locs, z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                                  marker=dict(line=dict(width=.9, color=line)), text=hover, hoverinfo="text",
                                  colorbar=dict(title=dict(text=title, side="right"), thickness=13, len=.72, outlinewidth=0)))
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT, size=15, color=INK), hoverlabel=dict(font=dict(family=FONT, size=15)))
    return fig

def style(fig, height, ytitle):
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=34, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis_title=ytitle, font=dict(family=FONT, size=15, color=INK), hoverlabel=dict(font=dict(family=FONT, size=15)))
    fig.update_xaxes(showgrid=False, showline=True, linecolor="#b9c8d3", ticks="outside", tickcolor="#b9c8d3", tickfont=dict(color=INK))
    fig.update_yaxes(showgrid=True, gridcolor="#e8eef3", zeroline=True, zerolinecolor="#b9c8d3", showline=False, tickfont=dict(color=INK), title_font=dict(color=INK))
    return fig

def nice_ts(fig, height=420, ytitle="TWSA (mm)"):
    style(fig, height, ytitle)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=-0.14, x=0, font=dict(size=14)))
    fig.add_shape(type="line", x0=QUAKE, x1=QUAKE, yref="paper", y0=0, y1=1, line=dict(color="#c0392b", dash="dot", width=1.5))
    fig.add_shape(type="rect", x0=GAP[0], x1=GAP[1], yref="paper", y0=0, y1=1, fillcolor="#e4eaef", line_width=0, layer="below")
    fig.add_annotation(x=QUAKE, yref="paper", y=1, yanchor="bottom", text="2004 earthquake", showarrow=False, font=dict(color="#c0392b", size=13))
    fig.add_annotation(x="2017-12-01", yref="paper", y=1, yanchor="bottom", text="Satellite gap", showarrow=False, font=dict(color=MUTED, size=13))
    return fig

# ------------------------------------------------------------------ priority index
W0 = {"loss": 30, "conf": 20, "gw": 20, "irr": 10, "decline": 10, "agree": 10}
WLAB = {"loss": "Residual loss", "conf": "Confidence (95% range below zero)", "gw": "Groundwater dependence",
        "irr": "Irrigation intensity", "decline": "Total storage decline", "agree": "Product agreement"}
clamp = lambda v: float(min(1, max(0, v)))

def priority(p, W, flag):
    rows = []
    for b in BAS:
        r = REG[b]; rb = r["residual_by_product"].get(p, {}) or {}; a = r.get("attributes", {}) or {}
        lo, hi, t = rb.get("lo"), rb.get("hi"), rb.get("trend")
        x = {"loss": clamp(-(t or 0) / 10), "conf": clamp((0 - lo) / (hi - lo)) if (lo is not None and hi and hi > lo) else 0,
             "gw": clamp((a.get("gw_share_pct") or 0) / 17), "irr": clamp((a.get("irrigated_pct") or 0) / 40),
             "decline": clamp(-((r["trend"].get("GRACE_ensemble") or {}).get("sen") or 0) / 10),
             "agree": sum((r["residual_by_product"].get(k) or {}).get("trend", 0) < 0 for k in TARG) / 5}
        ws = sum(W.values()) or 1; score = 100 * sum(W[k] * x[k] for k in W) / ws
        unc = (flag and r.get("group") == "M_S") or (r.get("mcmc") and not r["mcmc"].get("converged", True))
        cls = "Uncertain" if unc else "High" if score >= 60 else "Medium" if score >= 35 else "Low"
        rows.append(dict(id=b, Basin=NAME[b], Class=cls, Score=round(score), Residual=t, GW=a.get("gw_share_pct"), Irr=a.get("irrigated_pct")))
    df = pd.DataFrame(rows); df["o"] = (df.Class == "Uncertain").astype(int)
    return df.sort_values(["o", "Score"], ascending=[True, False]).drop(columns="o")

DEFAULT = priority("GRACE_ensemble", W0, True)
DEF_ID = DEFAULT.set_index("id")

def class_tags(df):
    n = df.Class.value_counts()
    return '<div class="tags">' + "".join(f'<span class="tag" style="background:{CLASS_COL[c]}">{c}: {int(n.get(c, 0))}</span>' for c in CLASS_COL) + "</div>"

# ------------------------------------------------------------------ header
html("""
<div class="hero">
  <div class="eye">Decision Support System</div>
  <h1>Thailand Water Storage Decision Support System</h1>
  <div class="sub">See where Thailand is losing water, how much of the change rain and temperature explain, and which river basins need action first.</div>
  <div class="chips">
    <span class="chip">22 river basins</span><span class="chip">5 GRACE products</span><span class="chip">2002–2024</span><span class="chip">Earthquake-corrected</span>
  </div>
  <svg viewBox="0 0 1440 90" preserveAspectRatio="none" aria-hidden="true">
    <path d="M0,46 C240,92 460,6 720,44 C980,82 1200,18 1440,52 L1440,90 L0,90 Z" fill="#ffffff" opacity=".12"/>
    <path d="M0,66 C260,30 500,92 760,62 C1010,34 1220,84 1440,60 L1440,90 L0,90 Z" fill="#f3f7fa"/>
  </svg>
</div>""")
tabs = st.tabs(["Overview", "Basin Explorer", "Priority Index", "Earthquake Check", "Data & Methods"])

# ================================================================== OVERVIEW
with tabs[0]:
    H = META["headline"]
    kpis([(fmt(H["ens_sen"], "{:.2f}", " mm/yr"), "National storage trend (corrected, Sen slope)", "#0f7c8a"),
          ("≈ 31 km³", "Water lost since 2002", "#c62828"),
          (f"{H['n_sig']} of 22", "Basins with a significant decline", "#e08e0b"),
          ("−2.6 to −4.7", "Residual trend, mm/yr (human water use + model error)", "#332288"),
          ("19 of 20", "Model tests satisfactory on unseen years", "#2e7d32")])
    html('<div class="callout"><b>Key message.</b> Thailand is losing water (≈ 2.7 mm/yr). Rain and temperature explain the floods and droughts but not the decline; '
         'the unexplained part is largest in the irrigated central plains, where well levels confirm a groundwater signal.</div>')
    left, right = st.columns([1.15, 1], gap="medium")
    with left.container(border=True, key="card_ov_ts"):
        st.subheader("Thailand: storage, climate and residual")
        mode = st.radio("Show", ["Five corrected products", "TWSA = climate + residual"], horizontal=True, key="ov_mode")
        s = series("TH"); d = pd.to_datetime(s["dates"])
        fig = go.Figure()
        if mode.startswith("Five"):
            for k in TARG:
                y = cont(s["model"][k]["grace"], d).rolling(12, center=True, min_periods=6).mean()
                fig.add_trace(go.Scatter(x=d, y=y, name=TLAB[k], line=dict(color=PCOL[k], width=2.2)))
        else:
            mdl = st.selectbox("Product", TARG, format_func=TLAB.get, key="ov_p")
            for comp, col, nm in (("grace", "#222222", "Corrected GRACE TWSA"), ("climate", "#332288", "TWSA climate"), ("residual", "#C0392B", "TWSA residual")):
                y = cont(s["model"][mdl][comp], d)
                y = (y - y.groupby(y.index.month).transform("mean")).rolling(12, center=True, min_periods=6).mean()
                fig.add_trace(go.Scatter(x=d, y=y, name=nm, line=dict(color=col, width=2.6)))
        st.plotly_chart(nice_ts(fig, 470), width="stretch", key="ov_ts")
    with right.container(border=True, key="card_ov_map"):
        st.subheader("Residual trend by basin")
        p = st.selectbox("GRACE product", TARG, index=2, format_func=TLAB.get, key="ov_map")
        z = [REG[b]["residual_by_product"].get(p, {}).get("trend") for b in BAS]
        hov = [f"<b>{NAME[b]}</b><br>Residual {v:+.1f} mm/yr" if v is not None else NAME[b] for b, v in zip(BAS, z)]
        st.plotly_chart(basemap(z, BAS, "RdBu", -8, 8, "mm/yr", hov, height=470), width="stretch", key="ov_chart")
        html('<div class="note">Red = storage loss not explained by rain and temperature.</div>')

    html('<div class="sec">Basins that need action first</div><div class="secsub">High-priority basins from the default Priority Index (ensemble of 5 products).</div>')
    high = DEFAULT[DEFAULT.Class == "High"]
    cards = "".join(
        f'<div class="tc"><div class="n">Priority {i}</div><div class="b">{row.Basin}</div>'
        f'<div class="bar"><i style="width:{row.Score}%"></i></div>'
        f'<div class="m">Score {row.Score} of 100<br>Residual {fmt(row.Residual)} mm/yr<br>Groundwater share {fmt(row.GW, "{:.0f}", "%")}</div></div>'
        for i, row in enumerate(high.itertuples(), 1))
    html(f'<div class="top">{cards}</div>')
    html(class_tags(DEFAULT))

# ================================================================== EXPLORER
with tabs[1]:
    with st.container(border=True, key="card_ex_ctl"):
        c1, c2, c3 = st.columns([1.2, 1, 1.2])
        rid = c1.selectbox("Basin", ["TH"] + BAS, format_func=lambda k: f"{k} · {NAME[k]}" if k != "TH" else "Thailand (national)")
        prod = c2.selectbox("Product", TARG, index=2, format_func=TLAB.get, key="ex_p")
        view = c3.radio("Series", ["Five products", "Decomposition", "Model fit"], horizontal=True)
    r = REG[rid]; a = r.get("attributes", {}) or {}
    tr = r["trend"].get(prod, {}) or {}; rb = r["residual_by_product"].get(prod, {}) or {}
    title = "Thailand (national)" if rid == "TH" else NAME[rid]
    tag = ""
    if rid != "TH":
        cl = DEF_ID.loc[rid]
        tag = f' &nbsp;<span class="tag" style="background:{CLASS_COL[cl.Class]}">{cl.Class} · score {cl.Score}</span>'
    html(f'<div class="sec">{title}{tag}</div>')
    kpis([(fmt(tr.get("sen")), "Storage trend, mm/yr (Sen)", "#0f7c8a"),
          (fmt(rb.get("trend")), "Residual trend, mm/yr", "#C0392B"),
          (fmt(rb.get("climate")), "Climate trend, mm/yr", "#332288"),
          (fmt(a.get("irrigated_pct"), "{:.0f}", "%"), "Area irrigated", "#2e7d32"),
          (fmt(a.get("gw_share_pct"), "{:.0f}", "%"), "Irrigation from groundwater", "#e08e0b")])
    s = series(rid); d = pd.to_datetime(s["dates"]); fig = go.Figure()
    if view == "Five products":
        for kk in TARG:
            fig.add_trace(go.Scatter(x=d, y=cont(s["model"][kk]["grace"], d), name=TLAB[kk], line=dict(color=PCOL[kk], width=1.7)))
    elif view == "Decomposition":
        for comp, col, nm in (("grace", "#222222", "Corrected GRACE TWSA"), ("climate", "#332288", "TWSA climate"), ("residual", "#C0392B", "TWSA residual")):
            fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod][comp], d), name=nm, line=dict(color=col, width=2)))
    else:
        fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod]["detrended"], d), name="Detrended GRACE TWSA", line=dict(color=PCOL[prod], width=2.6)))
        fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod]["climate"], d), name="TWSA climate (model)", line=dict(color="#332288", width=1.6, dash="dash")))
    left, right = st.columns([3, 1], gap="medium")
    with left.container(border=True, key="card_ex_ts"):
        st.plotly_chart(nice_ts(fig, 450), width="stretch", key="ex_ts")
    with right.container(border=True, key="card_ex_map"):
        hl = [1 if (rid == "TH" or b == rid) else 0 for b in BAS]
        loc = basemap(hl, BAS, [[0, "#dbe7ef"], [1, TEAL]], 0, 1, "", [NAME[b] for b in BAS], height=450)
        loc.update_traces(showscale=False)
        st.plotly_chart(loc, width="stretch", key="ex_map")
    if rid != "TH":
        sk = (r.get("skill") or {}).get(prod, {}) or {}
        html(f'<div class="callout"><b>How well the model fits ({TLAB[prod]}).</b> Calibration NSE {fmt(sk.get("cal"), "{:.2f}")} · '
             f'prediction-test NSE {fmt(sk.get("pred"), "{:.2f}")} · r {fmt(sk.get("r"), "{:.2f}")}</div>')
        if r.get("group") == "M_S":
            html('<div class="callout warn"><b>Southern peninsula basin.</b> Nearest the 2004 earthquake, so results are uncertain.</div>')

# ================================================================== PRIORITY
with tabs[2]:
    html('<div class="sec">Which basins need action first?</div><div class="secsub">Change the weights and the map, ranking and actions update at once.</div>')
    left, right = st.columns([1, 2.2], gap="medium")
    with left.container(border=True, key="card_pr_ctl"):
        p = st.selectbox("GRACE product", ["GRACE_ensemble"] + TARG, format_func=TLAB.get, key="pr_p")
        flag = st.checkbox("Mark southern basins as Uncertain", True)
        st.markdown("**Weights**")
        W = {k: st.slider(WLAB[k], 0, 50, v, 5, key=f"w_{k}") for k, v in W0.items()}
    df = priority(p, W, flag)
    by_id = df.set_index("id")
    with right.container(border=True, key="card_pr_map"):
        cmap = {"High": 3, "Medium": 2, "Low": 1, "Uncertain": 0}
        z = [cmap[c] for c in by_id.loc[BAS].Class]
        hov = [f"<b>{NAME[b]}</b><br>{c} · score {s}" for b, c, s in zip(BAS, by_id.loc[BAS].Class, by_id.loc[BAS].Score)]
        cs = [[0, CLASS_COL["Uncertain"]], [.25, CLASS_COL["Uncertain"]], [.25, CLASS_COL["Low"]], [.5, CLASS_COL["Low"]],
              [.5, CLASS_COL["Medium"]], [.75, CLASS_COL["Medium"]], [.75, CLASS_COL["High"]], [1, CLASS_COL["High"]]]
        fig = basemap(z, BAS, cs, 0, 3, "Class", hov, height=560); fig.update_traces(showscale=False)
        st.plotly_chart(fig, width="stretch", key="pr_map")
        html(class_tags(df))
    html('<div class="sec">What to do</div>')
    acts = ""
    for c in CLASS_COL:
        names = ", ".join(df[df.Class == c].Basin) or "None at these settings"
        acts += f'<div class="act" style="--a:{CLASS_COL[c]}"><div class="h">{c} priority</div><div class="t">{ACTIONS[c]}</div><div class="bs"><b>Basins:</b> {names}</div></div>'
    html(f'<div class="acts">{acts}</div>')
    html('<div class="sec">Full ranking</div>')
    show = df.rename(columns={"Residual": "Residual (mm/yr)", "GW": "Groundwater share (%)", "Irr": "Irrigated (%)"})
    show["Recommended action"] = show.Class.map(ACTIONS)
    st.dataframe(show.drop(columns="id"), width="stretch", hide_index=True, height=450,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
    st.download_button("Download ranking (CSV)", show.to_csv(index=False).encode("utf-8"), "priority_ranking.csv", "text/csv")

# ================================================================== EARTHQUAKE
with tabs[3]:
    html('<div class="sec">The 2004 Sumatra–Andaman earthquake signal in GRACE</div>')
    html('<div class="callout">CSR and GSFC drop ≈ 100–150 mm after December 2004 while COST-G, GFZ and G3P do not: a gravity signal from the earthquake, '
         'largest in the southern peninsula. CSR and GSFC were corrected by fitting a sudden step plus a slow after-change to GRACE − WaterGAP in every 0.25° cell.</div>')
    left, right = st.columns([1, 1], gap="medium")
    sc = pd.DataFrame(META["screening"]).set_index("product").loc[["CSR", "GSFC", "GravIS", "GravIS_GFZ", "G3P"]].reset_index()
    sc["short"] = ["CSR", "GSFC", "COST-G GravIS", "GFZ GravIS", "G3P"]
    with left.container(border=True, key="card_eq_bar"):
        st.subheader("Size of the December 2004 jump")
        fig = go.Figure(go.Bar(x=sc.short, y=sc.jump, marker_color=[PCOL[p] for p in sc["product"]], marker_cornerradius=6,
                               text=[f"{v:+.0f} mm<br>{'PASS' if ok else 'FAIL'}" for v, ok in zip(sc.jump, sc.passed)], textposition="outside", cliponaxis=False))
        wg = sc.watergap.iloc[0]
        fig.add_hrect(y0=wg - 83, y1=wg + 66, fillcolor="#dceee3", line_width=0, layer="below", annotation_text="normal range", annotation_position="bottom right")
        fig.add_hline(y=wg, line=dict(color="#D55E00", dash="dot"), annotation_text=f"WaterGAP {wg:+.0f} mm", annotation_position="top left")
        style(fig, 470, "Dec-2004 jump (mm)"); fig.update_layout(yaxis_range=[-160, 120], showlegend=False)
        st.plotly_chart(fig, width="stretch", key="eq_bar")
    with right.container(border=True, key="card_eq_ts"):
        st.subheader("Storage around the earthquake")
        s = series("TH"); d = pd.to_datetime(s["dates"]); fig = go.Figure()
        for kk in ["CSR", "GSFC", "GravIS", "GravIS_GFZ", "G3P"]:
            y = cont(s["products"][kk], d); y = y - y.groupby(y.index.month).transform("mean"); y = y - y["2002":"2004-11"].mean()
            fig.add_trace(go.Scatter(x=d, y=y.rolling(3, center=True, min_periods=1).mean(), name=kk + (" (not corrected)" if kk in ("CSR", "GSFC") else ""),
                                     line=dict(color=PCOL[kk], width=2, dash="dash" if kk in ("CSR", "GSFC") else "solid")))
        st.plotly_chart(nice_ts(fig, 470, "TWSA (mm), seasons removed"), width="stretch", key="eq_ts")

# ================================================================== DATA & METHODS
with tabs[4]:
    html('<div class="sec">How the results were produced</div>')
    STEPS = [("Five GRACE products", "CSR, GSFC, COST-G GravIS, GFZ GravIS and G3P, 2002–2024, regridded to 0.25° and averaged over Thailand and 22 basins."),
             ("Earthquake correction", "CSR and GSFC fail a placebo test (Dec-2004 jump outside −83 to +66 mm); a step + slow after-change is removed using WaterGAP as a water-only reference."),
             ("Trends", "Seasonal Mann–Kendall and Sen slope; false-discovery rate across basins."),
             ("Climate models", "Model A (physical one-store model, Liu et al., 2021) and Model B (random forest, Ji et al., 2025) learn detrended GRACE from CHIRPS rain and GHCN-CAMS temperature to give <b>TWSA climate</b>."),
             ("Evaluation", "Unseen test years (2016–2024 and 2002–2009); comparison with WaterGAP, GLDAS and ESA CCI soil moisture."),
             ("Residual", "TWSA residual = corrected GRACE − TWSA climate = human water use + model error; tested against reservoirs (MSEA-Res), wells (GROW) and groundwater irrigation share (FAO GMIA).")]
    cells = "".join(f'<div class="step"><div class="c">{i}</div><div><div class="h">{h}</div><div class="t">{t}</div></div></div>' for i, (h, t) in enumerate(STEPS, 1))
    html(f'<div class="steps">{cells}</div>')
    html('<div class="callout"><b>Data.</b> CSR, NASA GSFC, GFZ GravIS and G3P; CHIRPS v2.0; GHCN-CAMS; WaterGAP 2.2e; GLDAS Noah; ESA CCI; GROW (IGRAC); MSEA-Res; FAO GMIA v5.</div>')

html('<div class="foot">Farzeen Khan · Master of Engineering, Water Engineering and Management · Asian Institute of Technology · 2026. '
     'Residual values include model error and should be confirmed with ground observations before management decisions.</div>')
