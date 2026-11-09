"""Thailand Water Storage Decision Support System (Streamlit).

Earthquake-corrected GRACE storage, climate-driven storage (Models A and B) and the
residual (human water use + model error) for Thailand and its 22 river basins.
Author: Farzeen Khan, Asian Institute of Technology (2026).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA = Path(__file__).parent / "data"
st.set_page_config(page_title="Thailand Water Storage DSS", page_icon="💧", layout="wide")

# ------------------------------------------------------------------ style
st.markdown("""
<style>
.block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1400px;}
h1, h2, h3 {color: #123650;}
.kpi {background: #f4f7fb; border: 1px solid #dbe4ef; border-radius: 10px; padding: 14px 16px; height: 100%;}
.kpi .v {font-size: 1.9rem; font-weight: 700; color: #123650; line-height: 1.1;}
.kpi .l {font-size: .9rem; color: #4c5961; margin-top: 4px;}
.tag {display: inline-block; padding: 2px 10px; border-radius: 12px; color: white; font-weight: 600; font-size: .85rem;}
.note {font-size: .88rem; color: #4c5961;}
</style>""", unsafe_allow_html=True)

# ------------------------------------------------------------------ data
@st.cache_data
def load():
    R = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
    G = json.loads((DATA / "basins.geojson").read_text(encoding="utf-8"))
    P = json.loads((DATA / "points.json").read_text(encoding="utf-8"))
    return R, G, P

@st.cache_data
def series(rid):
    return json.loads((DATA / "ts" / f"{rid}.json").read_text(encoding="utf-8"))

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
CLASS_COL = {"High": "#c62828", "Medium": "#f9a825", "Low": "#2e7d32", "Uncertain": "#9e9e9e"}
ACTIONS = {"High": "Introduce monthly groundwater-level monitoring, meter and license large abstractions, review dry-season irrigation demand, assess managed aquifer recharge.",
           "Medium": "Verify the residual with wells and reservoir records, apply dry-season demand management, include in the next groundwater survey.",
           "Low": "Storage change is small or mostly climate-driven: maintain the existing observation network.",
           "Uncertain": "Affected by the 2004 earthquake signal or model limits: collect ground observations before deciding."}

GAP = ("2017-07-01", "2018-05-31")
def cont(values, idx):
    """Continuous monthly line: gaps of up to 3 months filled; only the 2017-18 satellite gap stays open."""
    y = pd.Series(values, index=idx, dtype=float).interpolate(limit=3, limit_area="inside")
    y[(y.index >= GAP[0]) & (y.index <= GAP[1])] = np.nan
    return y

def kpi(col, value, label):
    col.markdown(f'<div class="kpi"><div class="v">{value}</div><div class="l">{label}</div></div>', unsafe_allow_html=True)

def basemap(z, locs, colorscale, zmin, zmax, title, hover, height=560, line="white"):
    fig = go.Figure(go.Choropleth(geojson=GEO, locations=locs, z=z, colorscale=colorscale, zmin=zmin, zmax=zmax,
                                  marker=dict(line=dict(width=.8, color=line)), text=hover, hoverinfo="text",
                                  colorbar=dict(title=dict(text=title, side="right"), thickness=14, len=.8)))
    fig.update_geos(fitbounds="locations", visible=False, projection_type="mercator", bgcolor="rgba(0,0,0,0)")
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=height, paper_bgcolor="rgba(0,0,0,0)")
    return fig

def nice_ts(fig, height=420, ytitle="TWSA (mm)"):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), template="simple_white", hovermode="x unified",
                      yaxis_title=ytitle, legend=dict(orientation="h", y=-0.18), font=dict(family="Times New Roman", size=15))
    fig.add_vline(x="2004-12-26", line=dict(color="#c0392b", dash="dot"))
    fig.add_vrect(x0="2017-07-01", x1="2018-05-31", fillcolor="#e6e6e6", line_width=0, layer="below")
    return fig

# ------------------------------------------------------------------ header
st.title("Thailand Water Storage Decision Support System")
st.caption("Earthquake-corrected GRACE storage · climate-driven storage (Models A and B) · residual = human water use + model error · 22 river basins · 2002–2024")
tabs = st.tabs(["📊 Overview", "🗺️ Basin Explorer", "🎯 Priority Index", "🌏 Earthquake Check", "📘 Data & Methods"])

# ================================================================== OVERVIEW
with tabs[0]:
    H = META["headline"]
    c = st.columns(5)
    kpi(c[0], f"{H['ens_sen']:.2f} mm/yr", "National storage trend (corrected, Sen slope)")
    kpi(c[1], "≈ 31 km³", "Water lost since 2002")
    kpi(c[2], f"{H['n_sig']} of 22", "Basins with a significant decline")
    kpi(c[3], "−2.6 to −4.7", "Residual trend, mm/yr (human + model error)")
    kpi(c[4], "19 of 20", "Model tests satisfactory on unseen years")
    st.markdown("")
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader("Thailand: storage, climate and residual")
        mode = st.radio("Show", ["Five corrected products", "TWSA = climate + residual"], horizontal=True, key="ov_mode")
        s = series("TH"); d = pd.to_datetime(s["dates"])
        fig = go.Figure()
        if mode.startswith("Five"):
            for k in TARG:
                y = cont(s["model"][k]["grace"], d).rolling(12, center=True, min_periods=6).mean()
                fig.add_trace(go.Scatter(x=d, y=y, name=TLAB[k], line=dict(color=PCOL[k], width=2)))
        else:
            mdl = st.selectbox("Product", TARG, format_func=TLAB.get, key="ov_p")
            for comp, col, nm in (("grace", "#222222", "Corrected GRACE TWSA"), ("climate", "#332288", "TWSA climate"), ("residual", "#C0392B", "TWSA residual")):
                y = cont(s["model"][mdl][comp], d)
                y = (y - y.groupby(y.index.month).transform("mean")).rolling(12, center=True, min_periods=6).mean()
                fig.add_trace(go.Scatter(x=d, y=y, name=nm, line=dict(color=col, width=2.6)))
        st.plotly_chart(nice_ts(fig), width="stretch")
    with right:
        st.subheader("Residual trend by basin")
        p = st.selectbox("GRACE product", TARG, index=2, format_func=TLAB.get, key="ov_map")
        z = [REG[b]["residual_by_product"].get(p, {}).get("trend") for b in BAS]
        hov = [f"<b>{NAME[b]}</b><br>Residual {v:+.1f} mm/yr" if v is not None else NAME[b] for b, v in zip(BAS, z)]
        st.plotly_chart(basemap(z, BAS, "RdBu", -8, 8, "mm/yr", hov, height=470), width="stretch", key="ov_chart")
        st.markdown('<div class="note">Red = storage loss not explained by rain and temperature.</div>', unsafe_allow_html=True)
    st.info("**Key message.** Thailand is losing water (≈ 2.7 mm/yr). Rain and temperature explain the floods and droughts but not the decline; "
            "the unexplained part is largest in the irrigated central plains, where well levels confirm a groundwater signal.")

# ================================================================== EXPLORER
with tabs[1]:
    c1, c2, c3 = st.columns([1.2, 1, 1])
    rid = c1.selectbox("Basin", ["TH"] + BAS, format_func=lambda k: f"{k} · {NAME[k]}" if k != "TH" else "Thailand (national)")
    prod = c2.selectbox("Product", TARG, index=2, format_func=TLAB.get, key="ex_p")
    view = c3.radio("Series", ["Five products", "Decomposition", "Model fit"], horizontal=True)
    r = REG[rid]; a = r.get("attributes", {}) or {}
    k = st.columns(5)
    tr = r["trend"].get(prod, {}); rb = r["residual_by_product"].get(prod, {})
    kpi(k[0], f"{tr.get('sen', float('nan')):+.1f}", "Storage trend, mm/yr (Sen)")
    kpi(k[1], f"{rb.get('trend', float('nan')):+.1f}", "Residual trend, mm/yr")
    kpi(k[2], f"{rb.get('climate', float('nan')):+.1f}", "Climate trend, mm/yr")
    kpi(k[3], f"{a.get('irrigated_pct', 0):.0f}%", "Area irrigated")
    kpi(k[4], f"{a.get('gw_share_pct', 0):.0f}%", "Irrigation from groundwater")
    s = series(rid); d = pd.to_datetime(s["dates"]); fig = go.Figure()
    if view == "Five products":
        for kk in TARG:
            fig.add_trace(go.Scatter(x=d, y=cont(s["model"][kk]["grace"], d), name=TLAB[kk], line=dict(color=PCOL[kk], width=1.6)))
    elif view == "Decomposition":
        for comp, col, nm in (("grace", "#222222", "Corrected GRACE TWSA"), ("climate", "#332288", "TWSA climate"), ("residual", "#C0392B", "TWSA residual")):
            fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod][comp], d), name=nm, line=dict(color=col, width=2)))
    else:
        fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod]["detrended"], d), name="Detrended GRACE TWSA", line=dict(color=PCOL[prod], width=2.6)))
        fig.add_trace(go.Scatter(x=d, y=cont(s["model"][prod]["climate"], d), name="TWSA climate (model)", line=dict(color="#332288", width=1.6, dash="dash")))
    st.plotly_chart(nice_ts(fig, 440), width="stretch")
    if rid != "TH":
        sk = (r.get("skill") or {}).get(prod, {})
        st.markdown(f"**Model skill ({TLAB[prod]})** — calibration NSE {sk.get('cal', float('nan')):.2f} · "
                    f"prediction-test NSE {sk.get('pred', float('nan')):.2f} · r {sk.get('r', float('nan')):.2f}")
        if r.get("group") == "M_S":
            st.warning("Southern peninsula basin: nearest the 2004 earthquake — results are uncertain.")

# ================================================================== PRIORITY
W0 = {"loss": 30, "conf": 20, "gw": 20, "irr": 10, "decline": 10, "agree": 10}
WLAB = {"loss": "Residual loss", "conf": "Confidence (95% range below zero)", "gw": "Groundwater dependence",
        "irr": "Irrigation intensity", "decline": "Total storage decline", "agree": "Product agreement"}
clamp = lambda v: float(min(1, max(0, v)))
with tabs[2]:
    st.subheader("Which basins need action first?")
    left, right = st.columns([1, 2.2])
    with left:
        p = st.selectbox("GRACE product", ["GRACE_ensemble"] + TARG, format_func=TLAB.get, key="pr_p")
        flag = st.checkbox("Mark southern basins as Uncertain", True)
        st.markdown("**Weights**")
        W = {k: st.slider(WLAB[k], 0, 50, v, 5, key=f"w_{k}") for k, v in W0.items()}
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
    df = pd.DataFrame(rows); df["o"] = (df.Class == "Uncertain").astype(int); df = df.sort_values(["o", "Score"], ascending=[True, False]).drop(columns="o")
    with right:
        cmap = {"High": 3, "Medium": 2, "Low": 1, "Uncertain": 0}
        z = [cmap[c] for c in df.set_index("id").loc[BAS].Class]
        hov = [f"<b>{NAME[b]}</b><br>{c} · score {s}" for b, c, s in zip(BAS, df.set_index("id").loc[BAS].Class, df.set_index("id").loc[BAS].Score)]
        cs = [[0, "#9e9e9e"], [.33, "#9e9e9e"], [.34, "#2e7d32"], [.55, "#2e7d32"], [.56, "#f9a825"], [.8, "#f9a825"], [.81, "#c62828"], [1, "#c62828"]]
        fig = basemap(z, BAS, cs, 0, 3, "Class", hov, height=520); fig.update_traces(showscale=False)
        st.plotly_chart(fig, width="stretch")
        st.markdown(" ".join(f'<span class="tag" style="background:{CLASS_COL[c]}">{c}</span>' for c in CLASS_COL), unsafe_allow_html=True)
    show = df.rename(columns={"Residual": "Residual (mm/yr)", "GW": "Groundwater share (%)", "Irr": "Irrigated (%)"})
    show["Recommended action"] = show.Class.map(ACTIONS)
    st.dataframe(show.drop(columns="id"), width="stretch", hide_index=True,
                 column_config={"Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d")})
    st.download_button("⬇ Download ranking (CSV)", show.to_csv(index=False).encode("utf-8"), "priority_ranking.csv", "text/csv")

# ================================================================== EARTHQUAKE
with tabs[3]:
    st.subheader("The 2004 Sumatra–Andaman earthquake signal in GRACE")
    left, right = st.columns([1, 1])
    sc = pd.DataFrame(META["screening"]).set_index("product").loc[["CSR", "GSFC", "GravIS", "GravIS_GFZ", "G3P"]].reset_index()
    sc["short"] = ["CSR", "GSFC", "COST-G GravIS", "GFZ GravIS", "G3P"]
    with left:
        fig = go.Figure(go.Bar(x=sc.short, y=sc.jump, marker_color=[PCOL[p] for p in sc["product"]],
                               text=[f"{v:+.0f} mm<br>{'PASS' if ok else 'FAIL'}" for v, ok in zip(sc.jump, sc.passed)], textposition="outside"))
        wg = sc.watergap.iloc[0]
        fig.add_hrect(y0=wg - 83, y1=wg + 66, fillcolor="#dceee3", line_width=0, layer="below", annotation_text="normal range", annotation_position="bottom right")
        fig.add_hline(y=wg, line=dict(color="#D55E00", dash="dot"), annotation_text=f"WaterGAP {wg:+.0f} mm", annotation_position="top left")
        fig.update_layout(template="simple_white", height=460, yaxis_title="Dec-2004 jump (mm)", yaxis_range=[-160, 120], margin=dict(t=30), font=dict(family="Times New Roman", size=15))
        st.plotly_chart(fig, width="stretch")
    with right:
        s = series("TH"); d = pd.to_datetime(s["dates"]); fig = go.Figure()
        for kk in ["CSR", "GSFC", "GravIS", "GravIS_GFZ", "G3P"]:
            y = cont(s["products"][kk], d); y = y - y.groupby(y.index.month).transform("mean"); y = y - y["2002":"2004-11"].mean()
            fig.add_trace(go.Scatter(x=d, y=y.rolling(3, center=True, min_periods=1).mean(), name=kk + (" (not corrected)" if kk in ("CSR", "GSFC") else ""),
                                     line=dict(color=PCOL[kk], width=2, dash="dash" if kk in ("CSR", "GSFC") else "solid")))
        st.plotly_chart(nice_ts(fig, 460, "TWSA (mm), seasons removed"), width="stretch")
    st.markdown("CSR and GSFC drop ≈ 100–150 mm after December 2004 while COST-G, GFZ and G3P do not — a gravity signal from the earthquake, "
                "largest in the southern peninsula. CSR and GSFC were corrected by fitting a sudden step plus a slow after-change to GRACE − WaterGAP in every 0.25° cell.")

# ================================================================== DATA & METHODS
with tabs[4]:
    st.subheader("How the results were produced")
    st.markdown("""
1. **Five GRACE products** (CSR, GSFC, COST-G GravIS, GFZ GravIS, G3P), 2002–2024, regridded to 0.25° and averaged over Thailand and 22 basins.
2. **Earthquake correction:** CSR and GSFC fail a placebo test (Dec-2004 jump outside −83 to +66 mm); a step + slow after-change is removed using WaterGAP as a water-only reference.
3. **Trends:** seasonal Mann–Kendall and Sen slope; false-discovery rate across basins.
4. **Climate models:** Model A (physical one-store model, Liu et al., 2021) and Model B (random forest, Ji et al., 2025) learn detrended GRACE from CHIRPS rain and GHCN-CAMS temperature → **TWSA climate**.
5. **Evaluation:** unseen test years (2016–2024 and 2002–2009); comparison with WaterGAP, GLDAS and ESA CCI soil moisture.
6. **Residual:** TWSA residual = corrected GRACE − TWSA climate = human water use + model error; tested against reservoirs (MSEA-Res), wells (GROW) and groundwater irrigation share (FAO GMIA).
""")
    st.markdown("**Data:** CSR, NASA GSFC, GFZ GravIS and G3P; CHIRPS v2.0; GHCN-CAMS; WaterGAP 2.2e; GLDAS Noah; ESA CCI; GROW (IGRAC); MSEA-Res; FAO GMIA v5.")
    st.caption("Farzeen Khan · Master of Engineering, Water Engineering and Management · Asian Institute of Technology · 2026. "
               "Residual values include model error and should be confirmed with ground observations before management decisions.")
