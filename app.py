"""Abaca Girth Prediction System - one-screen Streamlit app.

Run:  streamlit run app.py        (needs abaca_ml.py)
Optional images in ./assets : logo.png, PHflag.png
"""

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import abaca_ml as ml

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(page_title="Abaca Girth Prediction", page_icon="🌴",
                   layout="wide", initial_sidebar_state="collapsed")

# ───────────────────────── themes ─────────────────────────
LIGHT = dict(
    bg="#f1f6ef", glow1="rgba(122,182,97,.30)", glow2="rgba(31,107,63,.14)",
    card="#ffffff", ink="#10281a", muted="#62786a", line="#e1eadf",
    field="#f4f7f2", step="#e6eee2", accent="#1f6b3f", ring="rgba(31,107,63,.18)",
    shadow="0 14px 40px rgba(16,40,26,.08)",
    hero="linear-gradient(150deg,#17482e 0%,#0e3020 60%,#0a2417 100%)",
    hero_ink="#f0fdf4", hero_kicker="#a7e3b6", hero_line="rgba(240,253,244,.2)",
    hero_shadow="0 22px 50px rgba(14,48,32,.35)",
    bar="linear-gradient(90deg,#1f6b3f,#a3cf62)", track="#e8efe5",
    ok_bg="rgba(255,255,255,.14)", ok_ink="#e8fbe9", warn_bg="#fef3c7", warn_ink="#92400e",
    btn="linear-gradient(135deg,#1f6b3f,#14452c)", btn_ink="#ffffff", btn2="#ffffff",
)
DARK = dict(
    bg="#0a1a12", glow1="rgba(122,182,97,.16)", glow2="rgba(31,107,63,.32)",
    card="#112619", ink="#e6f4ea", muted="#8aac96", line="#1e3a2a",
    field="#0b1d14", step="#183324", accent="#b6d97f", ring="rgba(182,217,127,.22)",
    shadow="0 14px 40px rgba(0,0,0,.35)",
    hero="linear-gradient(150deg,#d4e8a8 0%,#bcdb8a 55%,#a6cb72 100%)",
    hero_ink="#12301f", hero_kicker="#33602f", hero_line="rgba(18,48,31,.22)",
    hero_shadow="0 22px 50px rgba(0,0,0,.45)",
    bar="linear-gradient(90deg,#5fa66a,#c5e58a)", track="#193326",
    ok_bg="rgba(18,48,31,.12)", ok_ink="#12301f", warn_bg="#fde68a", warn_ink="#78350f",
    btn="linear-gradient(135deg,#c5e58a,#a6cb72)", btn_ink="#12301f", btn2="#112619",
)
pal = DARK if st.session_state.get("dark", False) else LIGHT
VARS = ":root{" + "".join(f"--{k.replace('_', '-')}:{v};" for k, v in pal.items()) + "}"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=DM+Sans:wght@400;500;700&display=swap');

html, body, .stApp { color-scheme: light; }
.stApp {
  background:
    radial-gradient(900px 520px at 96% -12%, var(--glow1), transparent 70%),
    radial-gradient(800px 520px at -10% 112%, var(--glow2), transparent 70%),
    var(--bg);
  color: var(--ink);
}
.stApp, .stApp p, .stApp label, .stApp span, .stApp div, .stApp input, .stApp button {
  font-family: 'DM Sans', system-ui, sans-serif; }
.stApp p, .stApp label, .stApp span { color: var(--ink); }
.stApp button p { color: inherit; }

#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"], [data-testid="stSidebar"],
[data-testid="collapsedControl"], [data-testid="stAppDeployButton"], .stAppDeployButton,
[data-testid="manage-app-button"] { display: none !important; }

.block-container { max-width: 1480px; padding: 2.4rem 1.8rem .8rem; }
[data-testid="stVerticalBlock"] { gap: .75rem; }

/* header */
.top { display: flex; align-items: center; gap: 16px; margin-bottom: .9rem; }
.top .logo { height: 54px; width: 54px; object-fit: contain; border-radius: 50%; flex-shrink: 0; }
.top .logo-fallback { height: 54px; width: 54px; border-radius: 50%; display: grid;
  place-items: center; font-size: 1.6rem; background: var(--step); flex-shrink: 0; }
.brand { font-family: 'Fraunces', Georgia, serif; font-size: 1.9rem; line-height: 1.1;
  color: var(--ink); font-weight: 500; }
.brand b { font-weight: 700; color: var(--accent); letter-spacing: .02em; }
.sub { font-size: .86rem; color: var(--muted); margin-top: 2px; }
.top .grow { flex: 1; }
.top .flag { height: 34px; width: 54px; object-fit: cover; border-radius: 6px;
  border: 1px solid var(--line); box-shadow: 0 3px 10px rgba(0,0,0,.12); }

/* cards */
[data-testid="stForm"], .card {
  background: var(--card); border: 1px solid var(--line); border-radius: 26px;
  box-shadow: var(--shadow); animation: rise .6s ease both; }
[data-testid="stForm"] { padding: 1.9rem 1.5rem 1.4rem; }
.card { padding: 1.4rem 1.6rem; }
.kicker { font-size: .72rem; letter-spacing: .2em; text-transform: uppercase;
  font-weight: 700; color: var(--accent); }
.h2 { font-family: 'Fraunces', Georgia, serif; font-size: 1.5rem; font-weight: 700;
  color: var(--ink); margin: .2rem 0 .2rem; }

/* inputs */
[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: .86rem; color: var(--ink); }
[data-testid="stWidgetLabel"] svg { color: var(--muted); }
[data-testid="stNumberInputContainer"], [data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--field) !important; border-radius: 14px !important; }
[data-testid="stNumberInputContainer"] { border: 1px solid var(--line); }
[data-baseweb="input"] { border: none !important; }
[data-testid="stNumberInputContainer"]:focus-within {
  border-color: var(--accent); box-shadow: 0 0 0 3px var(--ring); }
[data-testid="stNumberInput"] input { color: var(--ink) !important; font-size: 1.05rem;
  font-weight: 500; height: 2.8rem; background: transparent !important; }
[data-testid="stNumberInputStepUp"], [data-testid="stNumberInputStepDown"] {
  background: var(--step) !important; color: var(--ink) !important; border: none !important; }
[data-testid="stNumberInputStepUp"] svg, [data-testid="stNumberInputStepDown"] svg { fill: var(--ink); }

/* buttons */
button[kind="primaryFormSubmit"], [data-testid="stBaseButton-primaryFormSubmit"] {
  background: var(--btn) !important; color: var(--btn-ink) !important; border: none !important;
  border-radius: 14px; font-weight: 700; height: 3rem;
  box-shadow: 0 10px 22px rgba(20,69,44,.25); transition: transform .15s ease; }
button[kind="primaryFormSubmit"]:hover, [data-testid="stBaseButton-primaryFormSubmit"]:hover {
  transform: translateY(-1px); filter: brightness(1.06); }
button[kind="secondaryFormSubmit"], [data-testid="stBaseButton-secondaryFormSubmit"],
.stDownloadButton > button, [data-testid="stBaseButton-secondary"] {
  background: var(--btn2) !important; color: var(--ink) !important;
  border: 1px solid var(--line) !important; border-radius: 14px; font-weight: 600; height: 3rem; }
button[kind="secondaryFormSubmit"]:hover, .stDownloadButton > button:hover {
  border-color: var(--accent) !important; color: var(--accent) !important; }
[data-testid="stToggle"] p { font-size: .8rem; color: var(--muted); }

/* hero */
.hero { position: relative; overflow: hidden; border-radius: 26px; color: var(--hero-ink);
  padding: 1.5rem 1.7rem 1.2rem; background: var(--hero); box-shadow: var(--hero-shadow);
  animation: rise .6s ease both; }
.hero * { color: var(--hero-ink) !important; }
.hero .kicker { color: var(--hero-kicker) !important; }
.hero .leaf { position: absolute; right: -30px; top: -20px; width: 78%; opacity: .16; }
.hero-title { font-family: 'Fraunces', Georgia, serif; font-size: 1.55rem; margin-top: .5rem; }
.num { font-family: 'Fraunces', Georgia, serif; font-weight: 700; line-height: 1;
  font-size: clamp(3.8rem, 14vh, 6.4rem); margin: clamp(.8rem, 3vh, 1.8rem) 0 .5rem; }
.num small { font-size: 1.6rem; font-weight: 500; margin-left: .45rem; opacity: .8; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; }
.chip { font-size: .8rem; font-weight: 600; padding: 5px 13px; border-radius: 99px;
  background: var(--ok-bg); }
.chip.warn { background: var(--warn-bg); }
.chip.warn, .hero .chip.warn { color: var(--warn-ink) !important; }
.hero-foot { border-top: 1px solid var(--hero-line); margin-top: 1.2rem; padding-top: .8rem;
  font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; opacity: .9; }

/* tiles */
.tiles { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
.tile { background: var(--card); border: 1px solid var(--line); border-radius: 20px;
  padding: .9rem 1.1rem; box-shadow: var(--shadow); animation: rise .7s ease both; }
.tile .t { font-size: .78rem; color: var(--muted); font-weight: 500; }
.tile .v { font-family: 'Fraunces', Georgia, serif; font-size: 1.65rem; font-weight: 700;
  color: var(--ink); line-height: 1.25; }
.tile .u { font-size: .72rem; color: var(--muted); }

/* importance */
.imp { display: flex; flex-direction: column; gap: clamp(.7rem, 2.3vh, 1.4rem); margin-top: 1.1rem; }
.imp-top { display: flex; justify-content: space-between; font-size: .92rem; margin-bottom: 6px; }
.imp-top b { font-weight: 700; }
.track { height: 9px; border-radius: 99px; background: var(--track); overflow: hidden; }
.fill { height: 100%; border-radius: 99px; background: var(--bar);
  animation: grow 1s cubic-bezier(.2,.8,.2,1) both; transform-origin: left; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

@media (max-height: 820px) {
  .block-container { padding-top: 1.6rem; }
  [data-testid="stVerticalBlock"] { gap: .5rem; }
  [data-testid="stNumberInput"] input { height: 2.4rem; }
  [data-testid="stForm"] { padding: 1.5rem 1.3rem 1rem; }
  .hero { padding: 1.2rem 1.5rem 1rem; }
  .card { padding: 1.1rem 1.4rem; }
}
@media (max-width: 900px) {
  .block-container { padding: .9rem .9rem 2rem; }
  .brand { font-size: 1.4rem; }
  .top .flag { height: 26px; width: 40px; }
  .num { font-size: 4.2rem; }
}
"""
st.markdown(f"<style>{VARS}{CSS}</style>", unsafe_allow_html=True)


# ───────────────────────── helpers ─────────────────────────
def flat(s):
    return "".join(line.strip() for line in s.strip().splitlines())


def find_asset(*names):
    wanted = {n.lower() for n in names}
    for folder in (BASE_DIR / "assets", BASE_DIR):
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.name.lower() in wanted:
                    return p
    return None


@st.cache_data(show_spinner=False)
def data_uri(path_str):
    p = Path(path_str)
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".svg": "image/svg+xml"}.get(p.suffix.lower(), "image/png")
    return f"data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"


def display_name(f):
    i = ml.FEATURE_INFO[f]
    return f"{i['label']} ({i['unit']})" if i["unit"] else i["label"]


# ───────────────────────── model ─────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    return ml.load_or_train()


try:
    with st.spinner("Preparing the model…"):
        bundle, notice = get_model()
except ml.ModelUnavailable as exc:
    st.error(f"**Model not ready.** {exc}")
    st.stop()
except Exception as exc:
    st.error(f"**Could not prepare the model.** {exc}")
    st.stop()
if notice:
    st.toast(notice, icon="⚠️")


def slider_bounds(b):
    out = {}
    for f in ml.FEATURES:
        info = ml.FEATURE_INFO[f]
        dmin, dmax = b["training_ranges"][f]
        pad = 0.10 * (dmax - dmin)
        plo, phi = info["phys"]
        lo, hi = max(plo, dmin - pad), min(phi, dmax + pad)
        if info["kind"] == "int":
            lo, hi = int(np.floor(lo)), int(np.ceil(hi))
            if hi <= lo:
                hi = lo + 1
        else:
            step = info["step"]
            lo = round(float(np.floor(lo / step) * step), 4)
            hi = round(float(np.ceil(hi / step) * step), 4)
            lo, hi = max(lo, float(plo)), min(hi, float(phi))
            if hi - lo < step:
                lo, hi = max(float(plo), lo - 5 * step), min(float(phi), hi + 5 * step)
        out[f] = (lo, hi)
    # fixed ranges: relative humidity and shade level -> 40 to 100
    for f in ("humidity_pct", "sun_shade_pct"):
        if f in out:
            if ml.FEATURE_INFO[f]["kind"] == "int":
                out[f] = (40, 100)
            else:
                out[f] = (40.0, 100.0)
    return out


def typical_value(b, bounds, f):
    info = ml.FEATURE_INFO[f]
    lo, hi = bounds[f]
    v = min(max(b["medians"][f], lo), hi)
    if info["kind"] == "int":
        return int(round(v))
    step = info["step"]
    return float(min(max(round(round(v / step) * step, 4), lo), hi))


bounds = slider_bounds(bundle)
typical = {f: typical_value(bundle, bounds, f) for f in ml.FEATURES}


def reset_all():
    for f, v in typical.items():
        st.session_state[f] = v
    st.session_state["submitted"] = dict(typical)


for f in ml.FEATURES:
    if f not in st.session_state:
        st.session_state[f] = typical[f]
    else:
        lo, hi = bounds[f]
        st.session_state[f] = type(typical[f])(min(max(st.session_state[f], lo), hi))
if "submitted" not in st.session_state:
    st.session_state["submitted"] = dict(typical)

# ───────────────────────── header ─────────────────────────
logo = find_asset("logo.png", "logo.jpg", "logo.jpeg", "logo.webp")
flag = find_asset("PHflag.png", "PHflag.jpg", "PHflag.jpeg", "PHflag.webp", "ph_flag.png", "flag.png")
logo_html = (f'<img class="logo" src="{data_uri(str(logo))}" alt="">' if logo
             else '<div class="logo-fallback">🌴</div>')
flag_html = f'<img class="flag" src="{data_uri(str(flag))}" alt="">' if flag else ""

h1, h2 = st.columns([14, 1.3], vertical_alignment="center")
with h1:
    st.markdown(flat(f"""
    <div class="top">{logo_html}
      <div><div class="brand"><b>ABACA</b> / Girth Prediction</div>
      <div class="sub">San Rafael National High School · Wilmer C. Valencia</div></div>
      <div class="grow"></div>{flag_html}
    </div>"""), unsafe_allow_html=True)
with h2:
    st.toggle("Dark", key="dark")

# ───────────────────────── body ─────────────────────────
c_in, c_out, c_imp = st.columns([1.1, 1, .9], gap="medium")

with c_in:
    with st.form("measure"):
        st.markdown('<div class="kicker">01 / Measure</div>'
                    '<div class="h2">Plant measurements</div>', unsafe_allow_html=True)
        grid = st.columns(2, gap="medium")
        for i, f in enumerate(ml.FEATURES):
            info = ml.FEATURE_INFO[f]
            lo, hi = bounds[f]
            with grid[i % 2]:
                if info["kind"] == "int":
                    st.number_input(display_name(f), min_value=int(lo), max_value=int(hi),
                                    step=1, format="%d", key=f, help=info["help"])
                else:
                    st.number_input(display_name(f), min_value=float(lo), max_value=float(hi),
                                    step=float(info["step"]), format=f"%.{info['decimals']}f",
                                    key=f, help=info["help"])
        b1, b2 = st.columns([1.35, 1], gap="small")
        with b1:
            go = st.form_submit_button("Predict girth", type="primary", use_container_width=True)
        with b2:
            st.form_submit_button("Reset", on_click=reset_all, use_container_width=True)

if go:
    st.session_state["submitted"] = {
        f: (int(st.session_state[f]) if ml.FEATURE_INFO[f]["kind"] == "int"
            else float(st.session_state[f])) for f in ml.FEATURES}

inputs = st.session_state["submitted"]
try:
    result = ml.predict(bundle, inputs)
except ValueError as exc:
    st.error(f"Invalid measurement: {exc}")
    st.stop()

with c_out:
    st.markdown(flat(f"""
    <div class="hero">
      <svg class="leaf" viewBox="0 0 400 300" fill="none" stroke="currentColor" stroke-width="1.4">
        <path d="M20 290 C 90 150, 210 50, 390 10 C 340 120, 220 230, 20 290 Z"/>
        <path d="M20 290 C 120 190, 250 100, 390 10"/>
        <path d="M90 230 C 130 200, 160 170, 190 120 M140 200 C 190 180, 230 150, 270 90 M200 170 C 250 150, 290 120, 330 60"/>
      </svg>
      <div class="kicker">02 / Predict</div>
      <div class="hero-title">Predicted girth</div>
      <div class="num">{result["prediction"]:.2f}<small>cm</small></div>
      <div class="hero-foot">Random forest · {bundle["n_records"]} plant records</div>
    </div>"""), unsafe_allow_html=True)

    st.download_button("Download CSV",
                       data=pd.DataFrame([{**inputs, "predicted_girth_cm": round(result["prediction"], 2)}])
                       .to_csv(index=False),
                       file_name="abaca_prediction.csv", mime="text/csv",
                       use_container_width=True)

    rf = bundle["metrics"]["rf"]
    st.markdown(flat(f"""
    <div class="tiles">
      <div class="tile"><div class="t">R²</div><div class="v">{rf["r2"]:.3f}</div><div class="u">score</div></div>
      <div class="tile"><div class="t">MAE</div><div class="v">{rf["mae_cm"]:.2f}</div><div class="u">cm</div></div>
      <div class="tile"><div class="t">RMSE</div><div class="v">{rf["rmse_cm"]:.2f}</div><div class="u">cm</div></div>
    </div>"""), unsafe_allow_html=True)

with c_imp:
    imp = {f: float(bundle["importance_impurity"][f]) for f in ml.FEATURES}
    total = sum(imp.values()) or 1.0
    rows = sorted(((ml.LABEL_OF[f], v / total * 100) for f, v in imp.items()),
                  key=lambda r: r[1], reverse=True)
    top = rows[0][1] or 1.0
    bars = "".join(
        f'<div><div class="imp-top"><span>{name}</span><b>{v:.2f}%</b></div>'
        f'<div class="track"><div class="fill" style="width:{max(v / top * 100, 2):.1f}%;'
        f'animation-delay:{i * 0.07:.2f}s"></div></div></div>'
        for i, (name, v) in enumerate(rows))
    st.markdown(flat(f"""
    <div class="card">
      <div class="kicker">03 / Understand</div>
      <div class="h2">Feature importance</div>
      <div class="imp">{bars}</div>
    </div>"""), unsafe_allow_html=True)