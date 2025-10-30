import os, io, re, requests
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import joblib  # <-- NEW

# =============== APP CONFIG ===============
st.set_page_config(page_title="Healthcare Dashboards", page_icon="📊", layout="wide")

# --- Compact KPI/heading styles for 14" screens ---
st.markdown("""
<style>
h2, .stSubheader { font-size: 1.05rem !important; margin-bottom: 0.35rem !important; }
[data-testid="stMetricValue"] { font-size: 1.55rem !important; }
[data-testid="stMetric"] { padding: 0.30rem 0.35rem !important; }
.block-container { padding-top: 0.9rem; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Tighten spacing in the sidebar */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3{
  margin-top: .25rem !important;
  margin-bottom: .35rem !important;
}

/* Reduce gaps between radio options */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"]{
  gap: .2rem !important;
}

/* Trim outer margins for each radio widget */
[data-testid="stSidebar"] .stRadio{
  margin-top: .2rem !important;
  margin-bottom: .6rem !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Make sidebar a flex column so we can push the logo to the bottom */
[data-testid="stSidebar"] > div:first-child {
  height: 100%;
  display: flex;
  flex-direction: column;
}
/* Bottom container styling */
.sidebar-bottom {
  margin-top: auto;
  padding-top: .5rem;
  border-top: 1px solid rgba(255,255,255,.08);
}
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "Dashboard_Data_Files"
BF_STEMS = ["bf_cost_ready"]
IP_STEMS = ["ip_curated", "inpatient", "ip_claims", "ip_ready", "inpatient_ready"]
OP_STEMS = ["op_curated", "outpatient", "op_claims", "op_ready", "outpatient_ready"]

# ====== Cost Prediction constants & encodings (NEW) ======
STATE_MAP = {
    "01":"AL","02":"AK","03":"AZ","04":"AR","05":"CA","06":"CO","07":"CT","08":"DE","09":"DC",
    "10":"FL","11":"GA","12":"HI","13":"ID","14":"IL","15":"IN","16":"IA","17":"KS","18":"KY",
    "19":"LA","20":"ME","21":"MD","22":"MA","23":"MI","24":"MN","25":"MS","26":"MO","27":"MT",
    "28":"NE","29":"NV","30":"NH","31":"NJ","32":"NM","33":"NY","34":"NC","35":"ND","36":"OH",
    "37":"OK","38":"OR","39":"PA","41":"RI","42":"SC","43":"SD","44":"TN","45":"TX","46":"UT",
    "47":"VT","49":"VA","50":"WA","51":"WV","52":"WI","53":"WY","54":"Others"
}
STATE_ABBR_TO_CODE = {v: k for k, v in STATE_MAP.items()}  # e.g., "CA" -> "05"

SEX_MAP  = {"Female": 0, "Male": 1}
RACE_MAP = {"Black": 0, "Hispanic": 1, "Others": 2, "White": 3}

CHRONIC_LIST = [
    'SP_ALZHDMTA','SP_CHF','SP_CHRNKIDN','SP_CNCR','SP_COPD',
    'SP_DEPRESSN','SP_DIABETES','SP_ISCHMCHT','SP_OSTEOPRS',
    'SP_RA_OA','SP_STRKETIA'
]
FEATURE_ORDER = [
    'sex_label','race_label','SP_STATE_CODE','BENE_ESRD_IND',
    'age_latest','chronic_count','los_total','dx_total','proc_total',
    'SP_ALZHDMTA','SP_CHF','SP_CHRNKIDN','SP_CNCR','SP_COPD',
    'SP_DEPRESSN','SP_DIABETES','SP_ISCHMCHT','SP_OSTEOPRS',
    'SP_RA_OA','SP_STRKETIA'
]

CLASS_TO_LABEL = {0: "Low", 1: "Medium", 2: "High"}
LABEL_COLOR = {"Low": "#22c55e", "Medium": "#eab308", "High": "#ef4444"}  # green, amber, red

# ======================== GOOGLE DRIVE REMOTE LOADERS =========================
DRIVE_LINKS = {
    "bf_csv": "https://drive.google.com/file/d/1Kreo7FQVOUJ9uZGylC4b_vXLloJpdFg2/view?usp=sharing",
    "bf_xlsx": "https://docs.google.com/spreadsheets/d/1Ml_o_eUGtwg_PSPAYigPSC3xYuZDR_CQ/edit?usp=sharing&rtpof=true&sd=true",
    "ip_csv": "https://drive.google.com/file/d/1Ar5YqCOWwXdv766sd4ZMKuy3r0X8-Rvp/view?usp=sharing",
    "ip_xlsx": "https://docs.google.com/spreadsheets/d/154hmxIB78XKrfMEsIN5bD_Jz080DJpup/edit?usp=sharing&rtpof=true&sd=true",
    "op_csv": "https://drive.google.com/file/d/1H3qDfK6z3xVYk0vvZpNx8pRpAeiGDAKa/view?usp=sharing",
    "op_xlsx": "https://docs.google.com/spreadsheets/d/1EjV4XqBrsfeP-3fLNVWG1jBDuIrNtWYe/edit?usp=sharing&rtpof=true&sd=true",
}

def _extract_file_id(url: str) -> Optional[str]:
    m = re.search(r"/d/([a-zA-Z0-9_-]{20,})/", url)
    return m.group(1) if m else None

def _gdrive_export_url(url: str, fmt: str = "csv") -> str:
    fid = _extract_file_id(url)
    if not fid:
        return url
    if "spreadsheets" in url:
        fmt = fmt.lower()
        if fmt not in {"csv", "xlsx"}:
            fmt = "csv"
        return f"https://docs.google.com/spreadsheets/d/{fid}/export?format={fmt}"
    return f"https://drive.google.com/uc?export=download&id={fid}"

@st.cache_data(show_spinner=False)
def read_gdrive_csv(url: str) -> pd.DataFrame:
    direct = _gdrive_export_url(url, fmt="csv")
    try:
        return pd.read_csv(direct)
    except Exception:
        r = requests.get(direct, allow_redirects=True)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.content.decode("utf-8", errors="ignore")))

@st.cache_data(show_spinner=False)
def read_gdrive_excel(url: str, sheet_name=0) -> pd.DataFrame:
    direct = _gdrive_export_url(url, fmt="xlsx")
    r = requests.get(direct, allow_redirects=True)
    r.raise_for_status()
    return pd.read_excel(io.BytesIO(r.content), sheet_name=sheet_name)

def load_bf_remote() -> pd.DataFrame:
    try:
        return read_gdrive_csv(DRIVE_LINKS["bf_csv"])
    except Exception:
        return read_gdrive_excel(DRIVE_LINKS["bf_xlsx"])

def load_ip_remote() -> pd.DataFrame:
    try:
        return read_gdrive_csv(DRIVE_LINKS["ip_csv"])
    except Exception:
        return read_gdrive_excel(DRIVE_LINKS["ip_xlsx"])

def load_op_remote() -> pd.DataFrame:
    try:
        return read_gdrive_csv(DRIVE_LINKS["op_csv"])
    except Exception:
        return read_gdrive_excel(DRIVE_LINKS["op_xlsx"])

# ======================== SPEED LAYER: Cached fast I/O ========================
def find_file(stems: List[str]) -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    for s in stems:
        for ext in (".xlsx", ".xls"):
            p = DATA_DIR / f"{s}{ext}"
            if p.exists():
                return p
    for s in stems:
        p = DATA_DIR / f"{s}.csv"
        if p.exists():
            return p
    return None

def _parquet_sidecar(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".parquet")

def _needs_rebuild(src: Path, pq: Path) -> bool:
    if not pq.exists():
        return True
    try:
        return os.path.getmtime(src) > os.path.getmtime(pq)
    except Exception:
        return True

@st.cache_data(show_spinner=False)
def _read_excel_cached(path_str: str) -> pd.DataFrame:
    return pd.read_excel(path_str, sheet_name=0, header=0)

@st.cache_data(show_spinner=False)
def _read_csv_cached(path_str: str) -> pd.DataFrame:
    df = pd.read_csv(path_str, engine="python")
    if df.shape[1] == 1 and ("," in str(df.columns[0]).lower()):
        raw = pd.read_csv(path_str, header=None, engine="python")
        if raw.shape[1] == 1:
            parts = raw.iloc[:, 0].astype(str).str.split(",", expand=True)
            parts.columns = parts.iloc[0].astype(str).str.strip()
            df = parts.iloc[1:].reset_index(drop=True)
        else:
            raw.columns = raw.iloc[0].astype(str).str.strip()
            df = raw.iloc[1:].reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def _read_parquet_cached(path_str: str) -> pd.DataFrame:
    return pd.read_parquet(path_str, engine="pyarrow")

def fast_read(path: Path) -> pd.DataFrame:
    pq = _parquet_sidecar(path)
    suf = path.suffix.lower()
    if _needs_rebuild(path, pq):
        if suf == ".csv":
            df = _read_csv_cached(str(path))
        elif suf in (".xlsx", ".xls"):
            df = _read_excel_cached(str(path))
        else:
            raise ValueError(f"Unsupported file: {path}")
        try:
            df.to_parquet(pq, engine="pyarrow", index=False)
        except Exception:
            return df
        return df
    return _read_parquet_cached(str(pq))

# ======================== SPEED LAYER: Cached heavy calcs =====================
@st.cache_data(show_spinner=False)
def compute_prevalence_over_years(df: pd.DataFrame, chronic_cols: Tuple[str, ...]) -> pd.DataFrame:
    work = df[pd.to_numeric(df["Year"], errors="coerce").notna()].copy()
    work["Year"] = work["Year"].astype(int)
    recs = []
    for yr, grp in work.groupby("Year"):
        denom = len(grp)
        if denom == 0:
            continue
        for c in chronic_cols:
            if c in grp.columns:
                prev = 100.0 * pd.to_numeric(grp[c], errors="coerce").fillna(0).astype(int).sum() / denom
                recs.append({"Year": yr, "Condition": c, "PrevalencePct": prev})
    return pd.DataFrame(recs)

@st.cache_data(show_spinner=False)
def compute_top3_race_prev(dff: pd.DataFrame, top3: Tuple[str, ...]) -> pd.DataFrame:
    rows = []
    if "race_label" not in dff.columns:
        return pd.DataFrame()
    for race, grp in dff.groupby("race_label"):
        denom = len(grp)
        if denom == 0:
            continue
        for c in top3:
            if c in grp.columns:
                prev = 100.0 * pd.to_numeric(grp[c], errors="coerce").fillna(0).astype(int).sum() / denom
                rows.append({"race_label": race, "Condition": c, "PrevalencePct": prev})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def compute_ageband_costs(dff: pd.DataFrame, combine_groups: Dict[str, list]) -> pd.DataFrame:
    needed = sorted({col for cols in combine_groups.values() for col in cols})
    tmp = dff[["age_band_20y"] + needed].copy()
    for c in needed:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(0)
    for new_name, cols in combine_groups.items():
        tmp[new_name] = tmp[cols].sum(axis=1)
    return (
        tmp.groupby("age_band_20y", as_index=False)[list(combine_groups.keys())]
           .mean()
           .melt(id_vars="age_band_20y", var_name="Metric", value_name="Value")
    )

# ======================== MODEL LOADER (cached) ===============================
@st.cache_resource(show_spinner=True)
def load_rf_model():
    candidates = [
        Path(__file__).parent / "rf_model.pkl",
        Path(__file__).parent / "models" / "rf_model.pkl",
        DATA_DIR / "rf_model.pkl",
    ]
    for p in candidates:
        if p.exists():
            return joblib.load(p)
    raise FileNotFoundError(
        "rf_model.pkl not found. Put it next to app.py, in ./models/, or inside Dashboard_Data_Files/."
    )

# ======================== NAVIGATION (sidebar) ================================
if "active_page" not in st.session_state:
    st.session_state.active_page = ("dash", "Beneficiary")  # section, item

def set_active_from_dash():
    st.session_state.active_page = ("dash", st.session_state.nav_dash)

def set_active_from_ml():
    st.session_state.active_page = ("ml", st.session_state.nav_ml)

with st.sidebar:
    st.header("Dashboards")
    st.radio(
        label="",
        options=["Beneficiary", "Inpatient", "Outpatient"],
        key="nav_dash",
        index=["Beneficiary", "Inpatient", "Outpatient"].index(
            st.session_state.active_page[1] if st.session_state.active_page[0]=="dash" else "Beneficiary"
        ),
        on_change=set_active_from_dash,
        label_visibility="collapsed",
    )

    st.header("Machine Learning")
    st.radio(
        label="",
        options=["Cost Prediction", "Readmission Rate"],
        key="nav_ml",
        index=["Cost Prediction", "Readmission Rate"].index(
            st.session_state.active_page[1] if st.session_state.active_page[0]=="ml" else "Cost Prediction"
        ),
        on_change=set_active_from_ml,
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    st.markdown('<div class="sidebar-bottom">', unsafe_allow_html=True)
    st.image(
        str(Path(__file__).parent / "streamlit_app_logo.png"),
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

section, page = st.session_state.active_page

# -------------------- Helper to render Cost Prediction (NEW) ------------------
def render_cost_prediction():
    st.title("💰 Cost Prediction")

    with st.expander("About this model", expanded=False):
        st.markdown(
            "- Target: **spend_label** (`Low`, `Medium`, `High`)\n"
            "- Estimator: Random Forest (`rf_model.pkl`)\n"
            "- Encodings match training exactly."
        )

    # Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sex_ui  = st.selectbox("Sex", list(SEX_MAP.keys()), index=1)
        race_ui = st.selectbox("Race", list(RACE_MAP.keys()), index=3)
    with c2:
        state_ui = st.selectbox(
            "State (abbr.)",
            sorted(STATE_ABBR_TO_CODE.keys()),
            index=sorted(STATE_ABBR_TO_CODE).index("CA") if "CA" in STATE_ABBR_TO_CODE else 0
        )
        esrd_ui  = st.selectbox("ESRD", ["No", "Yes"])
    with c3:
        age_ui   = st.number_input("Age (age_latest)", min_value=0, max_value=120, value=65, step=1)
        los_ui   = st.number_input("Length of stay (los_total)", min_value=0, max_value=365, value=0, step=1)
    with c4:
        dx_ui    = st.number_input("Diagnosis count (dx_total)", min_value=0, max_value=200, value=0, step=1)
        proc_ui  = st.number_input("Procedure count (proc_total)", min_value=0, max_value=200, value=0, step=1)

    cc_col1, cc_col2 = st.columns([1,1])
    with cc_col1:
        chronic_count_ui = st.number_input("Chronic count (0–11)", min_value=0, max_value=11, value=0, step=1)
    with cc_col2:
        chronic_selected = st.multiselect(
            "Select chronic conditions present (multi-select)",
            CHRONIC_LIST,
            default=[]
        )

    chronic_flags = {c: (1 if c in chronic_selected else 0) for c in CHRONIC_LIST}

    def make_feature_row():
        sex_enc   = SEX_MAP[sex_ui]
        race_enc  = RACE_MAP[race_ui]
        state_enc = int(STATE_ABBR_TO_CODE[state_ui])  # '05' -> 5
        esrd_enc  = 1 if esrd_ui == "Yes" else 0
        ordered = [
            sex_enc, race_enc, state_enc, esrd_enc,
            int(age_ui), int(chronic_count_ui), int(los_ui),
            int(dx_ui), int(proc_ui),
        ] + [int(chronic_flags[c]) for c in CHRONIC_LIST]
        return np.array(ordered, dtype=float).reshape(1, -1)

    st.markdown("---")

    left, right = st.columns([1,1])
    with left:
        if st.button("🔮 Predict spend label", use_container_width=True):
            try:
                model = load_rf_model()
                X = make_feature_row()

                raw_pred = model.predict(X)[0]
                if hasattr(model, "classes_"):
                    label_by_class = {}
                    for c in model.classes_:
                        try:
                            label_by_class[c] = CLASS_TO_LABEL[int(c)]
                        except Exception:
                            label_by_class[c] = str(c)
                else:
                    label_by_class = {0: "Low", 1: "Medium", 2: "High"}

                try:
                    pretty_pred = CLASS_TO_LABEL[int(raw_pred)]
                except Exception:
                    pretty_pred = str(raw_pred)

                pcol, bcol = st.columns([3, 1], vertical_alignment="center")
                with pcol:
                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X)[0]
                        classes = list(getattr(model, "classes_", [0,1,2]))
                        disp_labels = [label_by_class.get(c, str(c)) for c in classes]
                        prob_df = pd.DataFrame({"Class": disp_labels, "Probability": probs})
                        figp = px.bar(prob_df, x="Class", y="Probability", range_y=[0,1])
                        figp.update_traces(
                            text=(prob_df["Probability"]*100).round(1).astype(str)+"%",
                            textposition="outside",
                            cliponaxis=False,
                        )
                        figp.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=320)
                        st.plotly_chart(figp, use_container_width=True, config={"displaylogo": False})
                    else:
                        st.info("Model does not expose class probabilities.")

                with bcol:
                    color = LABEL_COLOR.get(pretty_pred, "#64748b")
                    st.markdown(
                        f"""
                        <div style="
                            border-radius: 12px;
                            padding: 18px 14px;
                            background: {color}1A;
                            border: 1px solid {color};
                            text-align: center;
                        ">
                            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 6px;">
                                Predicted Spend Label
                            </div>
                            <div style="font-size: 1.6rem; font-weight: 700; color: {color};">
                                {pretty_pred}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            except FileNotFoundError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with right:
        X_preview = pd.DataFrame(
            [dict(zip(FEATURE_ORDER, list(make_feature_row().ravel().astype(int))))]
        )
        st.caption("Feature vector (model order)")
        st.dataframe(X_preview, use_container_width=True)

# =============================================================================
# BENEFICIARY DASHBOARD
# =============================================================================
if section == "dash" and page == "Beneficiary":
    st.title("📊 Beneficiary Dashboard")
    st.caption("Data source: Google Drive (remote). Falling back to local files if needed.")

    try:
        df = load_bf_remote()
    except Exception as e:
        path = find_file(BF_STEMS)
        if not path:
            st.error(f"Could not load bf_cost_ready from Drive or local. Error: {e}")
            st.stop()
        df = fast_read(path)

    chronic_cols = ("SP_ALZHDMTA","SP_CHF","SP_CHRNKIDN","SP_CNCR","SP_COPD","SP_DEPRESSN",
                    "SP_DIABETES","SP_ISCHMCHT","SP_OSTEOPRS","SP_RA_OA","SP_STRKETIA")

    core_missing = [c for c in ["Year","sex_label","race_label","chronic_count","DESYNPUF_ID"] if c not in df.columns]
    if core_missing:
        st.error(f"Missing required column(s): {core_missing}")
        st.stop()

    years = sorted(pd.to_numeric(df["Year"], errors="coerce").dropna().astype(int).unique().tolist())
    left_f, right_f = st.columns([1,3], vertical_alignment="center")
    with left_f:
        year_choice = st.selectbox("Year", options=["All"] + years, index=0)
    dff = df.copy() if year_choice == "All" else df[pd.to_numeric(df["Year"], errors="coerce").astype("Int64") == int(year_choice)].copy()
    with right_f:
        st.markdown(f"### Rows considered: **{len(dff):,}**")

    st.divider()

    def top2_conditions(sub: pd.DataFrame):
        counts = {c: int(pd.to_numeric(sub.get(c, 0), errors="coerce").fillna(0).astype(int).sum())
                  for c in chronic_cols if c in sub.columns}
        if not counts:
            return [("N/A", 0), ("N/A", 0)]
        return sorted(counts.items(), key=lambda x: x[1], reverse=True)[:2]

    def chronic_bucket_counts_unique_sum(sub: pd.DataFrame):
        if not {"DESYNPUF_ID", "chronic_count"}.issubset(sub.columns):
            return 0, 0, 0, 0
        cc_num = pd.to_numeric(sub["chronic_count"], errors="coerce").fillna(0)
        tmp = sub[["DESYNPUF_ID"]].copy()
        tmp["cc"] = cc_num
        per_id_sum = tmp.groupby("DESYNPUF_ID", as_index=True)["cc"].sum()
        eq0 = int((per_id_sum == 0).sum())
        eq1 = int((per_id_sum == 1).sum())
        gt1 = int((per_id_sum > 1).sum())
        uniq_ids = int(per_id_sum.shape[0])
        return eq0, eq1, gt1, uniq_ids

    (t1_name, t1_val), (t2_name, t2_val) = top2_conditions(dff)
    eq0_count, eq1_count, gt1_count, uniq_ids = chronic_bucket_counts_unique_sum(dff)

    k1, k2, k3, k4, k5 = st.columns([0.95, 0.95, 0.95, 0.95, 1.0])
    with k1:
        st.subheader("Top 1"); st.caption(t1_name); st.metric("Count", f"{t1_val:,}")
    with k2:
        st.subheader("Top 2"); st.caption(t2_name); st.metric("Count", f"{t2_val:,}")
    with k3:
        st.subheader("Chronic = 1"); st.caption("Unique B_ID (sum=1)")
        st.metric("Count", f"{eq1_count:,}")
    with k4:
        st.subheader("> 1 Chronic"); st.caption("Unique B_ID (sum>1)")
        st.metric("Count", f"{gt1_count:,}")
    with k5:
        st.subheader("Unique IDs"); st.caption("DESYNPUF_ID")
        st.metric("Count", f"{uniq_ids:,}")

    st.caption(f"Other IDs (sum=0): {eq0_count:,}")
    st.divider()

    r2l, r2r = st.columns([1, 1])
    with r2l:
        st.subheader("Gender distribution")
        g = dff["sex_label"].fillna("Unknown").astype(str).value_counts().reset_index()
        g.columns = ["sex_label","count"]
        fig_g = px.pie(g, names="sex_label", values="count", hole=0.60)
        fig_g.update_layout(height=320, margin=dict(l=0, r=0, t=8, b=0), showlegend=False)
        fig_g.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig_g, use_container_width=True, config={"displaylogo": False})

    with r2r:
        st.subheader("Top 3 chronic prevalence by Race")
        counts_sel = {c: int(pd.to_numeric(dff.get(c, 0), errors="coerce").fillna(0).astype(int).sum())
                      for c in chronic_cols if c in dff.columns}
        top3 = tuple([k for k,_ in sorted(counts_sel.items(), key=lambda x: x[1], reverse=True)[:3]]) if counts_sel else tuple()
        rprev = compute_top3_race_prev(dff, top3)
        if rprev.empty:
            st.info("No race-wise prevalence computed.")
        else:
            fig_r = px.bar(rprev, x="race_label", y="PrevalencePct", color="Condition", barmode="group")
            fig_r.update_layout(height=320, margin=dict(l=0, r=0, t=8, b=0))
            fig_r.update_yaxes(title="Prevalence (%)", range=[0,100])
            fig_r.update_xaxes(title="Race")
            st.plotly_chart(fig_r, use_container_width=True, config={"displaylogo": False})

    st.divider()

    st.subheader("Prevalence of chronic conditions over years")
    work = df if year_choice == "All" else dff
    prev_df = compute_prevalence_over_years(work, chronic_cols)
    if prev_df.empty:
        st.info("No prevalence data available.")
    else:
        figl = px.line(prev_df, x="Year", y="PrevalencePct", color="Condition", markers=True)
        figl.update_yaxes(range=[0,100], title="Prevalence (%)")
        figl.update_xaxes(title="Year")
        figl.update_layout(height=480, margin=dict(l=0, r=0, t=8, b=0))
        st.plotly_chart(figl, use_container_width=True, config={"displaylogo": False})

    st.divider()

    st.subheader("Cost distribution by Age Band (mean, cumulative by setting)")
    combine_groups = {
        "BENRES_total":  ["BENRES_IP", "BENRES_OP", "BENRES_CAR"],
        "MEDREIMB_total":["MEDREIMB_IP","MEDREIMB_OP","MEDREIMB_CAR"],
        "PPPYMT_total":  ["PPPYMT_IP", "PPPYMT_OP", "PPPYMT_CAR"],
    }
    needed = sorted({col for cols in combine_groups.values() for col in cols})
    missing_any = [c for c in needed if c not in dff.columns]
    if ("age_band_20y" not in dff.columns) or missing_any:
        msg = []
        if "age_band_20y" not in dff.columns: msg.append("age_band_20y")
        msg += missing_any
        st.info(f"Missing columns: {msg}")
    else:
        agg = compute_ageband_costs(dff, combine_groups)
        label_map = {
            "BENRES_total": "BENRES (IP+OP+CAR)",
            "MEDREIMB_total": "MEDREIMB (IP+OP+CAR)",
            "PPPYMT_total": "PPPYMT (IP+OP+CAR)",
        }
        agg["Metric"] = agg["Metric"].map(label_map)
        fig_cost_total = px.bar(agg, x="age_band_20y", y="Value", color="Metric", barmode="group")
        fig_cost_total.update_xaxes(title="Age band")
        fig_cost_total.update_yaxes(title="Mean value")
        fig_cost_total.update_layout(margin=dict(l=0, r=0, t=8, b=0), height=520)
        st.plotly_chart(fig_cost_total, use_container_width=True, config={"displaylogo": False})

# =============================================================================
# INPATIENT DASHBOARD
# =============================================================================
elif section == "dash" and page == "Inpatient":
    st.title("🏥 Inpatient Dashboard")
    st.caption("Data source: Google Drive (remote). Falling back to local files if needed.")

    try:
        ip = load_ip_remote()
    except Exception as e:
        ip_path = find_file(IP_STEMS)
        if not ip_path:
            st.error("Could not load inpatient data from Drive or local. " + str(e))
            st.stop()
        ip = fast_read(ip_path)

    def norm_col(c: str) -> str:
        return str(c).strip().lower().replace("\u00a0", " ").replace(".", "_").replace(" ", "_")
    ip.columns = [norm_col(c) for c in ip.columns]

    alias_map = {
        "desynpuf_id": ["desynpuf_id", "beneficiary_id", "id", "bene_id"],
        "los": ["los", "length_of_stay", "len_of_stay"],
        "dx_count": ["dx_count", "diagnosis_count", "diag_count"],
        "proc_count": ["proc_count", "procedure_count", "proc_cnt"],
        "clm_pmt_amt": ["clm_pmt_amt", "claim_payment_amount", "claim_amt", "payment_amount", "pmt_amt"],
    }
    def resolve(key: str) -> Optional[str]:
        for c in alias_map[key]:
            if c in ip.columns:
                return c
        return None

    col_desyn = resolve("desynpuf_id")
    col_los   = resolve("los")
    col_dx    = resolve("dx_count")
    col_proc  = resolve("proc_count")
    col_amt   = resolve("clm_pmt_amt")

    missing = [name for name, col in
               [("DESYNPUF_ID", col_desyn), ("LOS", col_los),
                ("dx_count", col_dx), ("proc_count", col_proc),
                ("CLM_PMT_AMT", col_amt)] if col is None]
    if missing:
        st.error(f"Missing column(s) for Inpatient dashboard: {missing}")
        st.stop()

    ip[col_los]  = pd.to_numeric(ip[col_los], errors="coerce")
    ip[col_dx]   = pd.to_numeric(ip[col_dx], errors="coerce")
    ip[col_proc] = pd.to_numeric(ip[col_proc], errors="coerce")
    ip[col_amt]  = pd.to_numeric(ip[col_amt], errors="coerce")

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.subheader("Unique IDs")
        st.metric("Count", f"{ip[col_desyn].nunique():,}")
    with c2:
        st.subheader("Avg. LOS")
        st.metric("Days", f"{ip[col_los].mean(skipna=True):.2f}")
    with c3:
        st.subheader("Avg. Dx Count")
        st.metric("Per admission", f"{ip[col_dx].mean(skipna=True):.2f}")
    with c4:
        st.subheader("Avg. Proc Count")
        st.metric("Per admission", f"{ip[col_proc].mean(skipna=True):.2f}")

    st.divider()

    st.subheader("Top IDs by total payment (sum of CLM_PMT_AMT)")
    tmp = (
        ip.assign(has_claim=ip[col_amt].notna().astype(int))
          .groupby(col_desyn, as_index=False)
          .agg(Count_of_claims=("has_claim","sum"),
               Total_spent=(col_amt,"sum"))
          .rename(columns={col_desyn: "DESYNPUF_ID"})
          .sort_values("Total_spent", ascending=False)
    )
    st.dataframe(tmp.head(10), use_container_width=True)
    st.download_button(
        "Download full table (CSV)",
        data=tmp.to_csv(index=False).encode("utf-8"),
        file_name="inpatient_spend_summary.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Distribution of Length of Stay (LOS)")
    los_series = pd.to_numeric(ip[col_los], errors="coerce").dropna()
    if los_series.empty:
        st.info("No LOS data available.")
    else:
        STEP = 10
        max_los = int(np.ceil(los_series.max()))
        edges = np.arange(0, max_los + STEP, STEP)
        if edges[-1] < max_los:
            edges = np.append(edges, max_los)
        labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges) - 1)]
        cats = pd.cut(los_series, bins=edges, right=False, include_lowest=True, labels=labels)
        counts = cats.value_counts().sort_index().reset_index()
        counts.columns = ["LOS_bin", "count"]

        def humanize(n: int) -> str:
            if n >= 1_000_000: return f"{n/1_000_000:.1f}M+"
            if n >= 1_000:     return f"{n/1_000:.1f}K+"
            return f"{n}"

        fig = px.bar(counts, x="LOS_bin", y="count")
        fig.update_traces(text=[humanize(n) for n in counts["count"]], textposition="outside", cliponaxis=False)
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=8, b=0),
                          xaxis_title="Length of Stay (days, binned)", yaxis_title="Count of admissions")
        fig.update_yaxes(tickformat="~s")
        fig.update_xaxes(type="category", tickangle=-20)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# =============================================================================
# OUTPATIENT DASHBOARD
# =============================================================================
elif section == "dash" and page == "Outpatient":
    st.title("🏥 Outpatient Dashboard")
    st.caption("Data source: Google Drive (remote). Falling back to local files if needed.")

    try:
        op = load_op_remote()
    except Exception as e:
        op_path = find_file(OP_STEMS)
        if not op_path:
            st.error("Could not load outpatient data from Drive or local. " + str(e))
            st.stop()
        op = fast_read(op_path)

    def norm_col(c: str) -> str:
        return str(c).strip().lower().replace("\u00a0", " ").replace(".", "_").replace(" ", "_")
    op.columns = [norm_col(c) for c in op.columns]

    alias_map = {
        "desynpuf_id": ["desynpuf_id", "beneficiary_id", "id", "bene_id"],
        "dx_count": ["dx_count", "diagnosis_count", "diag_count"],
        "proc_count": ["proc_count", "procedure_count", "proc_cnt"],
        "dur": ["dur", "duration", "visit_duration", "encounter_duration"],
        "clm_pmt_amt": ["clm_pmt_amt", "claim_payment_amount", "claim_amt", "payment_amount", "pmt_amt"],
        "icd9_dgns_cd_1": ["icd9_dgns_cd_1", "principal_diagnosis", "diag_principal", "icd9_principal"]
    }
    def resolve(key: str) -> Optional[str]:
        for c in alias_map[key]:
            if c in op.columns:
                return c
        return None

    col_desyn = resolve("desynpuf_id")
    col_dx    = resolve("dx_count")
    col_proc  = resolve("proc_count")
    col_dur   = resolve("dur")
    col_amt   = resolve("clm_pmt_amt")
    col_dx1   = resolve("icd9_dgns_cd_1")

    missing = [name for name, col in
               [("DESYNPUF_ID", col_desyn), ("dx_count", col_dx),
                ("proc_count", col_proc), ("DUR", col_dur),
                ("CLM_PMT_AMT", col_amt), ("ICD9_DGNS_CD_1", col_dx1)]
               if col is None]
    if missing:
        st.error(f"Missing column(s) for Outpatient dashboard: {missing}")
        st.stop()

    op[col_dx]   = pd.to_numeric(op[col_dx], errors="coerce")
    op[col_proc] = pd.to_numeric(op[col_proc], errors="coerce")
    op[col_dur]  = pd.to_numeric(op[col_dur], errors="coerce")
    op[col_amt]  = pd.to_numeric(op[col_amt], errors="coerce")

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        st.subheader("Unique IDs")
        st.metric("Count", f"{op[col_desyn].nunique():,}")
    with c2:
        st.subheader("Avg. Dx Count")
        st.metric("Per visit", f"{op[col_dx].mean(skipna=True):.2f}")
    with c3:
        st.subheader("Avg. Proc Count")
        st.metric("Per visit", f"{op[col_proc].mean(skipna=True):.2f}")
    with c4:
        st.subheader("Avg. Duration")
        st.metric("DUR", f"{op[col_dur].mean(skipna=True):.2f}")

    st.divider()

    st.subheader("Top IDs by total payment (sum of CLM_PMT_AMT)")
    tmp = (
        op.assign(has_claim=op[col_amt].notna().astype(int))
          .groupby(col_desyn, as_index=False)
          .agg(Count_of_claims=("has_claim","sum"),
               Total_spent=(col_amt,"sum"))
          .rename(columns={col_desyn: "DESYNPUF_ID"})
          .sort_values("Total_spent", ascending=False)
    )
    st.dataframe(tmp.head(10), use_container_width=True)
    st.download_button(
        "Download full table (CSV)",
        data=tmp.to_csv(index=False).encode("utf-8"),
        file_name="outpatient_spend_summary.csv",
        mime="text/csv"
    )

    st.divider()

    st.subheader("Top 20 principal diagnoses (by frequency)")
    diag_series = op[col_dx1].astype(str).str.strip().replace({"": pd.NA}).dropna()
    if diag_series.empty:
        st.info("No principal diagnosis codes available.")
    else:
        topn = (
            diag_series.value_counts()
            .head(20)
            .rename_axis("Diagnosis")
            .reset_index(name="Frequency")
        )
        topn = topn.iloc[::-1].reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(topn["Diagnosis"], topn["Frequency"])
        ax.set_title("Top 20 Most Common Diagnoses", pad=10)
        ax.set_xlabel("Frequency")
        ax.set_ylabel("ICD-9 Code")
        ax.tick_params(axis='y', labelsize=10)
        ax.tick_params(axis='x', labelsize=10)
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)

# =============================================================================
# MACHINE LEARNING: Cost Prediction (IMPLEMENTED)
# =============================================================================
elif section == "ml" and page == "Cost Prediction":
    render_cost_prediction()

# =============================================================================
# MACHINE LEARNING: Readmission Rate (future scope)
# =============================================================================
elif section == "ml" and page == "Readmission Rate":
    st.title("This is a Future Scope")
    st.markdown("""
**Our Project aim is to predicts whether a patient is likely to be readmitted to the hospital within 15/30 days after being discharged.**
- Predictive models can identify high-risk patients before discharge.
- Personalized discharge planning and post-discharge monitoring can lead to better recovery, especially for chronic conditions like diabetes, heart failure, and COPD.
- Readmissions are costly to hospitals (especially under Medicare penalties). Predicting them helps optimize resource allocation, reducing operational and insurance costs.
""")