"""Streamlit dashboard: `streamlit run app.py`."""

from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from nids import __version__
from nids.config import (AFFILIATION, DEVELOPER, FEATURE_NAMES, ORGANIZATION,
                         PRODUCT_NAME, PROGRAM, footer_text)
from nids.predict import NIDSPredictor
from nids.train import run_training

st.set_page_config(page_title=f"{PRODUCT_NAME}", page_icon="🛡️", layout="wide")

MAX_UPLOAD_ROWS = 50_000
# (name, kind, min, max, default, step) - all numeric args share one type.
LIVE_FEATURES = [
    ("duration", "int", 0, 10000, 0, 1),
    ("src_bytes", "int", 0, 100000, 500, 10),
    ("dst_bytes", "int", 0, 100000, 1000, 10),
    ("count", "int", 0, 500, 10, 1),
    ("srv_count", "int", 0, 500, 10, 1),
    ("serror_rate", "float", 0.0, 1.0, 0.0, 0.01),
    ("same_srv_rate", "float", 0.0, 1.0, 1.0, 0.01),
    ("dst_host_count", "int", 0, 255, 100, 1),
    ("dst_host_same_srv_rate", "float", 0.0, 1.0, 0.8, 0.01),
    ("dst_host_serror_rate", "float", 0.0, 1.0, 0.0, 0.01),
]


@st.cache_resource(show_spinner=False)
def load_predictor():
    """Cached model load. Returns (predictor, error_message)."""
    try:
        return NIDSPredictor(), None
    except FileNotFoundError:
        return None, None
    except ValueError as exc:  # integrity failure
        return None, str(exc)


def footer() -> None:
    st.divider()
    st.caption(footer_text(__version__))


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


# ---------------------------------------------------------------- header ---
st.title(f"🛡️ {PRODUCT_NAME} v{__version__}")
st.subheader("Machine Learning Network Intrusion Detection System | IBM Cybersecurity")

predictor, load_error = load_predictor()

if load_error:
    st.error(f"Model rejected: {load_error}")
    footer()
    st.stop()

if predictor is None:
    st.warning("No trained model found. Run `python ml_nids.py` first, "
               "or train a quick demo model below.")
    if st.button("Train demo model (synthetic data, ~30 s)", type="primary"):
        with st.spinner("Training..."):
            run_training(n_samples=8000, run_cv=False, n_estimators=100,
                         verbose=False)
        st.cache_resource.clear()
        st.rerun()
    footer()
    st.stop()

m = predictor.metrics
hold = m.get("holdout", {})
if m.get("dataset", {}).get("synthetic"):
    st.warning("This model was trained on SYNTHETIC data. Figures below are a "
               "demo, not real-world detection performance. Add NSL-KDD to "
               "`data/` and retrain for real results.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", pct(hold.get("accuracy")))
c2.metric("Precision (weighted)", pct(hold.get("precision_weighted")))
c3.metric("Recall (weighted)", pct(hold.get("recall_weighted")))
c4.metric("F1-Score (weighted)", pct(hold.get("f1_weighted")))
st.divider()

tab1, tab2, tab3 = st.tabs(
    ["🔍 Live Detection", "📊 Analytics", "⚙️ Enterprise Features"])

# ------------------------------------------------------------- live tab ----
with tab1:
    st.markdown("### Real-time Network Flow Classifier")
    st.write("Set the 10 key features; the other 31 NSL-KDD features use safe defaults.")
    cats = predictor.categories()

    left, right = st.columns(2)
    flow = {}
    for i, (name, kind, lo, hi, default, step) in enumerate(LIVE_FEATURES):
        target = left if i < 5 else right
        flow[name] = target.slider(name, lo, hi, default, step, key=f"live_{name}")

    with st.expander("Categorical features"):
        k1, k2, k3 = st.columns(3)
        for col, widget in zip(("protocol_type", "service", "flag"), (k1, k2, k3)):
            options = cats[col]
            idx = options.index("tcp") if col == "protocol_type" and "tcp" in options else 0
            flow[col] = widget.selectbox(col, options, index=idx, key=f"live_{col}")

    if st.button("🚀 Analyze Traffic", type="primary"):
        result = predictor.predict_one(flow)
        label, conf = result["prediction"], result["confidence"]
        if not result["is_attack"]:
            st.success(f"✅ Traffic classified: {label} "
                       f"(confidence {conf * 100:.1f}%) - no threat detected")
        else:
            st.error(f"🚨 THREAT DETECTED: {label} "
                     f"(confidence {conf * 100:.1f}%) - threat level: "
                     f"{result['threat_level']}")
        probs = pd.DataFrame({"class": list(result["probabilities"]),
                              "probability": list(result["probabilities"].values())})
        st.plotly_chart(px.bar(probs, x="class", y="probability",
                               range_y=[0, 1], title="Class probabilities"),
                        use_container_width=True)
        with st.expander("SIEM event (JSON)"):
            st.json(predictor.to_siem_event(result, flow))

    st.markdown("---")
    st.markdown("### Batch analysis (CSV upload)")
    st.caption("CSV with a header row using NSL-KDD feature names. Missing "
               "columns are filled with defaults.")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is not None:
        try:
            raw = pd.read_csv(upload)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
            st.error(f"Could not read CSV: {exc}")
            raw = None
        if raw is not None:
            if len(raw) > MAX_UPLOAD_ROWS:
                st.warning(f"Only the first {MAX_UPLOAD_ROWS:,} rows are scored.")
                raw = raw.head(MAX_UPLOAD_ROWS)
            missing = predictor.missing_columns(raw)
            if len(missing) == len(FEATURE_NAMES):
                st.error("No recognised NSL-KDD columns found. Expected: "
                         + ", ".join(FEATURE_NAMES[:6]) + ", ...")
            else:
                if missing:
                    st.info(f"{len(missing)} columns missing; defaults used.")
                res = predictor.predict(raw)
                out = pd.concat([raw.reset_index(drop=True),
                                 res.reset_index(drop=True)], axis=1)
                counts = res["prediction"].value_counts().reset_index()
                counts.columns = ["class", "flows"]
                a, b = st.columns([1, 2])
                a.metric("Flows scored", f"{len(res):,}")
                a.metric("Flagged as attack",
                         f"{int((res['prediction'] != 'Normal').sum()):,}")
                b.plotly_chart(px.bar(counts, x="class", y="flows",
                                      title="Predicted class distribution"),
                               use_container_width=True)
                st.dataframe(out.head(500), use_container_width=True)
                st.download_button("⬇️ Download results (CSV)",
                                   out.to_csv(index=False).encode("utf-8"),
                                   "nids_predictions.csv", "text/csv")

# ---------------------------------------------------------- analytics tab --
with tab2:
    st.markdown("### Model Performance (hold-out set)")
    labels = m.get("labels", predictor.classes)
    cm = hold.get("confusion_matrix")
    left, right = st.columns(2)
    if cm:
        fig = go.Figure(go.Heatmap(z=cm, x=labels, y=labels, colorscale="Blues",
                                   text=cm, texttemplate="%{text}"))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted",
                          yaxis_title="True", yaxis_autorange="reversed")
        left.plotly_chart(fig, use_container_width=True)
    imp = pd.DataFrame(m.get("feature_importance", [])).head(15)
    if not imp.empty:
        right.plotly_chart(
            px.bar(imp.iloc[::-1], x="importance", y="feature", orientation="h",
                   title="Top 15 feature importances"),
            use_container_width=True)

    sec = hold.get("security") or {}
    s1, s2, s3 = st.columns(3)
    s1.metric("Detection rate (attack vs normal)", pct(sec.get("detection_rate")))
    s2.metric("False-alarm rate", pct(sec.get("false_alarm_rate")))
    s3.metric("Macro F1", pct(hold.get("f1_macro")))

    st.markdown("#### Per-class metrics")
    pc = pd.DataFrame(hold.get("per_class", {})).T
    if not pc.empty:
        st.dataframe(pc.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                      "f1": "{:.3f}", "support": "{:.0f}"}),
                     use_container_width=True)
    cv = m.get("cross_validation")
    if cv:
        st.caption(f"5-fold CV accuracy: {pct(cv['accuracy_mean'])} "
                   f"(± {cv['accuracy_std'] * 100:.2f}%) | macro F1: "
                   f"{pct(cv['f1_macro_mean'])}")
    ext = m.get("external_test")
    if ext:
        st.caption(f"Official KDDTest+ accuracy: {pct(ext['accuracy'])} | "
                   f"macro F1: {pct(ext['f1_macro'])}")

# ------------------------------------------------------- enterprise tab ----
with tab3:
    st.markdown("### Enterprise Deployment")
    bench = m.get("benchmark", {})
    e1, e2, e3 = st.columns(3)
    e1.metric("Latency / flow (mean)",
              f"{bench.get('single_flow_latency_ms_mean', float('nan')):.2f} ms")
    e2.metric("Batch throughput",
              f"{bench.get('batch_throughput_flows_per_sec', float('nan')):,.0f} flows/s")
    e3.metric("Model size", f"{m.get('model_size_mb', 'n/a')} MB")
    st.caption(f"Measured on {bench.get('cpu_threads', 1)} CPU thread at "
               f"training time; your hardware will differ.")

    st.markdown("**Python integration**")
    st.code("""from nids.predict import NIDSPredictor
p = NIDSPredictor()                       # verifies model SHA-256
result = p.predict_one({"src_bytes": 500, "count": 10, "serror_rate": 0.0})
event = p.to_siem_event(result)           # JSON for Wazuh / Splunk / Elastic
""", language="python")

    st.markdown("**REST API (FastAPI)**")
    st.code("""uvicorn api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/predict \\
     -H "X-API-Key: $NIDS_API_KEY" -H "Content-Type: application/json" \\
     -d '{"features": {"src_bytes": 500, "count": 10}}'""", language="bash")

    st.markdown("**Example SIEM event**")
    st.json(predictor.to_siem_event(predictor.predict_one({})))

    st.markdown("**Model card**")
    st.json({"trained_at": m.get("trained_at"), "dataset": m.get("dataset"),
             "environment": m.get("environment"),
             "sha256": m.get("model_sha256")})
    st.markdown(f"Developed by **{DEVELOPER}** · {AFFILIATION} · {ORGANIZATION} · {PROGRAM}")

footer()
