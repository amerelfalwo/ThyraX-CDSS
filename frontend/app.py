"""
ThyraX CDSS — Professional Streamlit Frontend
================================================
A modern, production-ready clinical decision support interface for physicians.
"""
import streamlit as st
import requests
import pandas as pd
import json
import os

API_AUTH_HEADERS = {
    "X-AI-Service-Key": os.getenv("INTERNAL_SERVICE_KEY", "thyrax-internal-sk-2026-secure")
}

# ═══════════════════════════════════════════════════════════════
# Page Config & Styling
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="ThyraX CDSS",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

# ── Custom CSS ──
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f766e 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.3);
    }
    .main-header h1 {
        color: #f0fdfa;
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: #99f6e4;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        font-weight: 400;
    }

    .metric-card {
        background: linear-gradient(135deg, #f0fdfa, #ecfdf5);
        border: 1px solid #99f6e4;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(15, 118, 110, 0.08);
    }
    .metric-card h3 {
        color: #0f766e;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin: 0 0 0.3rem 0;
    }
    .metric-card .value {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
    }

    .risk-badge-high {
        background: linear-gradient(135deg, #fef2f2, #fee2e2);
        border: 1px solid #fca5a5;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #991b1b;
        font-weight: 600;
    }
    .risk-badge-low {
        background: linear-gradient(135deg, #f0fdf4, #dcfce7);
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        color: #166534;
        font-weight: 600;
    }

    .disclaimer {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.78rem;
        color: #92400e;
        font-style: italic;
        margin-top: 0.8rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stNumberInput label {
        color: #94a3b8 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Header
# ═══════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🩺 ThyraX: Clinical Decision Support System</h1>
    <p>AI-Powered Thyroid Cancer Screening &amp; Longitudinal Patient Monitoring</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Sidebar — Patient Management
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 🏥 Patient Management")
    st.divider()

    mode = st.radio("Action", ["Select Existing Patient", "Create New Patient"], label_visibility="collapsed")

    if mode == "Select Existing Patient":
        pid = st.number_input("Patient ID", min_value=1, step=1, value=1, key="sidebar_pid")
        if st.button("🔍 Load Patient", use_container_width=True):
            st.session_state.patient_id = pid
            st.toast(f"✅ Active patient set to ID **{pid}**")
    else:
        with st.form("create_patient_form"):
            name = st.text_input("Patient Name")
            age = st.number_input("Age", min_value=0, max_value=120, value=45)
            gender = st.selectbox("Gender", ["M", "F", "Other"])
            submitted = st.form_submit_button("➕ Create Patient", use_container_width=True)
            if submitted and name:
                try:
                    res = requests.post(f"{API_BASE}/patient/create", json={"name": name, "age": age, "gender": gender}, headers=API_AUTH_HEADERS, timeout=10)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.patient_id = data["id"]
                        st.success(f"Patient **{name}** created with ID **{data['id']}**")
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    st.divider()
    active_pid = st.session_state.get("patient_id", None)
    if active_pid:
        st.markdown(f"**Active Patient:** `ID {active_pid}`")
    else:
        st.warning("No patient selected.")

    # Initialize session state for OCR results
    for key in ["ocr_TSH", "ocr_T3", "ocr_TT4", "ocr_FTI", "ocr_T4U"]:
        if key not in st.session_state:
            st.session_state[key] = 0.0
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


# ═══════════════════════════════════════════════════════════════
# Main Tabs
# ═══════════════════════════════════════════════════════════════

tab_dashboard, tab_clinical, tab_ultrasound, tab_chat = st.tabs([
    "📊 Dashboard",
    "🧪 Clinical & OCR",
    "🖼️ Ultrasound AI",
    "💬 AI Assistant",
])


# ───────────────────────────────────────────────────────────────
# Tab 1: Patient Dashboard
# ───────────────────────────────────────────────────────────────

with tab_dashboard:
    pid = st.session_state.get("patient_id")
    if not pid:
        st.info("👈 Please select or create a patient from the sidebar to view their dashboard.")
    else:
        try:
            res = requests.get(f"{API_BASE}/patient/{pid}/dashboard", headers=API_AUTH_HEADERS, timeout=10)
            if res.status_code == 200:
                dash = res.json()
                patient = dash["patient"]
                visits = dash["visits"]
                imaging = dash["imaging"]

                # ── Demographics Row ──
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.markdown(f"""<div class="metric-card"><h3>Patient</h3><p class="value">{patient['name']}</p></div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""<div class="metric-card"><h3>Age</h3><p class="value">{patient['age']}</p></div>""", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"""<div class="metric-card"><h3>Gender</h3><p class="value">{patient.get('gender', 'N/A')}</p></div>""", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"""<div class="metric-card"><h3>Total Visits</h3><p class="value">{len(visits)}</p></div>""", unsafe_allow_html=True)

                st.divider()

                # ── Latest Labs ──
                if visits:
                    latest = visits[-1]
                    l1, l2, l3 = st.columns(3)
                    l1.metric("Latest TSH", f"{latest.get('tsh', 'N/A')} µIU/mL")
                    l2.metric("Latest T3", f"{latest.get('t3', 'N/A')} ng/mL")
                    l3.metric("Latest T4", f"{latest.get('t4', 'N/A')} µg/dL")

                    # ── Longitudinal TSH Trend ──
                    st.markdown("#### 📈 TSH Longitudinal Trend")
                    tsh_data = []
                    for v in visits:
                        if v.get("tsh") is not None and v.get("visit_date"):
                            tsh_data.append({"Date": v["visit_date"][:10], "TSH": v["tsh"]})

                    if tsh_data:
                        df = pd.DataFrame(tsh_data)
                        df["Date"] = pd.to_datetime(df["Date"])
                        df = df.sort_values("Date")
                        st.line_chart(df.set_index("Date")["TSH"], use_container_width=True)
                    else:
                        st.caption("Not enough data points for trend visualization.")

                    # ── Latest Recommendation ──
                    if dash.get("latest_clinical_recommendation"):
                        st.markdown("#### 🩺 Latest Clinical Recommendation")
                        st.info(dash["latest_clinical_recommendation"])
                else:
                    st.info("No visit records found for this patient.")

                # ── Imaging History ──
                if imaging:
                    st.markdown("#### 🖼️ Imaging History")
                    img_df = pd.DataFrame(imaging)
                    st.dataframe(img_df[["id", "classification_label", "confidence", "tirads_stage", "processed_at"]], use_container_width=True)

            elif res.status_code == 404:
                st.warning(f"Patient ID {pid} not found. Please create the patient first.")
            else:
                st.error(f"Backend error: {res.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to the backend. Ensure the FastAPI server is running on `localhost:8000`.")
        except Exception as e:
            st.error(f"Unexpected error: {e}")


# ───────────────────────────────────────────────────────────────
# Tab 2: Clinical Assessment & AI OCR
# ───────────────────────────────────────────────────────────────

with tab_clinical:
    col_ocr, col_form = st.columns(2, gap="large")

    # ── Column 1: Premium AI OCR ──
    with col_ocr:
        st.markdown("#### 🤖 Premium AI Lab Extraction")
        st.caption("Upload a lab report image and let Gemini AI extract the values automatically.")

        uploaded_lab = st.file_uploader("Upload Lab Report", type=["png", "jpg", "jpeg", "webp"], key="lab_uploader")

        if uploaded_lab and st.button("⚡ Extract with AI", use_container_width=True, type="primary"):
            with st.spinner("🔬 Gemini is analyzing the lab report..."):
                try:
                    files = {"file": (uploaded_lab.name, uploaded_lab.getvalue(), uploaded_lab.type)}
                    res = requests.post(f"{API_BASE}/labs/extract", files=files, headers=API_AUTH_HEADERS, timeout=30)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.ocr_TSH = data.get("TSH") or 0.0
                        st.session_state.ocr_T3 = data.get("T3") or 0.0
                        st.session_state.ocr_TT4 = data.get("TT4") or 0.0
                        st.session_state.ocr_FTI = data.get("FTI") or 0.0
                        st.session_state.ocr_T4U = data.get("T4U") or 0.0

                        st.success("✅ Values extracted successfully!")
                        r1, r2, r3, r4, r5 = st.columns(5)
                        r1.metric("TSH", f"{st.session_state.ocr_TSH}")
                        r2.metric("T3", f"{st.session_state.ocr_T3}")
                        r3.metric("TT4", f"{st.session_state.ocr_TT4}")
                        r4.metric("FTI", f"{st.session_state.ocr_FTI}")
                        r5.metric("T4U", f"{st.session_state.ocr_T4U}")

                        if data.get("test_date"):
                            st.caption(f"📅 Test Date: {data['test_date']}")
                        if data.get("clinical_notes"):
                            st.caption(f"📝 Notes: {data['clinical_notes']}")
                    else:
                        st.error(f"Extraction failed: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to the backend.")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ── Column 2: Manual Entry & Submit ──
    with col_form:
        st.markdown("#### 📋 Clinical Assessment Form")
        st.caption("Values auto-fill from OCR if available, or enter manually.")

        with st.form("clinical_form"):
            pid = st.session_state.get("patient_id", 1)
            TSH = st.number_input("TSH (µIU/mL)", min_value=0.0, value=float(st.session_state.ocr_TSH), format="%.3f")
            T3 = st.number_input("T3 (ng/mL)", min_value=0.0, value=float(st.session_state.ocr_T3), format="%.3f")
            TT4 = st.number_input("Total T4 (µg/dL)", min_value=0.0, value=float(st.session_state.ocr_TT4), format="%.3f")
            FTI = st.number_input("Free Thyroxine Index", min_value=0.0, value=float(st.session_state.ocr_FTI), format="%.3f")
            T4U = st.number_input("T4 Uptake", min_value=0.0, value=float(st.session_state.ocr_T4U), format="%.3f")
            nodule = st.checkbox("🔴 Nodule Present on Physical Exam")

            submitted = st.form_submit_button("🩺 Run Clinical Assessment", use_container_width=True, type="primary")

            if submitted:
                payload = {
                    "patient_id": pid,
                    "TSH": TSH,
                    "T3": T3,
                    "TT4": TT4,
                    "FTI": FTI,
                    "T4U": T4U,
                    "nodule_present": nodule,
                }
                try:
                    res = requests.post(f"{API_BASE}/clinical/assess", json=payload, headers=API_AUTH_HEADERS, timeout=15)
                    if res.status_code == 200:
                        result = res.json()
                        st.divider()

                        # Display functional status
                        st.markdown(f"**Functional Status:** `{result['functional_status'].upper()}`")
                        st.markdown(f"**Risk Level:** `{result['risk_level'].upper()}`")

                        # Route-based coloring
                        next_step = result.get("next_step", "")
                        if next_step == "routine_followup":
                            st.success(f"✅ {result['clinical_recommendation']}")
                        elif next_step == "upload_ultrasound":
                            st.error(f"🚨 {result['clinical_recommendation']}")
                        else:
                            st.warning(f"⚠️ {result['clinical_recommendation']}")

                        # Probabilities
                        with st.expander("🔬 Model Probabilities"):
                            for label, prob in result.get("probabilities", {}).items():
                                st.progress(prob, text=f"{label}: {prob:.2%}")

                        st.markdown(f'<div class="disclaimer">{result.get("disclaimer", "")}</div>', unsafe_allow_html=True)
                    else:
                        st.error(f"Assessment failed: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to the backend.")
                except Exception as e:
                    st.error(f"Error: {e}")


# ───────────────────────────────────────────────────────────────
# Tab 3: Ultrasound AI Pipeline
# ───────────────────────────────────────────────────────────────

with tab_ultrasound:
    st.markdown("#### 🖼️ Thyroid Ultrasound AI Analysis")
    st.caption("Upload a thyroid ultrasound image. The AI Gatekeeper will validate it before running the segmentation & classification pipeline.")

    uploaded_us = st.file_uploader("Upload Ultrasound Image", type=["png", "jpg", "jpeg", "webp"], key="us_uploader")

    if uploaded_us:
        col_img, col_result = st.columns([1, 1], gap="large")

        with col_img:
            st.image(uploaded_us, caption="Uploaded Image", use_container_width=True)

        with col_result:
            if st.button("🔬 Analyze Ultrasound", use_container_width=True, type="primary"):
                # Step 1: Gatekeeper Validation
                with st.spinner("🛡️ AI Gatekeeper validating image..."):
                    try:
                        files = {"file": (uploaded_us.name, uploaded_us.getvalue(), uploaded_us.type)}
                        val_res = requests.post(f"{API_BASE}/image/validate", files=files, headers=API_AUTH_HEADERS, timeout=30)

                        if val_res.status_code == 200:
                            val_data = val_res.json()
                            if not val_data.get("is_ultrasound", False):
                                st.error("🚫 **Invalid Image Type.** The AI Gatekeeper determined this is NOT a valid medical ultrasound scan. Please upload a proper thyroid ultrasound image.")
                                st.stop()
                            else:
                                st.success("✅ Gatekeeper: Valid ultrasound confirmed.")
                        else:
                            st.error(f"Validation error: {val_res.text}")
                            st.stop()
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Cannot connect to the backend.")
                        st.stop()
                    except Exception as e:
                        st.error(f"Validation error: {e}")
                        st.stop()

                # Step 2: Prediction Pipeline
                with st.spinner("🧠 Running Segmentation & Classification Pipeline..."):
                    try:
                        files = {"file": (uploaded_us.name, uploaded_us.getvalue(), uploaded_us.type)}
                        pred_res = requests.post(f"{API_BASE}/image/predict", files=files, headers=API_AUTH_HEADERS, timeout=60)

                        if pred_res.status_code == 200:
                            pred_data = pred_res.json()

                            st.markdown("---")
                            st.markdown("### 📊 AI Analysis Results")

                            m1, m2 = st.columns(2)
                            m1.metric("Classification", pred_data.get("classification_label", "N/A"))
                            m2.metric("Confidence", f"{pred_data.get('confidence', 0):.1%}")

                            if pred_data.get("tirads_stage"):
                                st.info(f"🏥 **TI-RADS Stage:** {pred_data['tirads_stage']}")

                            st.markdown(f'<div class="disclaimer">⚕️ AI-generated analysis is a clinical decision support tool only. Final diagnosis must be made by a qualified physician.</div>', unsafe_allow_html=True)
                        else:
                            st.error(f"Prediction error: {pred_res.text}")
                    except requests.exceptions.ConnectionError:
                        st.error("⚠️ Cannot connect to the backend.")
                    except Exception as e:
                        st.error(f"Prediction error: {e}")


# ───────────────────────────────────────────────────────────────
# Tab 4: AI Medical Assistant
# ───────────────────────────────────────────────────────────────

with tab_chat:
    st.markdown("#### 💬 ThyraX AI Medical Assistant")
    st.caption("Ask clinical questions about thyroid conditions, patient history, or medical guidelines.")

    # Display chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask ThyraX a medical question...")

    if user_query:
        # Show user message
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Call the agent
        with st.chat_message("assistant"):
            with st.spinner("🧠 ThyraX is thinking..."):
                try:
                    payload = {
                        "query": user_query,
                        "patient_id": st.session_state.get("patient_id"),
                    }
                    res = requests.post(f"{API_BASE}/agent/chat", json=payload, headers=API_AUTH_HEADERS, timeout=60)

                    if res.status_code == 200:
                        data = res.json()
                        response_text = data.get("response", "No response generated.")
                        st.markdown(response_text)

                        # Tools used
                        tools = data.get("tools_used", [])
                        if tools:
                            st.caption(f"🔧 Tools used: {', '.join(tools)}")

                        # Disclaimer
                        st.markdown(f'<div class="disclaimer">{data.get("disclaimer", "⚕️ ThyraX is an AI assistant. Final clinical decisions must be made by a certified physician.")}</div>', unsafe_allow_html=True)

                        st.session_state.chat_history.append({"role": "assistant", "content": response_text})
                    else:
                        error_msg = f"Agent error: {res.text}"
                        st.error(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Cannot connect to the backend. Ensure the FastAPI server is running.")
                except Exception as e:
                    st.error(f"Error: {e}")
