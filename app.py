import traceback
import streamlit as st
from pipeline import run_pipeline

# ==========================================================
# Page Configuration
# ==========================================================

st.set_page_config(
    page_title="Medical Hallucination Detector",
    page_icon="🩺",
    layout="wide",
)

# ==========================================================
# Header
# ==========================================================

st.title("🩺 Medical Hallucination Detector")

st.markdown(
    """
Verify AI-generated medical responses using trusted
**PubMed evidence** and Natural Language Inference (NLI).
"""
)

st.markdown("---")

# ==========================================================
# Ask Medical Question
# ==========================================================

st.subheader("🩺 Ask a Medical Question")

question = st.text_area(
    "Medical Question",
    placeholder="Example: What is the best medicine for fever in children?",
    height=120,
)

col1, col2, col3 = st.columns([3, 2, 3])

with col2:
    verify = st.button(
        "🔍 Verify Response",
        use_container_width=True,
        type="primary",
    )

st.markdown("---")

# ==========================================================
# Run Pipeline
# ==========================================================

if verify:
    if not question.strip():
        st.warning("⚠️ Please enter a medical question.")
        st.stop()

    with st.spinner("🔍 Generating response and verifying medical facts..."):
        try:
            st.session_state["result"] = run_pipeline(question)
        except Exception:
            st.error("❌ Verification Failed")
            st.code(traceback.format_exc(), language="python")
            st.stop()

# ==========================================================
# Display AI Response & Results
# ==========================================================

if "result" in st.session_state:
    result = st.session_state["result"]

    st.success("✅ Verification Completed")

    st.markdown("## 🧠 AI Generated Response")
    st.info(result["answer"])

    st.markdown("---")
    st.markdown("## 📋 Claim Verification")

    for idx, claim in enumerate(result["claims"], start=1):
        verdict = claim["verdict"]
        confidence = claim["confidence"]
        evidence = claim["evidence"]

        if verdict == "Supported":
            icon = "✅"
            color = "green"
        elif verdict == "Contradicted":
            icon = "❌"
            color = "red"
        else:
            icon = "⚠️"
            color = "orange"

        with st.expander(f"{icon} Claim {idx}"):
            st.markdown("### Claim")
            st.write(claim["claim"])

            st.markdown("### Verdict")
            if verdict == "Supported":
                st.success(verdict)
            elif verdict == "Contradicted":
                st.error(verdict)
            else:
                st.warning(verdict)

            st.markdown("### Confidence")
            st.progress(float(confidence))
            st.write(f"**{confidence:.2%}**")

            st.markdown("### Supporting Evidence")
            if evidence:
                st.info(evidence)
            else:
                st.info("No supporting evidence available.")

    # ==========================================================
    # Corrected Response
    # ==========================================================

    st.markdown("---")
    st.markdown("## ✨ Corrected Response")
    st.success(result["corrected_response"])

    # ==========================================================
    # Verification Summary
    # ==========================================================

    st.markdown("---")
    st.markdown("## 📊 Verification Summary")

    summary = result["summary"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Claims", summary["total"])

    with col2:
        st.metric("✅ Supported", summary["supported"])

    with col3:
        st.metric("❌ Contradicted", summary["contradicted"])

    with col4:
        st.metric("⚠️ Not Enough", summary["not_enough"])

    st.markdown("---")

    col1, col2, col3 = st.columns([4, 2, 4])

    with col2:
        if st.button("🗑️ Clear Results", use_container_width=True):
            del st.session_state["result"]
            st.rerun()