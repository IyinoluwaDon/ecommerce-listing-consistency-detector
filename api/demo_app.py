"""
demo_app.py — Streamlit demo for the multimodal content moderation model.

Run:
    streamlit run demo_app.py

Assumes the FastAPI backend (main.py) is already running at localhost:8000.
For a self-contained demo without a separate backend process, see the
"standalone mode" note at the bottom of this file.
"""
import streamlit as st
import requests

API_URL = "http://localhost:8000/predict"

st.set_page_config(page_title="Content Moderation Demo", page_icon="🛍️")
st.title("🛍️ Multimodal Content Moderation Demo")
st.markdown(
    "Upload a product photo and its listing text. The model predicts whether "
    "they genuinely correspond, or whether the pairing looks inconsistent — "
    "a proxy signal for mislabeled or recycled-image listings."
)

col1, col2 = st.columns(2)

with col1:
    uploaded_image = st.file_uploader("Product image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded image", use_container_width=True)

with col2:
    designation = st.text_input("Product title", placeholder="e.g. Piscine hors sol gré 915 x 470 cm")
    description = st.text_area("Description (optional)", placeholder="Additional listing details...")

if st.button("Check consistency", type="primary", disabled=not (uploaded_image and designation)):
    with st.spinner("Analyzing..."):
        files = {"image": (uploaded_image.name, uploaded_image.getvalue(), uploaded_image.type)}
        data = {"designation": designation, "description": description}
        try:
            response = requests.post(API_URL, files=files, data=data, timeout=30)
            response.raise_for_status()
            result = response.json()

            if result["label"] == "compliant":
                st.success(f"✅ **Compliant** — text and image appear to match ({result['confidence']:.1%} confidence)")
            else:
                st.error(f"⚠️ **Non-compliant** — text and image appear inconsistent ({result['confidence']:.1%} confidence)")

            st.progress(result["non_compliant_prob"])
            st.caption(
                f"Compliant probability: {result['compliant_prob']:.1%} | "
                f"Non-compliant probability: {result['non_compliant_prob']:.1%}"
            )
        except requests.exceptions.ConnectionError:
            st.error(
                "Could not reach the API. Make sure `uvicorn main:app --reload --port 8000` "
                "is running in a separate terminal."
            )

st.divider()
st.caption(
    "⚠️ This model was trained on a synthetic mismatch-detection proxy task "
    "(Rakuten France product catalog with artificially swapped images), not "
    "real content-policy violation data. See the project README for full "
    "methodology and limitations."
)

# --- Standalone mode note ---
# To run this demo without a separate FastAPI process (e.g. for a single-file
# Hugging Face Spaces deployment), import run_inference directly from api/main.py
# instead of making an HTTP request, and load the model once at the top of this
# script with @st.cache_resource.
