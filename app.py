import os
import json
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

MODEL_PATH = "models/disease_model.keras"
LABEL_PATH = "models/labels.json"

st.set_page_config(page_title="Disease Prediction - DL", page_icon="🩺")

st.title("🩺 Disease Prediction Using Deep Learning")
st.write("Select the symptoms and click **Predict**.")

st.warning(
    "Educational project only — this prediction is not medical advice or a diagnosis. "
    "For health concerns, consult a qualified healthcare professional."
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()

model = keras.models.load_model(MODEL_PATH)
with open(LABEL_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

symptoms = metadata["symptoms"]
classes = metadata["classes"]

# Make symptom names easier to read
pretty = {s: s.replace("_", " ").title() for s in symptoms}

selected = st.multiselect(
    "Choose symptoms:",
    options=symptoms,
    format_func=lambda x: pretty[x]
)

if st.button("🔍 Predict", type="primary"):
    x = np.array([[1 if s in selected else 0 for s in symptoms]], dtype="float32")
    probs = model.predict(x, verbose=0)[0]
    order = np.argsort(probs)[::-1]

    best = order[0]
    st.success(f"Predicted class: **{classes[best]}**")
    st.info(
        f"Model confidence for this class: **{probs[best] * 100:.1f}%**. "
        "This confidence is only a model score on this educational dataset."
    )

    results = pd.DataFrame({
        "Disease": [classes[i] for i in order[:3]],
        "Model score (%)": [round(float(probs[i] * 100), 2) for i in order[:3]]
    })
    st.dataframe(results, hide_index=True, use_container_width=True)

st.caption("Deep Learning mini-project | Keras + Streamlit")
