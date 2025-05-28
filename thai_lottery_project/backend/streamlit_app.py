# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/streamlit_app.py
# /Users/apichet/Downloads/thai_lottery_project/thai_lottery_project/backend/streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from quantum_multiverse_simulator import (
    simulate_multiverse,
    analyze_multiverse_draws
)
from quantum_model_bank import MODEL_LIST

# ===== 🌌 INITIALIZATION =====
state_ids = [f"Draw_{i}" for i in range(500)]

st.set_page_config(page_title="Quantum Universe Explorer", layout="wide")
st.title("🌌 Quantum Universe Explorer")
st.markdown("Simulate parallel quantum lottery universes and explore entangled outcomes.")

# ===== ⚙️ Controls =====
n_worlds = st.slider("🌐 Universes per Model", 10, 200, 50)
entropy_cutoff = st.slider("🌀 Entropy Threshold (for Reliable Universe)", 3.0, 10.0, 6.0)
top_k = st.slider("🔝 Top K Draws to Show", 3, 10, 5)

# ===== 🎮 Run Button =====
if st.button("🚀 Simulate Multiverse"):
    with st.spinner("🌀 Generating multiverse..."):
        df = simulate_multiverse(state_ids, models=MODEL_LIST, n_worlds=n_worlds)

    # ===== 🔍 Result Display =====
    st.subheader("🧪 All Universe Collapses")
    st.dataframe(df)

    st.subheader("📊 Entropy Distribution")
    st.bar_chart(df["entropy"])

    st.subheader("🌈 Top Recommended Draws")
    df_top = analyze_multiverse_draws(df, entropy_threshold=entropy_cutoff, top_k=top_k)
    st.dataframe(df_top)

    # Optional: Pie chart of frequency
    fig, ax = plt.subplots()
    df_top.set_index("draw")["frequency"].plot.pie(autopct="%.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)
