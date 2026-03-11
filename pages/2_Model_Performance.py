import streamlit as st
import json
import pandas as pd

st.title("Model Performance Comparison")

# load metrics
with open("metrics.json") as f:
    metrics = json.load(f)

# convert to table
df = pd.DataFrame(metrics).T

st.subheader("Evaluation Metrics (Test Set)")
st.dataframe(df.style.format("{:.3f}"))

# highlight best model
best_model = df["f1"].idxmax()

st.success(f"Best Model: {best_model} (Highest F1 Score)")

st.markdown("""
**Metrics Explained**

- Accuracy → overall correct predictions  
- Precision → correctness of approved loans  
- Recall → capture of truly approved loans  
- F1 → balance of precision & recall  
""")
