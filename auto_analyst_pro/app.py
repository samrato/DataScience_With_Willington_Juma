import streamlit as st
import pandas as pd
from utils import *
import time 

st.title(" Auto Data Analyst AI Agent")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("### Dataset Preview")
    st.dataframe(df.head())

    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Analyzing data...")
    progress_bar.progress(25)
    summary = analyze_data(df)
    target = detect_target(df)
    progress_bar.progress(50)

    st.write("### Detected Target Column:", target)
    
    status_text.text("Generating insights with Gemini...")
    gemini_text = gemini_suggest(summary, target)
    progress_bar.progress(75)

    with st.spinner("Generating graph..."):
        generate_graph(df)

    with st.spinner("Generating PDF report..."):
        generate_pdf(summary, gemini_text)
    
    progress_bar.progress(100)
    status_text.text("Analysis complete!")


    st.write("##  Gemini Suggestion")
    st.write(gemini_text)

    st.image("graph.png")

    with open("report.pdf", "rb") as f:
        st.download_button(" Download PDF Report", f, "report.pdf")
