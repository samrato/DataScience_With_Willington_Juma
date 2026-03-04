import os
import pandas as pd
import google.genai as google_genai
import matplotlib.pyplot as plt
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import logging
import streamlit as st
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_logger(name):
    return logging.getLogger(name)

# Load API Key
load_dotenv()
gemini_model = google_genai.GenerativeModel("models/gemini-pro-latest")


def analyze_data(df):
    logger = get_logger(__name__)
    logger.info("Starting data analysis...")
    # Basic Info
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": {col: str(df[col].dtype) for col in df.columns}
    }

    # Numerical Analysis
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        logger.info("Performing numerical analysis...")
        summary["numerical_summary"] = df[numerical_cols].describe().to_dict()
        if len(numerical_cols) > 1:
            summary["correlation_matrix"] = df[numerical_cols].corr().to_dict()

    # Categorical Analysis
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        logger.info("Performing categorical analysis...")
        summary["categorical_summary"] = {}
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "value_counts": df[col].value_counts().to_dict()
            }
    logger.info("Data analysis complete.")
    return summary

def detect_target(df):
    logger = get_logger(__name__)
    logger.info("Detecting target column...")
    target = df.columns[-1]
    logger.info(f"Detected target column: {target}")
    return target

def gemini_suggest(summary, target):
    logger = get_logger(__name__)
    logger.info("Generating insights with Gemini...")
    time.sleep(5)  # Simulate network latency or complex processing for demonstration
    prompt = f"""
    You are a senior data scientist.
    Here is a detailed summary of a dataset:
    {summary}
    The target column is: {target}

    Provide a detailed data analysis based on this summary. 
    - What are the key characteristics of the dataset?
    - What are some potential data quality issues?
    - What are the relationships between the features and the target?
    - What are some interesting insights you can find?
    
    Do not suggest any machine learning models.
    """
    response = gemini_model.generate_content(prompt)
    logger.info("Gemini insights generated.")
    return response.text




def generate_graph(df):
    logger = get_logger(__name__)
    logger.info("Generating graph...")
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        with st.spinner("Generating graph..."):
            df[numeric_cols[0]].hist()
            plt.title("Distribution")
            plt.savefig("graph.png")
            plt.close()
            logger.info("Graph generated and saved as graph.png")


def generate_pdf(summary, gemini_text):
    logger = get_logger(__name__)
    logger.info("Generating PDF report...")
    with st.spinner("Generating PDF report..."):
        doc = SimpleDocTemplate("report.pdf")
        elements = []

        style = ParagraphStyle(
            name="Normal",
            fontSize=12,
            textColor=colors.black
        )

        elements.append(Paragraph("<b>AUTO DATA ANALYST REPORT</b>", style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(str(summary), style))
        elements.append(Spacer(1, 0.3 * inch))
        elements.append(Paragraph(gemini_text, style))
        elements.append(Spacer(1, 0.3 * inch))


        doc.build(elements)
        logger.info("PDF report generated and saved as report.pdf")