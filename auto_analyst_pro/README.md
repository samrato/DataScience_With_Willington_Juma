# Auto Data Analyst AI Agent

This project is a Streamlit web application that automates data analysis, model training, and report generation using a generative AI model.

## Features

-   **Upload CSV:** Users can upload their own CSV files for analysis.
-   **Data Analysis:** The application provides a summary of the dataset, including the number of rows and columns, column names, and missing values.
-   **Target Detection:** It automatically detects the target column for machine learning tasks.
-   **AI-Powered Suggestions:** Utilizes a generative AI model to suggest a problem type (Classification/Regression) and a suitable model.
-   **Model Training:** Trains a Random Forest model (Classifier or Regressor) based on the suggested problem type.
-   **Performance Evaluation:** Displays the model's performance score (Accuracy for Classification, Mean Squared Error for Regression).
-   **Report Generation:** Generates a downloadable PDF report summarizing the analysis, AI suggestions, and model performance.
-   **Downloadable Model:** Allows users to download the trained model as a `.pkl` file.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your environment variables:**
    -   Create a `.env` file in the `auto_analyst_pro` directory.
    -   Add your Gemini API key to the `.env` file:
        ```
        GEMINI_API_KEY=your_api_key_here
        ```

## How to Run

1.  **Make sure you are in the `auto_analyst_pro` directory and the virtual environment is activated.**

2.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

3.  **Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).**
