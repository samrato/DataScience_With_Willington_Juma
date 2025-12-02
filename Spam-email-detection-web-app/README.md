📧 Spam Email Detection Web App

A simple yet effective web application for detecting spam emails using Logistic Regression and TF-IDF vectorization. The model is trained on a labeled dataset of emails and deployed using Flask, allowing users to check whether a message is spam or not through a web interface.

🚀 Features

Machine Learning Model: Logistic Regression with class balancing for better spam classification.

Vectorization: TF-IDF (TfidfVectorizer) to convert text into numerical features.

Web Interface: User-friendly form to input messages and view predictions.

High Accuracy: Achieved ~98% accuracy on the test dataset.

Visualization: Confusion matrix for model performance analysis.

📂 Project Structure
├── app.py                # Flask application  
├── email_spam_detection.pkl   # Trained ML model  
├── vectorizer.pkl        # Saved TF-IDF vectorizer  
├── templates/  
│   ├── form.html         # Input form for message  
│   └── result.html       # Displays prediction results  
├── requirements.txt      # Project dependencies  
└── README.md             # Project documentation  

⚙️ Installation & Setup

Clone the repository

git clone https://github.com/Amon-Mugo/Spam-email-detection-web-app.git
cd Spam-email-detection-web-app


Create and activate a virtual environment (recommended)

python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows


Install dependencies

pip install -r requirements.txt


Run the Flask app

python app.py


Open your browser at http://127.0.0.1:5000/

🖼️ Screenshots
Input Form

Prediction Result

📊 Model Performance

Accuracy: ~98%

Precision (Spam detection): ~95%

Confusion Matrix: Visualized using Seaborn for better interpretability.

🔮 Future Improvements

✅ Deploy on Heroku / Render for public access.

✅ Add support for multiple models (Naive Bayes, SVM).

✅ Improve UI using Bootstrap / Tailwind CSS.

✅ Extend dataset for more robust generalization.

✅ Implement an API endpoint for integration with other services.

🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to enhance the app.

📜 License

This project is licensed under the MIT License.
