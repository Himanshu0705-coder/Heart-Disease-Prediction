
# ❤️ Heart Disease Prediction and AI Health Assistant
## 🌟 Overview
Live link - https://himanshu0705-coder-heart-disease-prediction-app-hcfrb1.streamlit.app/
This project is a comprehensive **Streamlit web application** designed for heart disease prediction and data exploration. It provides a user-friendly interface for:

1.  **Exploring a dataset** of heart health indicators.
2.  **Visualizing data insights** through interactive Plotly charts.
3.  **Predicting a patient's risk** of heart disease using a pre-trained Random Forest model.
4.  **Interacting with an AI Health Assistant** powered by the Google Gemini API for general health inquiries.

The application is structured into four main pages accessible via the sidebar:
* `📊 Dataset View`
* `📈 Insights & Evaluation`
* `🔮 Predict Heart Disease`
* `🤖 AI Health assistance`

## 🚀 Features

* **Interactive Data Visualization:** Uses `plotly.express` and `plotly.graph_objects` for dynamic histograms, pie charts, correlation heatmaps, confusion matrices, and ROC curves.
* **Machine Learning Prediction:** Loads a pre-trained `random_forest_model.pkl` to predict heart disease risk based on user input.
* **AI Integration:** Leverages the **Google Gemini API** (`google-genai` and `langchain-google-genai`) to provide an intelligent health assistant.
* **Custom Styling:** Utilizes custom CSS for a clean, modern, and health-themed UI.

## ⚙️ Setup and Installation

### 1. Prerequisites

* Python (3.9 or higher)
* Git

### 2. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
````

### 3\. Create and Activate Virtual Environment (Recommended)

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 4\. Install Dependencies

The application relies on the following libraries (from your `app.py`):

  * `streamlit`
  * `pandas`
  * `numpy`
  * `scikit-learn` (for `RandomForestClassifier`)
  * `plotly`
  * `seaborn`, `matplotlib`
  * `google-genai`
  * `langchain-google-genai`

Create a `requirements.txt` file in the root directory:

```
streamlit
pandas
numpy
scikit-learn
plotly
seaborn
matplotlib
google-genai
langchain-google-genai
```

Now, install them:

```bash
pip install -r requirements.txt
```

### 5\. API Key Configuration

The **AI Health Assistant** page requires a Gemini API key. You must configure this in a file named **`.streamlit/secrets.toml`** in your repository's root directory.

```toml
# .streamlit/secrets.toml
GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"
```

> **Note:** Get your free API key from Google AI Studio. **Never commit your actual API key directly to GitHub.** Streamlit Cloud securely manages this file for deployment.

### 6\. Model and Data Files

The application expects two files in the root directory:

  * **`dataset.csv`**: The heart disease data (a placeholder DataFrame is created if not found).
  * **`random_forest_model.pkl`**: Your pre-trained Random Forest model (a placeholder model is created if not found).

## 🏃 How to Run the App

Execute the following command from the project's root directory:

```bash
streamlit run app.py
```

The app will automatically open in your web browser at `http://localhost:8501`.

## 📂 Project Structure

```
.
├── app.py                      # Main Streamlit application file
├── requirements.txt            # Python dependencies
├── .streamlit/                 # Directory for Streamlit configuration (secrets.toml)
│   └── secrets.toml            # Used to store the GEMINI_API_KEY
├── random_forest_model.pkl     # Pre-trained ML model (required)
└── dataset.csv                 # Dataset file (optional, placeholder created if missing)
└── README.md                   # This file
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

## 🤝 Contribution

Contributions are welcome\! Please feel free to submit a pull request or open an issue.

## 👨‍💻 Author

  * **Your Name** (or GitHub Username)
  * *Link to your portfolio or LinkedIn (Optional)*

<!-- end list -->

