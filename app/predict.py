
from pathlib import Path

import joblib

from app.preprocess import clean_text

# ==================== LOAD MODEL ====================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = PROJECT_DIR / "models"

model = joblib.load(
    MODEL_DIR / "news_classifier.pkl"
)

vectorizer = joblib.load(
    MODEL_DIR / "vectorizer.pkl"
)


# ==================== LABEL MAPPING ====================

label_mapping = {
    "Sports": "क्रीडा",
    "Politics": "राजकारण",
    "Auto": "वाहन",
    "Tech": "तंत्रज्ञान",
    "Health": "आरोग्य",
    "Crime": "गुन्हे",
    "International": "आंतरराष्ट्रीय",
    "Education": "शिक्षण",
    "Travel": "प्रवास",
    "Fashion": "फॅशन",
    "Bhakti": "भक्ती",
    "Manoranjan": "मनोरंजन"
}


# ==================== PREDICTION ====================

def predict_news(news):
    """Predict the category of Marathi news."""

    if not isinstance(news, str) or not news.strip():
        return "कृपया मराठी बातमी प्रविष्ट करा."

    # Use exactly the same preprocessing as training
    cleaned_news = clean_text(news, mode="lemma")

    if not cleaned_news:
        return "प्रक्रियेनंतर बातमीमध्ये वैध मजकूर उरलेला नाही."

    # Convert the cleaned text into TF-IDF features
    vector = vectorizer.transform([cleaned_news])

    # Predict the news category
    prediction = model.predict(vector)[0]

    # Return the Marathi category name
    return label_mapping.get(prediction, prediction)


# ==================== TERMINAL TEST ====================

if __name__ == "__main__":

    news = input("मराठी बातमी लिहा: ")

    category = predict_news(news)

    print("श्रेणी:", category)
