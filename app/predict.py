import re
import joblib
from pathlib import Path


# Path to project root
BASE_DIR = Path(__file__).resolve().parent


# Load trained model and TF-IDF vectorizer
model = joblib.load(
    BASE_DIR.parent / "models" / "news_classifier.pkl"
)

vectorizer = joblib.load(
    BASE_DIR.parent / "models" / "vectorizer.pkl"
)


# Marathi stopwords
stopwords = {
    "आहे", "आहेत", "होते", "होती", "होता", "आणि", "किंवा", "मध्ये",
    "मधील", "यांच्या", "यांचा", "याचे", "यासाठी", "म्हणून", "परंतु",
    "मात्र", "तर", "ही", "हे", "हा", "ते", "त्या", "तो", "ती", "एक",
    "या", "ने", "ला", "ना", "चा", "ची", "चे", "वर", "पासून", "पर्यंत"
}


# Marathi label mapping
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


def clean_text(text):
    """
    Preprocess Marathi news text.
    This should match the preprocessing used during model training.
    """

    text = str(text)

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Keep only Marathi characters and spaces
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove Marathi stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]

    return " ".join(tokens)


def predict_news(news):
    """
    Predict the category of Marathi news.
    """

    # Preprocess input
    news = clean_text(news)

    # Convert text into TF-IDF features
    vector = vectorizer.transform([news])

    # Predict category
    prediction = model.predict(vector)[0]

    # Convert English label to Marathi label
    return label_mapping.get(prediction, prediction)


# Test prediction from terminal
if __name__ == "__main__":

    news = input("मराठी बातमी लिहा : ")

    category = predict_news(news)

    print("श्रेणी :", category)