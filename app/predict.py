import re
import joblib

# Load saved model and vectorizer
model = joblib.load("../models/news_classifier.pkl")
vectorizer = joblib.load("../models/vectorizer.pkl")


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
    text = str(text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def predict_news(news):

    news = clean_text(news)

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)[0]

    return label_mapping.get(prediction, prediction)


# Test
if __name__ == "__main__":

    news = input("मराठी बातमी लिहा : ")

    print("श्रेणी :", predict_news(news))