import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv("../dataset/LDC.csv")

df = df.dropna(subset=["Text"])
df = df[df["Label"] != "Label"]
df["Label"] = df["Label"].str.strip()

stopwords = {
    "आहे","आहेत","होते","होती","होता","आणि","किंवा","मध्ये",
    "मधील","यांच्या","यांचा","याचे","यासाठी","म्हणून","परंतु",
    "मात्र","तर","ही","हे","हा","ते","त्या","तो","ती","एक",
    "या","ने","ला","ना","चा","ची","चे","वर","पासून","पर्यंत"
}

def clean_text(text):
    text = str(text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    return " ".join(tokens)

df["Clean_Text"] = df["Text"].apply(clean_text)
df = df[df["Clean_Text"].str.len() > 0]

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["Clean_Text"])
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

model = LinearSVC(
    C=2.0,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

print("\nTraining Accuracy:",
      round(accuracy_score(y_train, train_predictions) * 100, 2), "%")

print("Testing Accuracy:",
      round(accuracy_score(y_test, test_predictions) * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, test_predictions, zero_division=0))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, test_predictions))

joblib.dump(model, "../models/news_classifier.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")

print("\nModel Saved Successfully!")