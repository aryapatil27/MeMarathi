import pandas as pd
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("../dataset/LDC_Train.csv")

# Remove missing text and duplicate header rows
df = df.dropna(subset=["Text"])
df = df[df["Label"] != "Label"]


# Function to clean Marathi text
def clean_text(text):
    text = str(text)
    text = re.sub(r"\d+", "", text)           # Remove numbers
    text = re.sub(r"[^\w\s]", " ", text)      # Remove punctuation
    text = re.sub(r"\s+", " ", text)          # Remove extra spaces
    return text.strip()


# Create cleaned text
df["Clean_Text"] = df["Text"].apply(clean_text)

# Convert text into TF-IDF vectors
vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df["Clean_Text"])
y = df["Label"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# Train the model
model = LinearSVC(
    C=2.0,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train, y_train)

# Predict on training data
train_predictions = model.predict(X_train)

# Predict on testing data
test_predictions = model.predict(X_test)

# Calculate accuracy
train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print("\nTraining Accuracy :", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy  :", round(test_accuracy * 100, 2), "%")

print("\nClassification Report\n")
print(classification_report(y_test, test_predictions, zero_division=0))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, test_predictions))

# Save model and vectorizer
joblib.dump(model, "../models/news_classifier.pkl")
joblib.dump(vectorizer, "../models/vectorizer.pkl")

print("\nModel Saved Successfully!")