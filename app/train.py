
from pathlib import Path
from preprocess import clean_text
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)




# Project paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_PATH = PROJECT_DIR / "dataset" / "LDC.csv"
CLEANED_PATH = PROJECT_DIR / "dataset" / "LDC_cleaned.csv"
MODEL_DIR = PROJECT_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==================== LOAD DATA ====================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["Text", "Label"]).copy()

# Remove repeated header rows, if present
df["Label"] = df["Label"].astype(str).str.strip()
df = df[df["Label"] != "Label"]

# Remove empty text and labels
df["Text"] = df["Text"].astype(str).str.strip()
df = df[
    (df["Text"] != "") &
    (df["Label"] != "")
].copy()

# Apply the same preprocessing used during prediction
df["Clean_Text"] = df["Text"].apply(
    lambda text: clean_text(text, mode="lemma")
)

df = df[df["Clean_Text"].str.len() > 0].copy()

print("Dataset samples:", len(df))
print("\nCategory distribution:")
print(df["Label"].value_counts())

# Save the cleaned dataset for inspection
df.to_csv(CLEANED_PATH, index=False, encoding="utf-8-sig")


# ==================== SPLIT DATA ====================

# Split text BEFORE fitting TF-IDF to avoid test-data leakage
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["Clean_Text"],
    df["Label"],
    test_size=0.2,
    random_state=42,
    stratify=df["Label"]
)

print("\nTraining samples:", len(X_train_text))
print("Testing samples:", len(X_test_text))


# ==================== TF-IDF ====================

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.90,
    sublinear_tf=True
)

# Learn vocabulary and IDF from training text only
X_train = vectorizer.fit_transform(X_train_text)

# Transform test text using the same fitted vectorizer
X_test = vectorizer.transform(X_test_text)

print("TF-IDF features:", X_train.shape[1])


# ==================== MODELS ====================

models = {
    "Linear SVM": LinearSVC(
        C=2.0,
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression": LogisticRegression(
        C=2.0,
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Multinomial Naive Bayes": MultinomialNB(),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}


# ==================== MODEL COMPARISON ====================

results = []
trained_models = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("MODEL:", name)
    print("=" * 60)

    model.fit(X_train, y_train)
    trained_models[name] = model

    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_accuracy = accuracy_score(
        y_train, train_predictions
    )
    test_accuracy = accuracy_score(
        y_test, test_predictions
    )

    report = classification_report(
        y_test,
        test_predictions,
        output_dict=True,
        zero_division=0
    )

    precision = report["weighted avg"]["precision"]
    recall = report["weighted avg"]["recall"]
    f1 = report["weighted avg"]["f1-score"]

    results.append({
        "Model": name,
        "Training Accuracy (%)": train_accuracy * 100,
        "Testing Accuracy (%)": test_accuracy * 100,
        "Precision (%)": precision * 100,
        "Recall (%)": recall * 100,
        "F1-Score (%)": f1 * 100
    })

    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Testing Accuracy:  {test_accuracy * 100:.2f}%")
    print(f"Precision:         {precision * 100:.2f}%")
    print(f"Recall:            {recall * 100:.2f}%")
    print(f"F1-Score:          {f1 * 100:.2f}%")


# ==================== COMPARISON TABLE ====================

results_df = pd.DataFrame(results)

print("\n" + "=" * 90)
print("MODEL COMPARISON")
print("=" * 90)
print(results_df.round(2).to_string(index=False))


# ==================== SELECT FINAL MODEL ====================

# Select using weighted F1-score on the held-out test set.
# For a final research comparison, use a validation set or
# cross-validation for model selection, then evaluate once
# on an untouched test set.
best_index = results_df["F1-Score (%)"].idxmax()
best_model_name = results_df.loc[best_index, "Model"]
final_model = trained_models[best_model_name]

print("\nSelected Model:", best_model_name)


# ==================== FINAL EVALUATION ====================

final_predictions = final_model.predict(X_test)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        final_predictions,
        zero_division=0
    )
)

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, final_predictions))


# ==================== SAVE MODEL ====================

joblib.dump(
    final_model,
    MODEL_DIR / "news_classifier.pkl"
)

joblib.dump(
    vectorizer,
    MODEL_DIR / "vectorizer.pkl"
)

print("\nCleaned dataset saved:", CLEANED_PATH)
print("Selected model:", best_model_name)
print("Model and vectorizer saved successfully!")
