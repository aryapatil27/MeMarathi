import pandas as pd 
import re 
import joblib 
 
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split 
 
from sklearn.svm import LinearSVC 
from sklearn.linear_model import LogisticRegression, SGDClassifier 
from sklearn.naive_bayes import MultinomialNB 
from sklearn.ensemble import RandomForestClassifier 
 
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
 
# ==================== TF-IDF FEATURE EXTRACTION ==================== 
 
vectorizer = TfidfVectorizer( 
 
    # Keep a maximum of 100,000 text features 
    max_features=100000, 
 
    # Use both: 
    # 1 → Unigrams (single words) 
    # 2 → Bigrams (two consecutive words) 
    ngram_range=(1,2), 
 
    # Ignore words/terms that appear in fewer than 2 documents 
    min_df=2, 
 
    # Ignore terms that appear in more than 90% of the documents 
    # because they are too common 
    max_df=0.90, 
 
    # Apply logarithmic scaling to term frequency 
    # to reduce the effect of very frequent words 
    sublinear_tf=True 
) 
 
X = vectorizer.fit_transform(df["Clean_Text"]) 
y = df["Label"] 
 
X_train, X_test, y_train, y_test = train_test_split( 
    X, y, test_size=0.2, random_state=42, stratify=y 
) 
 
print("Training Samples:", X_train.shape[0]) 
print("Testing Samples:", X_test.shape[0]) 
 
 
# Models 
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
    ), 
 
    "SGD Classifier": SGDClassifier( 
        loss="hinge", 
        max_iter=1000, 
        class_weight="balanced", 
        random_state=42 
    ) 
} 
 
 
# Model comparison 
results = [] 
 
for name, model in models.items(): 
 
    print("\n" + "=" * 60) 
    print("MODEL:", name) 
    print("=" * 60) 
 
    model.fit(X_train, y_train) 
 
    train_predictions = model.predict(X_train) 
    test_predictions = model.predict(X_test) 
 
    train_accuracy = accuracy_score( 
        y_train, 
        train_predictions 
    ) 
 
    test_accuracy = accuracy_score( 
        y_test, 
        test_predictions 
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
        "Training Accuracy": train_accuracy * 100, 
        "Testing Accuracy": test_accuracy * 100, 
        "Precision": precision * 100, 
        "Recall": recall * 100, 
        "F1-Score": f1 * 100 
    }) 
 
    print("Training Accuracy:", round(train_accuracy * 100, 2), "%") 
    print("Testing Accuracy:", round(test_accuracy * 100, 2), "%") 
    print("Precision:", round(precision * 100, 2), "%") 
    print("Recall:", round(recall * 100, 2), "%") 
    print("F1-Score:", round(f1 * 100, 2), "%") 
 
 
# Comparison table 
results_df = pd.DataFrame(results) 
 
print("\n") 
print("=" * 80) 
print("MODEL COMPARISON") 
print("=" * 80) 
 
print(results_df.round(2).to_string(index=False)) 
 
 
# Select final model 
best_model_name = results_df.loc[ 
    results_df["F1-Score"].idxmax(), 
    "Model" 
] 
 
print("\nSelected Model:", best_model_name) 
 
 
# Train final model 
final_model = models[best_model_name] 
 
final_model.fit(X_train, y_train) 
 
final_predictions = final_model.predict(X_test) 
 
 
# Final evaluation 
print("\nClassification Report\n") 
 
print( 
    classification_report( 
        y_test, 
        final_predictions, 
        zero_division=0 
    ) 
) 
 
print("\nConfusion Matrix\n") 
 
print( 
    confusion_matrix( 
        y_test, 
        final_predictions 
    ) 
) 
 
 
# Save model 
joblib.dump( 
    final_model, 
    "../models/news_classifier.pkl" 
) 
 
joblib.dump( 
    vectorizer, 
    "../models/vectorizer.pkl" 
) 
 
print("\nFinal Model Saved Successfully!")