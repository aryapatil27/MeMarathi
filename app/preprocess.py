
import pandas as pd
import re
from pathlib import Path

from app.marathi_stemmer import stem_word, lemmatize_word


# Stopwords
stopwords = {
    "आहे", "आहेत", "होते", "होती", "होता", "आणि", "किंवा", "मध्ये",
    "मधील", "यांच्या", "यांचा", "याचे", "यासाठी", "म्हणून", "परंतु",
    "मात्र", "तर", "ही", "हे", "हा", "ते", "त्या", "तो", "ती", "एक",
    "या", "ने", "ला", "ना", "चा", "ची", "चे", "वर", "पासून", "पर्यंत"
}


# Text preprocessing function
def clean_text(text, mode="lemma"):

    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Remove numbers
    text = re.sub(r"\d+", " ", text)

    # Keep only Devanagari characters and spaces
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Split text into words
    words = text.split()

    # Apply lemmatization or stemming
    if mode == "lemma":
        words = [lemmatize_word(word) for word in words]

    elif mode == "stem":
        words = [stem_word(word) for word in words]

    elif mode != "none":
        raise ValueError(
            "mode must be 'lemma', 'stem', or 'none'"
        )

    # Remove stopwords
    words = [word for word in words if word not in stopwords]

    # Return cleaned text
    return " ".join(words)


# Dataset preprocessing
# This section runs only when this file is executed directly.
if __name__ == "__main__":

    # Get project and dataset paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASET_DIR = BASE_DIR / "dataset"

    input_file = DATASET_DIR / "LDC.csv"
    output_file = DATASET_DIR / "LDC_cleaned.csv"

    # Load original dataset
    df = pd.read_csv(input_file)

    print("Original dataset:")
    print(df["Label"].value_counts())

    # Remove rows with missing text
    df = df.dropna(subset=["Text"])

    # Remove repeated header rows
    df = df[df["Label"] != "Label"].copy()

    # Remove extra spaces from labels
    df["Label"] = df["Label"].str.strip()

    # Apply text preprocessing
    df["Clean_Text"] = df["Text"].apply(clean_text)

    # Remove empty cleaned text
    df = df[df["Clean_Text"].str.len() > 0]

    print("\nFinal dataset:")
    print(df["Label"].value_counts())

    # Save the cleaned dataset
    df.to_csv(output_file, index=False)

    print("\nPreprocessing completed successfully!")

