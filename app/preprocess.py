import pandas as pd
import re
from marathi_stemmer import stem_word, lemmatize_word
df = pd.read_csv("../dataset/LDC.csv")

print("Original dataset:")
print(df["Label"].value_counts())

df = df.dropna(subset=["Text"])
df = df[df["Label"] != "Label"]
df["Label"] = df["Label"].str.strip()

stopwords = {
    "आहे","आहेत","होते","होती","होता","आणि","किंवा","मध्ये",
    "मधील","यांच्या","यांचा","याचे","यासाठी","म्हणून","परंतु",
    "मात्र","तर","ही","हे","हा","ते","त्या","तो","ती","एक",
    "या","ने","ला","ना","चा","ची","चे","वर","पासून","पर्यंत"
}


import re

def clean_text(text, mode="lemma"):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    # Remove numbers and non-Devanagari characters
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\u0900-\u097F\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    if mode == "lemma":
        words = [lemmatize_word(word) for word in words]
    elif mode == "stem":
        words = [stem_word(word) for word in words]
    elif mode != "none":
        raise ValueError("mode must be 'lemma', 'stem', or 'none'")

    words = [word for word in words if word not in stopwords]

    return " ".join(words)


df["Clean_Text"] = df["Text"].apply(clean_text)
df = df[df["Clean_Text"].str.len() > 0]

print("\nFinal dataset:")
print(df["Label"].value_counts())

df.to_csv("../dataset/LDC_cleaned.csv", index=False)

print("\nPreprocessing completed successfully!")