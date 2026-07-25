import pandas as pd
import re

# Load dataset
df = pd.read_csv("../dataset/LDC_Train.csv")


# Check original distribution
print("Original dataset:")
print(df["Label"].value_counts())


# Remove missing text rows
df = df.dropna(subset=["Text"])


# Remove extra header rows
df = df[df["Label"] != "Label"]


# Remove spaces from labels
df["Label"] = df["Label"].str.strip()


# Check after cleaning labels
print("\nAfter header removal:")
print(df["Label"].value_counts())


# Text cleaning function
def clean_text(text):

    # Convert to string
    text = str(text)

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation and special characters
    text = re.sub(r"[^\w\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    # Remove starting and ending spaces
    text = text.strip()

    return text


# Apply cleaning
df["Clean_Text"] = df["Text"].apply(clean_text)


# Remove empty cleaned text
df = df[df["Clean_Text"].str.len() > 0]


# Final dataset distribution
print("\nFinal dataset:")
print(df["Label"].value_counts())


# Save processed dataset
df.to_csv("../dataset/LDC_Train_cleaned.csv", index=False)


print("\nPreprocessing completed successfully!")