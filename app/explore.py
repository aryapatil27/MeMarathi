import pandas as pd

# Load training dataset
df = pd.read_csv("../dataset/LDC_Train.csv")
print(df[df["Label"] == "Label"])
print(df[df["Text"].isnull()].head())

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nUnique Labels:")
print(df["Label"].unique())

print("\nNumber of Categories:")
print(df["Label"].nunique())

print("\nCategory Counts:")
print(df["Label"].value_counts())