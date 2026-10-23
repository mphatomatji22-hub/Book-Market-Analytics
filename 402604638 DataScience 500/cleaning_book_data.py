
import pandas as pd

# Step 1: Load the books scraped from catalogue page 20 onwards
df = pd.read_csv("books_raw.csv")

print("ORIGINAL DATASET")
print("Number of books:", len(df))
print(df.head())

# Step 2: Remove unnecessary spaces from text columns
text_columns = ["title", "rating_word", "stock_status", "genre"]

for column in text_columns:
    if column in df.columns:
        df[column] = df[column].astype("string").str.strip()

# Step 3: Convert numerical columns to numeric data types
df["price_gbp"] = pd.to_numeric(df["price_gbp"], errors="coerce")
df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
df["stock_quantity"] = pd.to_numeric(
    df["stock_quantity"], errors="coerce"
)

# Step 4: Standardise rating values if the numeric rating is missing
rating_map = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}

if "rating_word" in df.columns:
    mapped_ratings = df["rating_word"].map(rating_map)
    df["rating"] = df["rating"].fillna(mapped_ratings)

# Step 5: Standardise stock status
if "stock_status" in df.columns:
    df["stock_status"] = (
        df["stock_status"]
        .str.lower()
        .str.strip()
        .map({
            "in stock": "In stock",
            "out of stock": "Out of stock"
        })
        .fillna("Unknown")
    )

# Step 6: Replace missing genre labels
if "genre" in df.columns:
    df["genre"] = df["genre"].fillna("Unknown")
else:
    df["genre"] = "Unknown"
    print("Warning: Genre column was missing in the raw dataset.")

# Step 7: Check for missing values
print("\nMISSING VALUES BEFORE CLEANING")
print(df.isnull().sum())

# Step 8: Remove duplicate books
before_duplicates = len(df)
df = df.drop_duplicates(subset=["title"], keep="first")
duplicates_removed = before_duplicates - len(df)

print("\nDuplicate books removed:", duplicates_removed)

# Step 9: Remove records without essential information
before_missing = len(df)
df = df.dropna(subset=["title", "price_gbp"])
missing_removed = before_missing - len(df)

print("Records removed for missing title or price:", missing_removed)

# Step 10: Remove invalid prices and ratings
df = df[df["price_gbp"] >= 0]
df = df[df["rating"].between(1, 5) | df["rating"].isna()]

# Stock quantity should not be negative
df.loc[df["stock_quantity"] < 0, "stock_quantity"] = pd.NA

# Step 11: Save the cleaned dataset
df.to_csv("books_cleaned.csv", index=False)

# Step 12: Display cleaning results
print("\nCLEANING COMPLETE")
print("Books remaining:", len(df))
print("\nCleaned dataset preview:")
print(df.head())

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nBooks by genre:")
print(df["genre"].value_counts())

print("\nSaved cleaned dataset as books_cleaned.csv")

