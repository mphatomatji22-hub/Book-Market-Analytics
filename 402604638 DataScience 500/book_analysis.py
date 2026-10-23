"""
BOOK MARKET ANALYTICS - BOOKS TO SCRAPE
Data analysis and linear regression for 220 books collected from Young Adult onwards across subsequent genres.

Input:
    books_cleaned.csv (preferred), or books_raw.csv as a fallback

Outputs:
    graphs/01_price_distribution_histogram.png
    graphs/02_price_vs_rating_scatter.png
    graphs/03_price_by_rating_violin.png
    graphs/04_stock_status_pie.png
    graphs/05_average_price_by_genre.png
    graphs/06_books_by_genre_lollipop.png
    graphs/07_correlation_heatmap.png (only when suitable)
    graphs/08_regression_actual_vs_predicted.png
    books_price_predictions.csv
    regression_metrics.csv
    summary_statistics.csv
"""

from pathlib import Path
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
GRAPHS_DIR = BASE_DIR / "graphs"
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)

# Use the cleaned dataset when available; otherwise use raw data.
cleaned_file = BASE_DIR / "books_cleaned.csv"
raw_file = BASE_DIR / "books_raw.csv"

if cleaned_file.exists():
    DATA_FILE = cleaned_file
elif raw_file.exists():
    DATA_FILE = raw_file
else:
    raise FileNotFoundError(
        "Could not find books_cleaned.csv or books_raw.csv.\n"
        "Place the CSV file in the same folder as book_analysis.py."
    )

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

print("=" * 60)
print("BOOK MARKET ANALYTICS")
print("Books to Scrape | Genre-based sample of 220 books")
print("=" * 60)
print(f"Loading dataset: {DATA_FILE.name}")

df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()

# ---------------------------------------------------------
# 2. STANDARDISE COLUMN NAMES
# Supports common alternative column names from earlier code.
# ---------------------------------------------------------
rename_candidates = {
    "price": "price_gbp",
    "price (£)": "price_gbp",
    "rating_number": "rating",
    "star_rating": "rating",
    "stock": "stock_status",
    "availability": "stock_status",
    "quantity": "stock_quantity",
    "stock_qty": "stock_quantity",
    "category": "genre",
}
for old_name, new_name in rename_candidates.items():
    if old_name in df.columns and new_name not in df.columns:
        df.rename(columns={old_name: new_name}, inplace=True)

# Ensure expected columns exist, allowing the script to explain
# missing data instead of crashing immediately.
for column in ["title", "price_gbp", "rating", "rating_word",
               "stock_status", "stock_quantity", "genre"]:
    if column not in df.columns:
        df[column] = pd.NA

# Clean text fields.
for column in ["title", "rating_word", "stock_status", "genre"]:
    df[column] = df[column].astype("string").str.strip()

# Convert numeric fields safely.
for column in ["price_gbp", "rating", "stock_quantity"]:
    df[column] = pd.to_numeric(
        df[column].astype("string").str.replace("£", "", regex=False),
        errors="coerce"
    )

# Convert written star ratings to numbers if numeric rating is missing.
rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
mapped_ratings = df["rating_word"].map(rating_map)
df["rating"] = df["rating"].fillna(mapped_ratings)

# Standardise stock labels.
df["stock_status"] = (
    df["stock_status"].str.lower().str.strip()
    .map({
        "in stock": "In stock",
        "out of stock": "Out of stock",
        "available": "In stock",
        "unavailable": "Out of stock",
    })
    .fillna("Unknown")
)

df["genre"] = df["genre"].fillna("Unknown").replace("", "Unknown")
df["title"] = df["title"].fillna("Untitled").replace("", "Untitled")

# Remove rows that cannot support price analysis and reject impossible values.
df = df.dropna(subset=["price_gbp"])
df = df[df["price_gbp"] >= 0].copy()
df.loc[~df["rating"].between(1, 5), "rating"] = pd.NA
df.loc[df["stock_quantity"] < 0, "stock_quantity"] = pd.NA

if df.empty:
    raise ValueError(
        "No rows with valid prices remain. Check the price_gbp column "
        "in your CSV file."
    )

# ---------------------------------------------------------
# 3. DATASET OVERVIEW AND SUMMARY STATISTICS
# ---------------------------------------------------------
print("\nDATASET OVERVIEW")
print("-" * 40)
print(f"Input file: {DATA_FILE.name}")
print(f"Books with valid prices: {len(df)}")
print(f"Genres represented: {df['genre'].nunique()}")
print(f"Missing ratings: {df['rating'].isna().sum()}")
print(f"Missing stock quantities: {df['stock_quantity'].isna().sum()}")

print("\nPRICE SUMMARY (GBP)")
print(df["price_gbp"].describe().round(2).to_string())

summary = df[["price_gbp", "rating", "stock_quantity"]].describe().round(2)
summary.to_csv(BASE_DIR / "summary_statistics.csv")
print("\nSaved summary_statistics.csv")

def save_figure(filename):
    """Save the current Matplotlib figure in the graphs folder."""
    output_path = GRAPHS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: graphs/{filename}")

# ---------------------------------------------------------
# GRAPH 1: HISTOGRAM - PRICE DISTRIBUTION
# ---------------------------------------------------------
print("\nCreating graph 1 of 6: Price distribution histogram...")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="price_gbp", bins=15, kde=True)
plt.title("Distribution of Book Prices", fontsize=16, weight="bold")
plt.xlabel("Price (£)")
plt.ylabel("Number of Books")
save_figure("01_price_distribution_histogram.png")

# ---------------------------------------------------------
# GRAPH 2: SCATTER PLOT - RATING VS PRICE
# ---------------------------------------------------------
print("Creating graph 2 of 6: Rating versus price scatter plot...")
scatter_df = df.dropna(subset=["rating", "price_gbp"])
plt.figure(figsize=(10, 6))

if not scatter_df.empty:
    sns.scatterplot(
        data=scatter_df,
        x="rating",
        y="price_gbp",
        hue="rating",
        palette="viridis",
        s=90,
        alpha=0.8,
        legend=False
    )
    plt.xticks([1, 2, 3, 4, 5])
else:
    plt.text(0.5, 0.5, "No valid ratings available", ha="center", va="center",
             transform=plt.gca().transAxes)

plt.title("Book Price Compared with Star Rating", fontsize=16, weight="bold")
plt.xlabel("Star Rating (1 to 5)")
plt.ylabel("Price (£)")
save_figure("02_price_vs_rating_scatter.png")

# ---------------------------------------------------------
# GRAPH 3: VIOLIN PLOT - PRICE BY RATING
# ---------------------------------------------------------
print("Creating graph 3 of 6: Price by rating violin plot...")
violin_df = df.dropna(subset=["rating", "price_gbp"]).copy()
plt.figure(figsize=(10, 6))

if not violin_df.empty and violin_df["rating"].nunique() >= 2:
    violin_df["rating"] = violin_df["rating"].astype(int).astype(str)
    rating_order = [str(n) for n in range(1, 6)
                    if str(n) in violin_df["rating"].unique()]
    sns.violinplot(
        data=violin_df,
        x="rating",
        y="price_gbp",
        order=rating_order,
        inner="quartile",
        cut=0
    )
    plt.xlabel("Star Rating")
    plt.ylabel("Price (£)")
else:
    plt.text(
        0.5, 0.5,
        "A violin plot needs valid prices across at least two rating groups.",
        ha="center", va="center", wrap=True,
        transform=plt.gca().transAxes
    )
    plt.xlabel("Star Rating")
    plt.ylabel("Price (£)")

plt.title("Price Distribution Across Star Ratings", fontsize=16, weight="bold")
save_figure("03_price_by_rating_violin.png")

# ---------------------------------------------------------
# GRAPH 4: PIE CHART - STOCK STATUS
# ---------------------------------------------------------
print("Creating graph 4 of 6: Stock status pie chart...")
stock_counts = df["stock_status"].value_counts()
plt.figure(figsize=(8, 8))

if not stock_counts.empty:
    plt.pie(
        stock_counts.values,
        labels=stock_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75
    )
    plt.title("Proportion of Books by Stock Status", fontsize=16, weight="bold")
    plt.axis("equal")
else:
    plt.text(0.5, 0.5, "No stock status data available",
             ha="center", va="center", transform=plt.gca().transAxes)

save_figure("04_stock_status_pie.png")

# ---------------------------------------------------------
# GRAPH 5: HORIZONTAL BAR CHART - AVERAGE PRICE BY GENRE
# ---------------------------------------------------------
print("Creating graph 5 of 6: Average price by genre horizontal bar chart...")
genre_prices = (
    df[df["genre"] != "Unknown"]
    .groupby("genre", dropna=False)["price_gbp"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(11, max(6, 0.35 * max(len(genre_prices), 1))))
if not genre_prices.empty:
    sns.barplot(
        x=genre_prices.values,
        y=genre_prices.index,
        orient="h"
    )
    plt.xlabel("Average Price (£)")
    plt.ylabel("Genre")
else:
    plt.text(
        0.5, 0.5, "Genre information is missing or unavailable.",
        ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.xlabel("Average Price (£)")
    plt.ylabel("Genre")

plt.title("Average Book Price by Genre", fontsize=16, weight="bold")
save_figure("05_average_price_by_genre.png")

# ---------------------------------------------------------
# GRAPH 6: LOLLIPOP CHART - NUMBER OF BOOKS BY GENRE
# ---------------------------------------------------------
print("Creating graph 6 of 6: Books by genre lollipop chart...")
genre_counts = (
    df[df["genre"] != "Unknown"]["genre"]
    .value_counts()
    .sort_values()
)

plt.figure(figsize=(11, max(6, 0.35 * max(len(genre_counts), 1))))
if not genre_counts.empty:
    positions = range(len(genre_counts))
    plt.hlines(
        y=list(positions),
        xmin=0,
        xmax=genre_counts.values,
        linewidth=2
    )
    plt.plot(
        genre_counts.values,
        list(positions),
        "o",
        markersize=8
    )
    plt.yticks(list(positions), genre_counts.index)
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")
else:
    plt.text(
        0.5, 0.5, "Genre information is missing or unavailable.",
        ha="center", va="center", transform=plt.gca().transAxes
    )
    plt.xlabel("Number of Books")
    plt.ylabel("Genre")

plt.title("Number of Books Collected per Genre", fontsize=16, weight="bold")
save_figure("06_books_by_genre_lollipop.png")

# ---------------------------------------------------------
# OPTIONAL GRAPH 7: CORRELATION HEATMAP
# Only saved if at least two numeric columns have enough
# non-missing data and variation to calculate correlations.
# ---------------------------------------------------------
numeric_columns = ["price_gbp", "rating", "stock_quantity"]
numeric_data = df[numeric_columns].copy()

# Exclude columns that are entirely missing or constant.
usable_columns = [
    col for col in numeric_data.columns
    if numeric_data[col].notna().sum() >= 3
    and numeric_data[col].nunique(dropna=True) >= 2
]

if len(usable_columns) >= 2:
    corr_data = numeric_data[usable_columns].dropna()

    if len(corr_data) >= 3:
        print("Creating optional correlation heatmap...")
        correlation_matrix = corr_data.corr(numeric_only=True)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True
        )
        plt.title("Correlation Between Numeric Book Features",
                  fontsize=15, weight="bold")
        save_figure("07_correlation_heatmap.png")
    else:
        print(
            "Skipped correlation heatmap: fewer than 3 books have complete "
            "values for the selected numeric variables."
        )
else:
    print(
        "Skipped correlation heatmap: at least two numeric columns need "
        "three or more valid values and variation."
    )



# ---------------------------------------------------------
# 4. LINEAR REGRESSION: PREDICT BOOK PRICE
# Uses star rating and genre to predict price.
# Genre is converted into numeric indicator columns.
# ---------------------------------------------------------
print("\nLINEAR REGRESSION MODEL")
print("-" * 40)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    model_df = df[["price_gbp", "rating", "genre"]].copy()
    model_df["rating"] = pd.to_numeric(model_df["rating"], errors="coerce")
    model_df["genre"] = model_df["genre"].fillna("Unknown").astype(str)

    # Rating is required as a numeric predictor. Rows without a rating
    # cannot be used for this particular model.
    model_df = model_df.dropna(subset=["price_gbp", "rating"])
    model_df = model_df[model_df["rating"].between(1, 5)].copy()

    if len(model_df) < 10:
        print(
            "Regression skipped: fewer than 10 books have both a valid "
            "price and a valid rating."
        )
    else:
        # One-hot encode genre so the model can use genre as a predictor.
        X = pd.get_dummies(
            model_df[["rating", "genre"]],
            columns=["genre"],
            drop_first=True,
            dtype=int
        )
        y = model_df["price_gbp"]

        # Keep a separate test set to evaluate predictions on unseen rows.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        metrics = pd.DataFrame([{
            "model": "Linear Regression",
            "target": "price_gbp",
            "predictors": "rating and genre",
            "total_usable_books": len(model_df),
            "training_books": len(X_train),
            "testing_books": len(X_test),
            "MAE_GBP": round(mae, 3),
            "RMSE_GBP": round(rmse, 3),
            "R_squared": round(r2, 3)
        }])
        metrics.to_csv(BASE_DIR / "regression_metrics.csv", index=False)

        prediction_results = pd.DataFrame({
            "actual_price_gbp": y_test.to_numpy(),
            "predicted_price_gbp": predictions.round(2),
            "prediction_error_gbp": (y_test.to_numpy() - predictions).round(2)
        })
        prediction_results.to_csv(
            BASE_DIR / "books_price_predictions.csv", index=False
        )

        print(f"Books used for model: {len(model_df)}")
        print(f"Training records: {len(X_train)}")
        print(f"Testing records: {len(X_test)}")
        print(f"Mean Absolute Error (MAE): £{mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): £{rmse:.2f}")
        print(f"R-squared (R²): {r2:.3f}")
        print("Saved regression_metrics.csv")
        print("Saved books_price_predictions.csv")

        # Plot actual versus predicted prices. Points nearer the diagonal
        # represent more accurate predictions.
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, alpha=0.75)
        lower = min(float(y_test.min()), float(predictions.min()))
        upper = max(float(y_test.max()), float(predictions.max()))
        plt.plot([lower, upper], [lower, upper], linestyle="--")
        plt.xlabel("Actual Price (£)")
        plt.ylabel("Predicted Price (£)")
        plt.title("Linear Regression: Actual vs Predicted Book Prices",
                  fontsize=14, weight="bold")
        save_figure("08_regression_actual_vs_predicted.png")

        print(
            "Note: this is a baseline model. Its accuracy should be judged "
            "using the test-set metrics, and it does not prove that rating "
            "or genre causes a book to cost more."
        )

except ImportError:
    print(
        "Regression could not run because scikit-learn is not installed.\n"
        "Install it in the VS Code terminal with: py -m pip install scikit-learn"
    )


# ---------------------------------------------------------
# 5. FINISH
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print(f"Graphs folder: {GRAPHS_DIR}")
print(f"Books analysed: {len(df)}")
print("Open the 'graphs' folder to view your PNG charts.")
print("Sampling method: started at Young Adult and continued through subsequent genres until 220 unique books were collected.")
