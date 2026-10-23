Book Market Analytics – Books to Scrape

## 1. Project Overview
This project was completed for the Data Science 500 module. It analyses book data collected from the Books to Scrape website.
The project uses Python to collect, clean, explore and analyse book information. It also uses data visualisation and linear regression to investigate book prices, ratings and genres.

## 2. Project Objectives
- Collect at least 200 book records using web scraping.
- Extract book titles, prices, ratings, stock availability and genres.
- Clean and prepare the collected data for analysis.
- Visualise book prices and compare average prices across genres.
- Build and evaluate a linear regression model to estimate book prices.

## 3. Technologies Used
- Python
- BeautifulSoup
- Requests
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook

## 4. Project Files
- `scrape_books.py` – collects book information from the website.
- `cleaning_book_data.py` – cleans and prepares the collected data.
- `book_analysis.py` – performs exploratory analysis, generates charts and evaluates the regression model.
- `Book_Market_Analytics_Notebook.ipynb` – documents the data science workflow.
- `books_raw.csv` – the collected raw dataset.
- `books_cleaned.csv` – the cleaned dataset.
- `graphs/` – contains the generated visualisations.
- `summary_statistics.csv` – contains descriptive statistics.
- `regression_metrics.csv` – contains the regression evaluation results.
- `books_price_predictions.csv` – contains model predictions.

## 5. How to Run the Project
1. Install Python and the required libraries.
2. Open the project folder in Visual Studio Code or another Python development environment.
3. Install the required packages using:

   `py -m pip install requests beautifulsoup4 pandas matplotlib seaborn scikit-learn jupyter`

4. Run `scrape_books.py` to collect the data.
5. Run `cleaning_book_data.py` to clean the dataset.
6. Run `book_analysis.py` to generate the visualisations and regression results.

Run the scripts from the project folder so that the input and output files can be found. The scraping script requires an internet connection.

## 6. Results Summary
The final dataset contains 220 books across 17 genres. The average book price was £34.41, and the average rating was 3.08 out of 5.
The linear regression model achieved a Mean Absolute Error of £14.31, a Root Mean Squared Error of £16.48 and an R-squared value of -0.0651. These results indicate that the model did not predict prices reliably on the test dataset.

## 7. Data Source
Books to Scrape: https://books.toscrape.com/
