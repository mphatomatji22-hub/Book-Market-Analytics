
"""
Book Market Analytics - Books to Scrape
Collect exactly 220 unique books, starting with Young Adult
and continuing through the website's category order.
"""

import re
import time
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

TARGET_BOOKS = 220
BASE_URL = "https://books.toscrape.com/"
OUTPUT_FILE = Path(__file__).resolve().parent / "books_raw.csv"

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (student data science project)"
})

ratings = {
    "One": 1,
    "Two": 2,
    "Three": 3,
    "Four": 4,
    "Five": 5
}


def get_soup(url):
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_book_details(card, genre):
    title = card.select_one("h3 a")["title"]
    book_url = urljoin(BASE_URL, card.select_one("h3 a")["href"])

    price_text = card.select_one(".price_color").get_text(strip=True)
    price_match = re.search(r"[\d.]+", price_text)
    price = float(price_match.group()) if price_match else None

    rating_element = card.select_one(".star-rating")
    rating_word = next(
        (word for word in rating_element["class"]
         if word in ratings),
        None
    )

    availability = card.select_one(".availability").get_text(" ", strip=True)
    quantity_match = re.search(r"\((\d+)\s+available\)", availability)

    stock_quantity = int(quantity_match.group(1)) if quantity_match else None
    stock_status = "In stock" if "In stock" in availability else "Out of stock"

    return {
        "title": title,
        "price_gbp": price,
        "rating_word": rating_word,
        "rating": ratings.get(rating_word),
        "stock_status": stock_status,
        "stock_quantity": stock_quantity,
        "genre": genre,
        "book_url": book_url
    }


def main():
    print("Connecting to Books to Scrape...")
    soup = get_soup(BASE_URL)

    # Use the category order displayed on the website.
    categories = []
    for link in soup.select(".side_categories ul li ul li a"):
        genre = link.get_text(strip=True)
        category_url = urljoin(BASE_URL, link["href"])

        if genre == "Young Adult":
            categories.append((genre, category_url))
            break

    if not categories:
        raise RuntimeError(
            "Young Adult category was not found on the website."
        )

    # Continue with every category after Young Adult.
    all_categories = []
    found_young_adult = False

    for link in soup.select(".side_categories ul li ul li a"):
        genre = link.get_text(strip=True)
        category_url = urljoin(BASE_URL, link["href"])

        if genre == "Young Adult":
            found_young_adult = True

        if found_young_adult:
            all_categories.append((genre, category_url))

    books = {}
    genre_counts = {}

    for genre, category_url in all_categories:
        if len(books) >= TARGET_BOOKS:
            break

        print(f"\nCollecting genre: {genre}")
        page_number = 1
        genre_counts.setdefault(genre, 0)

        while len(books) < TARGET_BOOKS:
            if page_number == 1:
                page_url = category_url
            else:
                page_url = urljoin(
                    category_url,
                    f"page-{page_number}.html"
                )

            try:
                page_soup = get_soup(page_url)
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    break
                raise

            cards = page_soup.select("article.product_pod")

            if not cards:
                break

            for card in cards:
                if len(books) >= TARGET_BOOKS:
                    break

                book = get_book_details(card, genre)
                title_key = book["title"].strip().casefold()

                if title_key in books:
                    # Keep track of additional genres for duplicates.
                    existing_genres = books[title_key]["genre"].split("; ")
                    if genre not in existing_genres:
                        books[title_key]["genre"] += f"; {genre}"
                    continue

                books[title_key] = book
                genre_counts[genre] += 1

            print(
                f"  Page {page_number}: "
                f"{len(books)} unique books collected so far"
            )

            if len(books) >= TARGET_BOOKS:
                break

            page_number += 1
            time.sleep(0.15)

        if len(books) >= TARGET_BOOKS:
            print(f"Target reached while collecting {genre}.")

    if len(books) < TARGET_BOOKS:
        print(
            f"\nWARNING: Only {len(books)} unique books were found "
            f"from Young Adult onward."
        )

    df = pd.DataFrame(list(books.values()))
    df.to_csv(OUTPUT_FILE, index=False)

    print("\n========== SCRAPING SUMMARY ==========")
    print(f"Unique books saved: {len(df)}")
    print(f"Target: {TARGET_BOOKS}")
    print(f"Output file: {OUTPUT_FILE}")

    print("\nBooks collected per genre:")
    for genre, count in genre_counts.items():
        if count:
            print(f"  {genre}: {count}")

    print("\nFirst five books:")
    print(df[["title", "price_gbp", "rating", "genre"]].head().to_string(index=False))

    if len(df) == TARGET_BOOKS:
        print("\nSUCCESS: Exactly 220 unique books collected.")
    else:
        print("\nThe dataset did not reach 220 books. Check the warning above.")


if __name__ == "__main__":
    main()
