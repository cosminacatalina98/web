# Exercițiul 1: Extrageți titlul, prețul și disponibilitatea cărților de pe pagina principală
# a site-ului http://books.toscrape.com. Normalizați prețurile (float) și disponibilitatea
# (1 pentru "In stock", 0 altfel), apoi stocați-le într-un DataFrame pandas.

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://books.toscrape.com/"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# === START ===
books = []
book_items = soup.find_all("article", class_="product_pod")

for book in book_items:
    title = book.find("h3").find("a")["title"]

    price_text = book.find("p", class_="price_color").text

    # price_clean = "".join(char for char in price_text if char.isdigit() or char == ".")
    price_clean = price_text[2:]
    price = float(price_clean)

    availability_text = book.find("p", class_="availability").text.strip()

    if "In stock" in availability_text:
        availability = 1
    else:
        availability = 0

    rating = book.find("p", class_="star-rating")
    rating_text = rating["class"][1]

    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    rating_num = rating_map[rating_text]

    books.append({
        "title": title,
        "price": price,
        "availability": availability,
        "rating": rating_num
    })
df = pd.DataFrame(books)
print(df)

# === END ===

# REGULI
# TEXTUL DINTRE TAGURI
# <h3 class="product-title">Laptop Lenovo</h3>
# title = item.find("h3", class_="product-title").get_text(strip=True)
#
# TEXTUL DIN TAGURI
# <a href="product.html">Details</a>
# <img src="image.jpg" alt="Laptop Lenovo">
#
# link = item.find("a")["href"]
# image = item.find("img")["src"]
# alt_text = item.find("img")["alt"]
#
# CLASA
# <p class="price_color">£51.77</p>
# price = item.find("p", class_="price_color").get_text(strip=True)
#
# find vs find_all -> FIND-1 element; FIND_ALL -mai multe
#
# <article class="product_pod">
#    ...
# </article>
# books = soup.find_all("article", class_="product_pod")
#
# data = []
