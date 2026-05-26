# Exercițiul 1: Accesați pagina http://quotes.toscrape.com și extrageți toate citatele,
# autorii și tag-urile de pe prima pagină. Salvați rezultatul într-un DataFrame pandas
# cu coloanele: 'quote', 'author', 'tags' (tags este o listă de șiruri).

import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "http://quotes.toscrape.com"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# === Your code starts here ===
quotes = []
quotes_items = soup.find_all("div", class_="quote")

for quote in quotes_items:
    text = quote.find("span", class_="text").text

    author = quote.find("small", class_="author").text

    tag_items = quote.find_all("a", class_="tag")
    tags = [tag.text for tag in tag_items]

    quotes.append({
        "text":text,
        "author": author,
        "tag": tags
    })

df = pd.DataFrame(quotes)
print(df)


print()
# === Your code ends here ===