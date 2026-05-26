# Exercițiul 2: Creați un model de clasificare text pentru categoriile 'alt.atheism'
# și 'soc.religion.christian' din 20 Newsgroups. Folosiți CountVectorizer, antrenați
# un model Naive Bayes și afișați raportul de clasificare pe setul de test.

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

categories = ['alt.atheism', 'soc.religion.christian']
data = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

# === Your code starts here ===

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

vectorizer = CountVectorizer(stop_words='english')

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

y_pred_nb = model_nb.predict(X_test)

print("\nClassification Report (precision, recall, F1-score):")
print(classification_report(
    y_test,
    y_pred_nb,
    target_names=data.target_names
))

# === Your code ends here ===