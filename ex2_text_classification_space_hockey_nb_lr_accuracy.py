# Exercițiul 2: Creați un model de clasificare text folosind un subset din 20 Newsgroups,
# cu două clase: 'sci.space' și 'rec.sport.hockey'. Vectorizați textul, antrenați modelul și
# calculați acuratețea pe setul de test.


from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === START ===
categories = ['sci.space', 'rec.sport.hockey']

data = fetch_20newsgroups(
    subset='all',
    categories=categories,
    shuffle=True,
    random_state=42
)

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

accuracy_nb = accuracy_score(y_test, y_pred_nb)

print("Accuracy Naive Bayes:", accuracy_nb)

model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)

y_pred_lr = model_lr.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)

print("Accuracy Logistic Regression:", accuracy_lr)
# === END ===