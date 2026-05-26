# Exercițiul 3: Descărcați setul de date despre automobile de la URL-ul de mai jos.
# Folosiți regresie polinomială (gradul 2) pentru a prezice consumul de combustibil
# pe baza puterii motorului. Afișați R² pe setul de test.

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

url = "https://raw.githubusercontent.com/murpi/wilddata/master/quests/cars.csv"
df = pd.read_csv(url)


# === Your code starts here ===
# Folosiți coloanele 'Horsepower' (predictor) și 'MPG' (target)
# === Your code starts here ===

X = df[["hp"]]
y = df["mpg"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)

y_pred = model.predict(X_test_poly)

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

print("Coeficient hp:", model.coef_[1])
print("Coeficient hp^2:", model.coef_[2])
print("Intercept:", model.intercept_)
# === Your code ends here ===

# pentru grafic: sortăm valorile hp ca să iasă curba frumoasă
X_plot = np.linspace(X["hp"].min(), X["hp"].max(), 100).reshape(-1, 1)

# transformăm X_plot la fel ca train/test
X_plot_poly = poly.transform(X_plot)

# modelul prezice mpg pentru valorile din X_plot
y_plot = model.predict(X_plot_poly)

# desenăm punctele reale
plt.scatter(X, y, label="Date reale")

# desenăm curba modelului
plt.plot(X_plot, y_plot, label="Model polinomial grad 2")

plt.xlabel("Horsepower")
plt.ylabel("MPG")
plt.title("Regresie polinomială: hp vs mpg")
plt.legend()
plt.show()