# Exercițiul 3: Folosiți setul de date despre prețuri imobiliare din California pentru a
# antrena un model de regresie Ridge. Tratați valorile lipsă, folosiți GridSearchCV
# pentru a căuta parametrul optim alpha și afișați eroarea MSE pe setul de test.

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

# === START ===
print(df.isna().sum())
df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# X = df.iloc[:, :-1]
# y = df.iloc[:, -1]
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

X = pd.get_dummies(X, columns=["ocean_proximity"], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

ridge = Ridge(alpha=1.0)

param_grid = {
    "alpha": [0.01, 0.1, 1, 10, 100]
}

grid_search = GridSearchCV(
    estimator=ridge,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
    #daca cere r, r2, neg_mean_absolute_error
)
# model = Lasso(max_iter=10000)
#
# param_grid = {
#     "alpha": [0.01, 0.1, 1, 10, 100]
# }
#
# grid_search = GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,
#     scoring="neg_mean_squared_error"
# )

grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred_test = best_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test)
print(f"Test Set MSE: {mse:.3f}")
# === END ===