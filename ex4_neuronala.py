# Exercițiul 4: Implementați o rețea neuronală simplă folosind PyTorch pentru a clasifica
# datele din setul Iris. Antrenați modelul pentru 100 de epoci și afișați acuratețea pe
# setul de test.

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# === START ===
# 1. Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 3)
)

# print(X.shape)
# print(set(y))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 100

for epoch in range(epochs):
    model.train()

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y_train).float().mean()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy.item() * 100:.2f}%")

model.eval()

with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = torch.argmax(test_outputs, dim=1)
    test_accuracy = (test_predictions == y_test).float().mean()

print(f"Test Accuracy: {test_accuracy.item() * 100:.2f}%")
# === END ===