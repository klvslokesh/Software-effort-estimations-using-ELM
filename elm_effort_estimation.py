import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# -----------------------------
# Create Synthetic Dataset
# -----------------------------

n_samples = 500

data = pd.DataFrame({
    "Project_Size": np.random.randint(50, 500, n_samples),
    "Team_Size": np.random.randint(3, 15, n_samples),
    "Experience_Level": np.random.randint(1, 10, n_samples)
})

# Effort formula that decreases with larger teams and more experience
effort_raw = (
    0.8 * data["Project_Size"]
    - 5 * data["Team_Size"]
    - 8 * data["Experience_Level"]
    + 100
    + np.random.normal(0, 10, n_samples)
)

data["Effort"] = np.maximum(0, effort_raw)

print("Dataset Sample\n")
print(data.head())

# -----------------------------
# Preprocessing
# -----------------------------

X = data.drop("Effort", axis=1)
y = data["Effort"]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Extreme Learning Machine
# -----------------------------

class ELM:

    def __init__(self, hidden=20):
        self.hidden = hidden

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):

        features = X.shape[1]

        self.W = np.random.randn(features, self.hidden)
        self.b = np.random.randn(self.hidden)

        H = self.sigmoid(X @ self.W + self.b)

        self.beta = np.linalg.pinv(H) @ y

    def predict(self, X):

        H = self.sigmoid(X @ self.W + self.b)

        return H @ self.beta


model = ELM(hidden=20)

model.fit(X_train, y_train)

# -----------------------------
# Testing
# -----------------------------

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)

print("\nModel MAE:", mae)

# -----------------------------
# Visualization
# -----------------------------

plt.scatter(y_test, predictions)

plt.xlabel("Actual Effort")
plt.ylabel("Predicted Effort")

plt.title("ELM Effort Prediction")

plt.show()