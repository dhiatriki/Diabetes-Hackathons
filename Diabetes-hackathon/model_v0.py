# ================================
# MODEL V0 — HORRIBLE FIRST MODEL
# ================================

# 1️⃣ Import libraries
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 2️⃣ Load dataset
df = pd.read_csv("diabetes.csv")

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())


# 3️⃣ Separate inputs (X) and target (y)
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]


# 4️⃣ Train-test split (BAD ON PURPOSE)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 5️⃣ Create a very basic model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 6️⃣ Make predictions
y_pred = model.predict(X_test)


# 7️⃣ Evaluate the model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
