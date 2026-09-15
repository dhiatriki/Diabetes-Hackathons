# ==========================================
# MODEL V4 — OPTIMIZED RANDOM FOREST
# ==========================================
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

start_time = time.time()

# 1️⃣ Load & Clean
df = pd.read_csv("diabetes.csv")
df.drop_duplicates(inplace=True)
df["Diabetes_012"] = df["Diabetes_012"].replace({1: 0, 2: 1})

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# 2️⃣ Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3️⃣ Train (Tuned for Accuracy)
model = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=5, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 4️⃣ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n================================")
print(f"MODEL V4: RANDOM FOREST")
print(f"Accuracy: {acc:.4f}")
print(f"Execution Time: {time.time() - start_time:.2f} seconds")
print(f"================================\n")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
