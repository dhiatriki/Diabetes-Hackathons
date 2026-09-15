# ================================
# MODEL V1 — DATA CLEANING & SCALING
# ================================

# 1️⃣ Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load dataset (Story: "I realized the problem wasn’t the algorithm — it was the data.")
df = pd.read_csv("diabetes.csv")

print("Original Dataset Shape:", df.shape)

# 3️⃣ DATA CLEANING (PHASE 1)

# A. Remove duplicates
# Many real-world datasets have duplicates that bias the model.
initial_rows = df.shape[0]
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {initial_rows - df.shape[0]}")
print("New Dataset Shape:", df.shape)

# B. Handle missing values
# Checking for nulls...
missing_values = df.isnull().sum().sum()
print(f"Missing values found: {missing_values}")
if missing_values > 0:
    df.dropna(inplace=True)
    print("Dropped missing values.")

# C. Check Class Imbalance
# The user noticed "diabetes != diabetes"
# Insight: "Pre-diabetes" (1) is a rare class and statistically closer to "No diabetes" (0) in this dataset.
# The model fails to predict it (Recall=0). Including it just adds noise.
# STRATEGY: Merge Class 1 (Pre-diabetes) into Class 0 (No diabetes).
print("\nClass Distribution (Before Fix):")
print(df["Diabetes_012"].value_counts(normalize=True) * 100)

df["Diabetes_012"] = df["Diabetes_012"].replace({1: 0})
# Now 0 = No Diabetes / Pre-diabetes, 2 = Diabetes
# We can optionally map 2 -> 1 to make it a standard binary classification, but let's keep 0 and 2 for clarity or map 2->1.
# Let's map 2 -> 1 for cleaner binary output.
df["Diabetes_012"] = df["Diabetes_012"].replace({2: 1})

print("Class Distribution (After Merging 1->0 and 2->1):")
print(df["Diabetes_012"].value_counts(normalize=True) * 100)
# This directly addresses the imbalance and simplifies the problem.

# D. Separate Numerical vs Categorical
# We scale numerical features for better model performance (especially Logistic Regression).
feature_cols = df.columns.drop("Diabetes_012")
numerical_cols = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]

X = df[feature_cols]
y = df["Diabetes_012"]

# Scaling numerical features
scaler = StandardScaler()
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

print("\nFeatures scaled (BMI, GenHlth, MentHlth, PhysHlth, Age, Education, Income).")


# 4️⃣ Train-test split (With Stratify)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5️⃣ Create a better model
# Increased max_iter to ensure convergence
# Added class_weight='balanced' to further help with the 0 vs 1 imbalance if needed,
# but since we merged, the imbalance is less severe (85/15 vs 84/2/14).
# We'll just use standard LogReg to maximize Accuracy as requested.
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)


# 6️⃣ Make predictions
y_pred = model.predict(X_test)


# 7️⃣ Evaluate the model
acc = accuracy_score(y_test, y_pred)
print(f"\nModel V1 Accuracy: {acc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Simple check for the "3-5% gain" goal
# (Previous V0 was around ~71% or whatever the baseline was on raw data)
