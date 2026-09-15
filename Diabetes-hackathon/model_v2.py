# ==========================================
# MODEL V2 — SMARTER BASELINE (PREPROCESSING)
# ==========================================

# Story: "I fixed the fundamentals before jumping to complex models."
# Lesson: Preprocessing alone can outperform bad modeling.

# 1️⃣ Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2️⃣ Load dataset
df = pd.read_csv("diabetes.csv")
print(f"Original Data Shape: {df.shape}")

# 3️⃣ DATA CLEANING (Carried over from Phase 1)
# Remove duplicates
df.drop_duplicates(inplace=True)

# Handle Class Imbalance (Merge Class 1 'Pre-diabetes' into Class 0 'No Diabetes')
# Why? Because Class 1 was noise that killed Model V0's performance.
df["Diabetes_012"] = df["Diabetes_012"].replace({1: 0, 2: 1})
# Target is now: 0 = No Diabetes/Pre, 1 = Diabetes

# 4️⃣ BITTER PREPROCESSING (PHASE 2 UPGRADE)
# We define exactly how each column should be treated.

# A. Numeric Features -> STANDARDIZE
# These are continuous or count-based. Scaling helps the algorithm converge and weight them fairly.
numeric_features = ["BMI", "MentHlth", "PhysHlth"] 

# B. Categorical Features -> ONE-HOT ENCODE
# These have levels (e.g., GenHlth 1-5, Age 1-13). 
# Treating them as just numbers (1,2,3) forces a linear relationship. 
# One-Hot Encoding lets the model treat "Age Group 10" differently from "Age Group 2".
categorical_features = ["GenHlth", "Age", "Education", "Income"]

# The rest are already binary (0/1), so we keep them as is? 
# Actually, let's process specifically.
binary_features = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke", 
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies", 
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
]

# We need to construct our X and y
X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

# Check if we missed any columns
all_cols = numeric_features + categorical_features + binary_features
print(f"Features used: {len(all_cols)} / {len(X.columns)}")

# 5️⃣ BUILD PREPROCESSING PIPELINE
# Using ColumnTransformer is the "Professional" way to do this.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_features), # drop='first' avoids dummy trap
        ("bin", "passthrough", binary_features)
    ]
)

# Apply transformations
print("\nApplying One-Hot Encoding and Standardization...")
X_processed = preprocessor.fit_transform(X)
print(f"New Feature Count after OHE: {X_processed.shape[1]}")
# (Expect increase due to OHE of Age/Income/etc.)

# 6️⃣ STRATIFIED TRAIN-TEST SPLIT
# Crucial for imbalanced data. Keeps the % of diabetics same in train and test.
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 7️⃣ TRAIN MODEL (Logistic Regression)
# Same model as before, but the DATA is now "feature engineered".
print("\nTraining Logistic Regression on refined data...")
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train, y_train)

# 8️⃣ EVALUATE
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\nModel V2 Accuracy: {acc:.4f}")
print("(Target: ~0.74 - 0.75)")

print("\nConfusion Matrix:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualizing the Recall improvement
tn, fp, fn, tp = conf_matrix.ravel()
recall = tp / (tp + fn)
print(f"Recall (Sensitivity) for Diabetes Class: {recall:.2f}")
