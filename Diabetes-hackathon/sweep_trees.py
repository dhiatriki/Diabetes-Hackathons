
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("diabetes.csv")
df.drop_duplicates(inplace=True)
df["Diabetes_012"] = df["Diabetes_012"].replace({1: 0, 2: 1})

X = df.drop("Diabetes_012", axis=1)
y = df["Diabetes_012"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Depth | Train Acc | Test Acc")
for depth in [6, 8, 10, 12, 14, 16, 20]:
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"{depth:5d} | {train_acc:.4f}    | {test_acc:.4f}")
