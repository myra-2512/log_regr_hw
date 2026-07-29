import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

x = np.arange(10).reshape(-1, 1)
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

unique_classes = np.unique(y)
num_classes = len(unique_classes)

print("--- Data Analysis ---")
print(f"Unique classes found in target variable: {unique_classes}")

if num_classes == 2:
    print(f"Classification Type: Binary Classification ({num_classes} classes)")
elif num_classes > 2:
    print(f"Classification Type: Multi-class Classification ({num_classes} classes)")
else:
    print("Error: Invalid number of classes.")

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

model = LogisticRegression(multi_class='auto', solver='lbfgs')
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n--- Model Evaluation ---")
print(f"Accuracy Score: {accuracy_score(y_test, predictions):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

