import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# Load dataset
data = pd.read_csv("heart.csv")

# Check and handle missing values
print("\n🔍 Checking for missing values:")
print(data.isnull().sum())
data.fillna(data.median(numeric_only=True), inplace=True)

# Feature and label split
X = data.drop("condition", axis=1)
y = data["condition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ✅ Train Decision Tree Classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)

print("\n✅ Decision Tree Evaluation:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Decision Tree - Confusion Matrix")
os.makedirs("images", exist_ok=True)
plt.savefig("images/confusion_matrix_dt.png")
plt.show()

# Plot and save the decision tree
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree Structure")
plt.show()

# Save model
os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\n✅ Decision Tree model trained and saved as model/model.pkl")
