# Decision Tree Implementation and Visualization Project
# This script is structured like a Jupyter Notebook for easy conversion.
# You can copy this into a .ipynb notebook if required.

# =====================================
# Cell 1: Import Libraries
# =====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================
# Cell 2: Load Dataset (Iris Dataset)
# =====================================
# You may replace this with your own dataset

data = load_iris()
X = data.data
y = data.target

feature_names = data.feature_names
target_names = data.target_names

# Convert to DataFrame (for better visualization)
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

df.head()

# =====================================
# Cell 3: Exploratory Data Analysis
# =====================================
print("Dataset Shape:", df.shape)
print("\nClass Distribution:\n", df['target'].value_counts())
print("\nSummary Statistics:\n", df.describe())

# =====================================
# Cell 4: Train-Test Split
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])

# =====================================
# Cell 5: Train Decision Tree Model
# =====================================
model = DecisionTreeClassifier(
    criterion="gini",      # Can also use "entropy"
    max_depth=3,           # Control overfitting
    random_state=42
)

model.fit(X_train, y_train)

# =====================================
# Cell 6: Model Prediction
# =====================================
y_pred = model.predict(X_test)

# =====================================
# Cell 7: Model Evaluation
# =====================================
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=target_names))

print("Confusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# =====================================
# Cell 8: Visualize Decision Tree
# =====================================
plt.figure(figsize=(20, 12))
plot_tree(
    model,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization")
plt.show()

# =====================================
# Cell 9: Feature Importance
# =====================================
importance = model.feature_importances_

feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values(by='Importance', ascending=False)

print("Feature Importance:\n")
print(feature_importance_df)

# =====================================
# Cell 10: Analysis and Observations
# =====================================
"""
Analysis:

1. The Decision Tree was trained using the Iris dataset.
2. The model achieved good accuracy due to clear class separation.
3. Limiting max_depth prevents overfitting.
4. Visualization helps understand decision rules.
5. Feature importance shows which attributes affect predictions most.

You can improve the model by:
- Tuning hyperparameters (max_depth, min_samples_split)
- Using cross-validation
- Trying different datasets

Conclusion:
Decision Trees are interpretable models suitable for classification and prediction tasks.
They provide clear decision rules and visual explanations.
"""
