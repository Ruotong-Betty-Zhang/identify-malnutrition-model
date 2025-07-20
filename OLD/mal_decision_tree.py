import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib  # 用于保存模型
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

dataset_folder = "./datasets/"
target_folder = "./outputs/"
# Check if the target folder exists, if not, create it
if not os.path.exists(target_folder):
    os.makedirs(target_folder)

# Set random seed for reproducibility
np.random.seed(42)
seed = 42

# Read mal_df
mal_df = pd.read_pickle(dataset_folder + 'mal_data.pkl')
print(f"Malnutrition DataFrame shape: {mal_df.shape}")

# Check the Malnutrition column, set [0, 1, 2] to 0, and [3, 4, 5] to 1
mal_df['Malnutrition'] = mal_df['Malnutrition'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
mal_df = mal_df.drop(columns=['IDno', 'Assessment_Date'])

# Seperate the features and the target
X = mal_df.drop(columns=["Malnutrition"])
y = mal_df["Malnutrition"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create a Decision Tree Classifier
params = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
}

grid = GridSearchCV(DecisionTreeClassifier(random_state=seed), params, cv=5)
grid.fit(X_train, y_train)
print("Best parameters found: ", grid.best_params_)

clf = grid.best_estimator_

y_test_pred = clf.predict(X_test)

print("Test Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Plot the top 12 feature importances
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 12
top_indices = indices[:top_n]
top_features = X.columns[top_indices]
top_importances = importances[top_indices]

# Create a bar plot for the top 12 feature importances
plt.figure(figsize=(12, 6))
plt.barh(range(top_n), top_importances[::-1], align='center')
plt.yticks(range(top_n), top_features[::-1])
plt.xlabel("Feature Importance")
plt.title("Top 12 Most Important Features")
plt.tight_layout()
plt.savefig(target_folder + "maldt_top_12_features.png", dpi=300)
plt.close()
print("Saved top 12 feature importance plot as maldt_top_12_features.png")

# Save the model
model_path = os.path.join(target_folder, 'mal_decision_tree_model.pkl')

joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

plt.figure(figsize=(40, 20))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=["0", "1"], fontsize=10)
plt.title("Decision Tree", fontsize=16)
plt.savefig(target_folder + "mal_decision_tree.png", dpi=300, bbox_inches='tight')
plt.close()
print("Saved as mal_decision_tree.png")