import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import f1_score, make_scorer

dataset_folder = "./datasets/"
target_folder = "./outputs/"
print(XGBClassifier().get_params())

if not os.path.exists(target_folder):
    os.makedirs(target_folder)

np.random.seed(42)
seed = 42

# Load data
cap_df = pd.read_pickle(dataset_folder + 'mal_long_data.pkl')
print(f"CAP DataFrame shape: {cap_df.shape}")
cap_df['Malnutrition'] = cap_df['Malnutrition'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
cap_df = cap_df.drop(columns=['IDno', 'Assessment_Date'])

# Assign X and y
X = cap_df.drop(columns=["Malnutrition"])
y = cap_df["Malnutrition"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# neg, pos = np.bincount(y_train)
# scale = neg / pos
# print(f"Scale for class imbalance: {scale}")
# scale = 50

# Grid search
params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.7, 1],
    'scale_pos_weight': range(20, 120, 2),
}

# Best parameters for XGBoost
params = {
    'max_depth': [5],
    'learning_rate': [0.01],
    'n_estimators': [50],
    'subsample': [1],
    'scale_pos_weight': [50],
}

# Define the XGBoost classifier
xgb = XGBClassifier(
    eval_metric='logloss',
    tree_method='hist',
    device='cuda',
    # scale_pos_weight=scale,
    random_state=seed
)

# 创建自定义评分函数，监控类别1的F1-score
f1_score_class_1 = make_scorer(f1_score, pos_label=1)

# 更新GridSearchCV中的scoring参数
grid = GridSearchCV(xgb, params, cv=5, verbose=1, n_jobs=-1, scoring=f1_score_class_1)
grid.fit(X_train, y_train)

print("Best parameters found: ", grid.best_params_)
clf = grid.best_estimator_

# Evaluate
y_test_pred = clf.predict(X_test)
print("Test Results:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Feature importance
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 12
top_indices = indices[:top_n]
top_features = X.columns[top_indices]
top_importances = importances[top_indices]

# plt.figure(figsize=(12, 6))
# plt.barh(range(top_n), top_importances[::-1], align='center')
# plt.yticks(range(top_n), top_features[::-1])
# plt.xlabel("Feature Importance")
# plt.title("Top 12 Most Important Features (XGBoost)")
# plt.tight_layout()
# plt.savefig(target_folder + "mal_xgb_top_12_features.png", dpi=300)
# plt.close()
# print("Saved top 12 feature importance plot as mal_xgb_top_12_features.png")

# 模型保存
model_path = os.path.join(target_folder, 'mal_xgboost_model_long.pkl')
joblib.dump(clf, model_path)
print(f"Model saved to {model_path}")

# 可选：绘制完整模型重要性图（内置）
plt.figure(figsize=(12, 8))
plot_importance(clf, max_num_features=12, importance_type='gain')
plt.tight_layout()
plt.savefig(target_folder + "mal_xgboost_plot_importance_long.png", dpi=300)
plt.close()
print("Saved as mal_xgboost_plot_importance_long.png")
