import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

class RandomForestModelTrainer:
    def __init__(self, seed=42):
        self.seed = seed
        self.model = None

    def train(self, dataset_path: str, output_folder: str):
        # 确保输出路径存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 设置随机种子
        np.random.seed(self.seed)

        # 加载数据
        df = pd.read_pickle(dataset_path)
        print(f"Malnutrition DataFrame shape: {df.shape}")

        # 删除无关列
        df = df.drop(columns=['IDno', 'Assessment_Date'])

        # 划分特征和标签
        if 'MAL' in dataset_path:
            X = df.drop(columns=["Malnutrition"])
            y = df["Malnutrition"]
        elif 'CAP' in dataset_path:
            X = df.drop(columns=["CAP_Nutrition"])
            y = df["CAP_Nutrition"]
        else:
            print("Unknown dataset format. Please provide a valid dataset start with CAP or MAL.")
            return None

        # 划分训练和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)

        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # 随机森林参数网格
        params = {
            'n_estimators': [100, 200, 300],
            'ccp_alpha': np.linspace(0.0, 0.05, 5),
            'max_depth': [3, 5, 10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }

        # 网格搜索 + 交叉验证
        grid = GridSearchCV(RandomForestClassifier(random_state=self.seed), params, cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)

        print("Best parameters found: ", grid.best_params_)

        # 使用最优模型
        self.model = grid.best_estimator_

        # 测试集预测与评估
        y_test_pred = self.model.predict(X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred))

        # 创建输出文件夹
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        target_folder = os.path.join(output_folder, 'rf_' + dataset_name)
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        
        # 保存模型
        model_path = os.path.join(target_folder, 'rf_' + dataset_name + '_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # 保存特征重要性图
        importances = self.model.feature_importances_

        # 将重要性归一化为百分比（总和为100）
        importances_pct = importances / importances.sum() * 100

        # 获取 top 12 特征的索引和数值
        indices = np.argsort(importances_pct)[::-1]
        top_n = 12
        top_indices = indices[:top_n]
        top_features = X.columns[top_indices]
        top_importances = importances_pct[top_indices]

        # 画图
        plt.figure(figsize=(10, 6))
        plt.barh(range(top_n), top_importances[::-1], align='center')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel("Feature Importance (%)")
        plt.title("Top 12 Most Important Features (XGBoost)")
        plt.tight_layout()

        # 保存图像
        plot_path = os.path.join(target_folder, f'{dataset_name}_xgb_importance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Top 12 feature importance plot saved to {plot_path}")

        return self.model
