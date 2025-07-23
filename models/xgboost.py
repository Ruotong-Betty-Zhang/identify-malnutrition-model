import pandas as pd
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier, plot_importance
import json
from sklearn.metrics import f1_score, recall_score, precision_score

class XGBoostModelTrainer:
    def __init__(self, seed=42, device='cuda'):
        self.seed = seed
        self.device = device
        self.model = None
        self.best_params = None

    def train(self, dataset_path: str, output_folder: str, parameters=None):
        # 确保输出文件夹存在
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # 设置随机种子
        np.random.seed(self.seed)

        # 加载数据
        df = pd.read_pickle(dataset_path)
        print(f"Dataset shape: {df.shape}")

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

        # 数据集拆分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.seed)
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Test set size: {X_test.shape[0]}")

        # 参数搜索空间（可替换为更大空间）
        params = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
        }

        if parameters:
            params = parameters

        # 定义模型
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            tree_method='hist',
            device=self.device,
            random_state=self.seed
        )

        # 网格搜索
        grid = GridSearchCV(xgb, params, cv=5, verbose=1, n_jobs=1)
        grid.fit(X_train, y_train)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        # 测试集预测与评估
        y_test_pred = self.model.predict(X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred))

        

        # 计算多分类指标（macro 是对每类分别计算再平均，不受类别不平衡影响）
        accuracy = accuracy_score(y_test, y_test_pred)
        f1_macro = f1_score(y_test, y_test_pred, average='macro')
        recall_macro = recall_score(y_test, y_test_pred, average='macro')  # Sensitivity
        precision_macro = precision_score(y_test, y_test_pred, average='macro')

        print("\n🔍 Multi-class Evaluation Metrics (macro average):")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Sensitivity    : {recall_macro:.4f}")      # i.e. macro recall
        print(f"F1-score       : {f1_macro:.4f}")
        print(f"Precision      : {precision_macro:.4f}")


        # 子文件夹命名
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        target_subfolder = os.path.join(output_folder, 'xgb_' + dataset_name)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        # 保存模型
        model_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # 保存最佳参数
        self.save_best_params(target_subfolder, dataset_name)

        
        # 保存特征重要性图
        importances = self.model.feature_importances_

        # 将重要性归一化为百分比（总和为100）
        importances_pct = importances / importances.sum() * 100

        # 获取 top 12 特征的索引和数值
        indices = np.argsort(importances_pct)[::-1]
        top_n = 8
        top_indices = indices[:top_n]
        top_features = X.columns[top_indices]
        top_importances = importances_pct[top_indices]

        # 画图
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(top_n), top_importances[::-1], align='center')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel("Feature Importance (%)")
        plt.title("Top 8 Most Important Features (xgb_" + dataset_name + ")")
        plt.tight_layout()

        # 在每个条形图旁边加上数值标签
        for i, (value, bar) in enumerate(zip(top_importances[::-1], bars)):
            plt.text(0.1, bar.get_y() + bar.get_height() / 2,
                    f'{value:.2f}%', va='center')

        # 保存图像
        plot_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_importance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Top 8 feature importance plot saved to {plot_path}")


        # # 可选：用 XGBoost 内置画法再画一张
        # plt.figure(figsize=(12, 8))
        # plot_importance(self.model, max_num_features=12, importance_type='gain')
        # plt.tight_layout()
        # builtin_plot_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_builtin_importance.png')
        # plt.savefig(builtin_plot_path, dpi=300)
        # plt.close()
        # print(f"XGBoost built-in feature importance plot saved to {builtin_plot_path}")

        return self.model

    def save_best_params(self, output_folder, dataset_name):
        if self.best_params is not None:
            params_path = os.path.join(output_folder, f'{dataset_name}_xgb_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best parameters saved to {params_path}")
        else:
            print("No best parameters to save.")