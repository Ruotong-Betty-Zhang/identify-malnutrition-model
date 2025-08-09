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
from sklearn.utils.class_weight import compute_sample_weight
import shap

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
        device_setting = 'cpu'
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            tree_method='hist',
            device=device_setting,
            random_state=self.seed
        )

        # 根据任务类型确定 scoring
        scoring_method = 'recall_macro' if len(np.unique(y)) > 2 else 'recall'

        min_class_count = y.value_counts().min()
        cv_folds = min(5, min_class_count)

        # 网格搜索
        grid = GridSearchCV(
            xgb,
            param_grid=params,
            scoring=scoring_method,
            cv=cv_folds,
            verbose=1,
            n_jobs=1
        )

        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        grid.fit(X_train, y_train, sample_weight=sample_weights)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        # 测试集预测与评估
        y_test_pred = self.model.predict(X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred, zero_division=0))

        

        # 计算多分类指标（macro 是对每类分别计算再平均，不受类别不平衡影响）
        accuracy = accuracy_score(y_test, y_test_pred)
        f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)  # Sensitivity
        precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)

        print("\n🔍 Multi-class Evaluation Metrics (macro average):")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Sensitivity    : {recall_macro:.4f}")      # i.e. macro recall
        print(f"F1-score       : {f1_macro:.4f}")
        print(f"Precision      : {precision_macro:.4f}")

        # 保存性能指标为 JSON
        performance = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(f1_macro, 4),
            "macro_recall": round(recall_macro, 4),
            "macro_precision": round(precision_macro, 4),
            "confusion_matrix": confusion_matrix(y_test, y_test_pred).tolist(),
            "classification_report": classification_report(y_test, y_test_pred, zero_division=0, output_dict=True)
        }

        # 子文件夹命名
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        target_subfolder = os.path.join(output_folder, 'xgb_' + dataset_name)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        performance_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_performance.json')
        with open(performance_path, 'w') as f:
            json.dump(performance, f, indent=4)
        print(f"Model performance saved to {performance_path}")

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
        top_n = 12
        top_indices = indices[:top_n]
        top_features = X.columns[top_indices]
        top_importances = importances_pct[top_indices]
        # Print the top 12 features in one line
        # top_features_str = ', '.join([f"{feat} ({imp:.2f}%)" for feat, imp in zip(top_features, top_importances)])
        top_features_str = ', '.join([f"{feat}" for feat, imp in zip(top_features, top_importances)])
        print(f"Top 12 features: {top_features_str}")

        # 画图
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(top_n), top_importances[::-1], align='center')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel("Feature Importance (%)")
        plt.title("Top " + str(top_n) + " Most Important Features (xgb_" + dataset_name + ")")
        plt.tight_layout()

        # 在每个条形图旁边加上白色数值标签
        for i, (value, bar) in enumerate(zip(top_importances[::-1], bars)):
            plt.text(0.1, bar.get_y() + bar.get_height() / 2,
                    f'{value:.2f}%', va='center', color='white')

        # 保存图像
        plot_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_importance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Top 12 feature importance plot saved to {plot_path}")

        # Print the top 12 features in one line
        # top_features_str = ', '.join([f"{feat} ({imp:.2f}%)" for feat, imp in zip(top_features, top_importances)])
        top_features_str = ', '.join([f"{feat}" for feat, imp in zip(top_features, top_importances)])
        print(f"Top 12 features: {top_features_str}")
        top_feature_path = os.path.join(target_subfolder, f'{dataset_name}_top12_features.json')
        with open(top_feature_path, 'w') as f:
            json.dump(top_features.tolist(), f, indent=2)
        print(f"Top 12 features saved to {top_feature_path}")


        # # 可选：用 XGBoost 内置画法再画一张
        # plt.figure(figsize=(12, 8))
        # plot_importance(self.model, max_num_features=12, importance_type='gain')
        # plt.tight_layout()
        # builtin_plot_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_builtin_importance.png')
        # plt.savefig(builtin_plot_path, dpi=300)
        # plt.close()
        # print(f"XGBoost built-in feature importance plot saved to {builtin_plot_path}")



        # 拟合好模型后
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X.values)

        print("Type of shap_values:", type(shap_values))
        if isinstance(shap_values, list):
            print("shap_values is a list with length:", len(shap_values))
            for i, val in enumerate(shap_values):
                print(f"shap_values[{i}] shape:", np.array(val).shape)
        else:
            print("shap_values shape:", np.array(shap_values).shape)

        # 可视化总体影响
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        summary_beeswarm_path = os.path.join(target_subfolder, f"{dataset_name}_shap_summary_beeswarm.png")
        plt.savefig(summary_beeswarm_path, dpi=300)
        plt.close()
        print(f"SHAP beeswarm plot saved to {summary_beeswarm_path}")

        plt.figure()
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.tight_layout()
        summary_bar_path = os.path.join(target_subfolder, f"{dataset_name}_shap_summary_bar.png")
        plt.savefig(summary_bar_path, dpi=300)
        plt.close()
        print(f"SHAP bar plot saved to {summary_bar_path}")

        return self.model

    def save_best_params(self, output_folder, dataset_name):
        if self.best_params is not None:
            params_path = os.path.join(output_folder, f'{dataset_name}_xgb_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best parameters saved to {params_path}")
        else:
            print("No best parameters to save.")