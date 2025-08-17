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
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.target_subfolder = None
        self.dataset_name = None

    def train(self, dataset_path: str, output_folder: str, parameters=None):
        # Main training pipeline
        self._setup_output_folder(dataset_path, output_folder)
        df = self._load_and_preprocess_data(dataset_path)
        self._split_data(df, dataset_path)
        self._perform_grid_search(parameters)
        self._evaluate_model()
        self._save_results()
        self._generate_feature_importance_plots()
        self._generate_shap_plots()
        return self.model

    def _setup_output_folder(self, dataset_path, output_folder):
        # Create output folder structure
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        self.dataset_name = os.path.basename(dataset_path).split('.')[0]
        self.target_subfolder = os.path.join(output_folder, 'xgb_' + self.dataset_name)
        if not os.path.exists(self.target_subfolder):
            os.makedirs(self.target_subfolder)

    def _load_and_preprocess_data(self, dataset_path):
        # Load and preprocess the dataset
        np.random.seed(self.seed)
        df = pd.read_pickle(dataset_path)
        print(f"Dataset shape: {df.shape}")
        return df.drop(columns=['IDno', 'Assessment_Date'])

    def _split_data(self, df, dataset_path):
        # Split data into features and target, then train-test split
        if 'MAL' in dataset_path:
            X = df.drop(columns=["Malnutrition"])
            y = df["Malnutrition"]
        elif 'CAP' in dataset_path:
            X = df.drop(columns=["CAP_Nutrition"])
            y = df["CAP_Nutrition"]
        else:
            raise ValueError("Unknown dataset format. Please provide a valid dataset start with CAP or MAL.")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")

    def _perform_grid_search(self, parameters):
        # Perform hyperparameter tuning with GridSearchCV
        params = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
        } if parameters is None else parameters

        device_setting = 'cpu'
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            tree_method='hist',
            device=device_setting,
            random_state=self.seed
        )

        scoring_method = 'recall_macro' if len(np.unique(self.y_train)) > 2 else 'recall'
        min_class_count = self.y_train.value_counts().min()
        cv_folds = min(5, min_class_count)

        grid = GridSearchCV(
            xgb,
            param_grid=params,
            scoring=scoring_method,
            cv=cv_folds,
            verbose=1,
            n_jobs=1
        )

        sample_weights = compute_sample_weight(class_weight='balanced', y=self.y_train)
        grid.fit(self.X_train, self.y_train, sample_weight=sample_weights)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    def _evaluate_model(self):
        # Evaluate model performance on test set
        y_test_pred = self.model.predict(self.X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(self.y_test, y_test_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_test_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_test_pred, zero_division=0))

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_test_pred)
        f1_macro = f1_score(self.y_test, y_test_pred, average='macro', zero_division=0)
        recall_macro = recall_score(self.y_test, y_test_pred, average='macro', zero_division=0)
        precision_macro = precision_score(self.y_test, y_test_pred, average='macro', zero_division=0)

        print("\n🔍 Multi-class Evaluation Metrics (macro average):")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Sensitivity    : {recall_macro:.4f}")
        print(f"F1-score       : {f1_macro:.4f}")
        print(f"Precision      : {precision_macro:.4f}")

        self.performance = {
            "accuracy": round(accuracy, 4),
            "macro_f1": round(f1_macro, 4),
            "macro_recall": round(recall_macro, 4),
            "macro_precision": round(precision_macro, 4),
            "confusion_matrix": confusion_matrix(self.y_test, y_test_pred).tolist(),
            "classification_report": classification_report(self.y_test, y_test_pred, zero_division=0, output_dict=True)
        }

    def _save_results(self):
        # Save model, parameters and performance metrics
        # Save performance metrics
        performance_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_performance.json')
        with open(performance_path, 'w') as f:
            json.dump(self.performance, f, indent=4)
        print(f"Model performance saved to {performance_path}")

        # Save model
        model_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # Save best parameters
        params_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_best_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        print(f"Best parameters saved to {params_path}")

    def _generate_feature_importance_plots(self, max_display: int = 12):
        """保存 XGBoost 特征重要性：优先用 booster 的 gain；回退到 sklearn 的 feature_importances_。导出柱状图与 CSV。"""
        if self.model is None:
            print("[FI] model is None, skip.")
            return

        # 1) 尝试用 booster 的 gain 重要性（更可信）
        fi_series = None
        try:
            booster = getattr(self.model, "get_booster", lambda: None)()
            if booster is not None:
                score = booster.get_score(importance_type="gain")  # dict: {'f0': 0.12, ...}
                if score:
                    # 将 f0/f1... 映射回列名
                    fmap = {f"f{i}": col for i, col in enumerate(self.X_train.columns)}
                    fi_series = pd.Series({fmap.get(k, k): v for k, v in score.items()}, dtype=float)
        except Exception as e:
            print(f"[FI] booster gain importance failed: {e}")

        # 2) 回退到 sklearn 接口
        if fi_series is None or fi_series.empty:
            try:
                importances = getattr(self.model, "feature_importances_", None)
                if importances is not None and len(importances) == self.X_train.shape[1]:
                    fi_series = pd.Series(importances, index=self.X_train.columns, dtype=float)
            except Exception as e:
                print(f"[FI] feature_importances_ failed: {e}")

        if fi_series is None or fi_series.empty:
            print("[FI] No feature importance available, skip.")
            return

        fi_series = fi_series.fillna(0.0).sort_values(ascending=False)
        top = fi_series.head(max_display)

        # 3) 保存 CSV
        csv_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_feature_importance_top{max_display}.csv")
        top.reset_index().rename(columns={"index": "feature", 0: "importance"}).to_csv(csv_path, index=False)
        print(f"[FI] Saved CSV: {csv_path}")

        # 4) 画柱状图（matplotlib）
        plt.figure(figsize=(9.5, max(4, 0.45 * len(top) + 1)))
        plt.barh(top.index[::-1], top.values[::-1])
        plt.xlabel("Importance (gain or impurity)")
        plt.title(f"Top-{max_display} Feature Importance — {self.dataset_name}")
        plt.tight_layout()
        out_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_feature_importance_top{max_display}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FI] Saved bar plot: {out_path}")


    def _generate_shap_plots(self):
        """Multiclass SHAP: 合并图 + 每类 vs 其他类（正类/其余样本分别绘制），Top-12。"""
        print("[SHAP] start: robust multiclass plotting (combined + per-class one-vs-rest)")

        # --- 1) 清洗为数值，规避 object/None/Inf 问题 ---
        def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out = out.replace({None: np.nan}).replace([np.inf, -np.inf], np.nan)
            for c in out.columns:
                s = out[c]
                if s.dtype == bool:
                    out[c] = s.astype(np.int8); continue
                if pd.api.types.is_categorical_dtype(s):
                    out[c] = s.cat.add_categories(["__NA__"]).fillna("__NA__").cat.codes.astype(np.int32); continue
                if s.dtype == object or pd.api.types.is_object_dtype(s):
                    num = pd.to_numeric(s, errors="coerce")
                    if num.notna().mean() >= 0.6:
                        out[c] = num
                    else:
                        codes, _ = pd.factorize(s.astype(str), sort=False)
                        out[c] = codes.astype(np.int32)
            return out.astype(np.float32)

        X_num = _sanitize(self.X_test)
        if not isinstance(X_num, pd.DataFrame):
            X_num = pd.DataFrame(X_num, columns=[f"Feature_{i}" for i in range(X_num.shape[1])])

        # --- 2) TreeExplainer，使用 raw 输出（更稳） ---
        explainer = shap.TreeExplainer(self.model, model_output="raw")
        sv = explainer.shap_values(X_num)  # 可能为 list[n_classes] 或 ndarray(三维/二维)
        print(f"[SHAP] raw shap_values type={type(sv)}")

        # --- 3) 统一成 list[n_classes]，每个元素形状 (n_samples, n_features) ---
        sv_list = None
        sv_single = None
        if isinstance(sv, list):
            sv_list = sv
            print(f"[SHAP] list of {len(sv_list)} arrays; each shape={sv_list[0].shape}")
        else:
            arr = np.asarray(sv)
            print(f"[SHAP] ndarray shape={arr.shape}")
            if arr.ndim == 3:
                n, a, b = arr.shape
                if a == getattr(self.model, "n_classes_", a) and b == X_num.shape[1]:
                    arr = np.transpose(arr, (1, 0, 2))  # (classes, samples, features)
                elif n == getattr(self.model, "n_classes_", n) and a == X_num.shape[0]:
                    pass
                elif b == getattr(self.model, "n_classes_", b) and a == X_num.shape[1]:
                    arr = np.transpose(arr, (2, 0, 1))
                else:
                    arr = np.transpose(arr, (1, 0, 2))
                sv_list = [arr[i] for i in range(arr.shape[0])]
                print(f"[SHAP] normalized to list of {len(sv_list)} arrays; each shape={sv_list[0].shape}")
            elif arr.ndim == 2:
                sv_single = arr
                print("[SHAP] single-output array detected")
            else:
                raise RuntimeError(f"Unexpected SHAP values shape: {arr.shape}")

        max_display = 12
        h = max(6, 0.58 * max_display + 1.5)

        # --- 4) 合并版（单张）：按 predict_proba 加权（若不可得则各类平均） ---
        if sv_list is not None:
            sv_stack = np.stack(sv_list, axis=0)  # (classes, samples, features)
            try:
                proba = self.model.predict_proba(self.X_test)
            except Exception:
                proba = None

            if proba is not None and proba.shape == (X_num.shape[0], sv_stack.shape[0]):
                sv_combined = np.einsum("sc,csf->sf", proba, sv_stack)
                print("[SHAP] combined by predict_proba weights")
            else:
                sv_combined = sv_stack.mean(axis=0)
                print("[SHAP] combined by class-wise mean (no proba)")

            plt.figure()
            shap.summary_plot(sv_combined, X_num, plot_type="dot", max_display=max_display, show=False)
            plt.gcf().set_size_inches(9.5, h); plt.tight_layout()
            combined_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_shap_beeswarm_combined_top{max_display}.png")
            plt.savefig(combined_path, dpi=300, bbox_inches="tight"); plt.close()
            print(f"[SHAP] Saved combined: {combined_path}")

            # 同步导出 Top-12 CSV
            mean_abs = np.mean(np.abs(sv_combined), axis=0)
            order = np.argsort(mean_abs)[::-1][:max_display]
            df_top = pd.DataFrame({"feature": X_num.columns[order], "mean_abs_shap": mean_abs[order]})
            csv_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_shap_top{max_display}_combined.csv")
            df_top.to_csv(csv_path, index=False); print(f"[SHAP] Saved combined CSV: {csv_path}")

            # --- 5) 关键新增：每类一对多（class_k vs Rest），正类/其余样本分别绘制 ---
            class_ids = getattr(self.model, "classes_", list(range(len(sv_list))))
            y_true = np.asarray(self.y_test)

            for ci, sv_c in enumerate(sv_list):
                # sv_c: (n_samples, n_features) —— 对应“该类”的 SHAP 贡献
                mask_pos = (y_true == class_ids[ci])
                mask_neg = ~mask_pos

                def _safe_plot(mask: np.ndarray, tag: str):
                    if mask.sum() < 5:
                        print(f"[SHAP] Skip class {class_ids[ci]} {tag}: only {mask.sum()} samples.")
                        return

                    # --- 1) beeswarm 图 ---
                    plt.figure()
                    shap.summary_plot(sv_c[mask], X_num.iloc[mask], plot_type="dot",
                                      max_display=max_display, show=False)
                    plt.gcf().set_size_inches(9.5, h)
                    plt.title(f"Class {class_ids[ci]} vs Rest — {tag}")
                    plt.tight_layout()
                    out = os.path.join(
                        self.target_subfolder,
                        f"{self.dataset_name}_shap_beeswarm_class{class_ids[ci]}_{tag.lower().replace(' ','_')}_top{max_display}.png"
                    )
                    plt.savefig(out, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"[SHAP] Saved beeswarm {tag}: {out}")

                    # --- 2) bar 图（Top-12 平均绝对 SHAP 值） ---
                    mean_abs = np.mean(np.abs(sv_c[mask]), axis=0)
                    order = np.argsort(mean_abs)[::-1][:max_display]
                    top_features = X_num.columns[order]
                    top_values = mean_abs[order]

                    plt.figure(figsize=(9, 0.5 * len(top_features) + 1.5))
                    plt.barh(top_features[::-1], top_values[::-1])
                    plt.xlabel("Mean |SHAP value|")
                    plt.title(f"Top-{max_display} Features — Class {class_ids[ci]} {tag}")
                    plt.tight_layout()
                    out_bar = os.path.join(
                        self.target_subfolder,
                        f"{self.dataset_name}_shap_bar_class{class_ids[ci]}_{tag.lower().replace(' ','_')}_top{max_display}.png"
                    )
                    plt.savefig(out_bar, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"[SHAP] Saved bar {tag}: {out_bar}")

                    # --- 3) 保存 CSV 方便统计 ---
                    df_top = pd.DataFrame({"feature": top_features, "mean_abs_shap": top_values})
                    csv_path = os.path.join(
                        self.target_subfolder,
                        f"{self.dataset_name}_shap_bar_class{class_ids[ci]}_{tag.lower().replace(' ','_')}_top{max_display}.csv"
                    )
                    df_top.to_csv(csv_path, index=False)
                    print(f"[SHAP] Saved CSV {tag}: {csv_path}")


                # 只看该类样本（正类）
                _safe_plot(mask_pos, "Positives (y==class)")

                # 只看其余样本（非该类）
                _safe_plot(mask_neg, "Rest (y!=class)")

        elif sv_single is not None:
            # 二分类 / 单输出
            plt.figure()
            shap.summary_plot(sv_single, X_num, plot_type="dot", max_display=max_display, show=False)
            plt.gcf().set_size_inches(9.5, h); plt.tight_layout()
            out_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_shap_beeswarm_top{max_display}.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
            print(f"[SHAP] Saved single-output: {out_path}")
