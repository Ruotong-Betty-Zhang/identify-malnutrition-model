# -*- coding: utf-8 -*-
import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Optional, Dict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, recall_score, precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

import shap

# -------------------- 全局设置 --------------------
warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("agg")  # 无界面环境下保存图片更稳

# 可读名称的数据字典默认路径（自动尝试，找不到则跳过）
FEATURE_DICT_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # 回到项目根目录
    "input",
    "InterRAI data dictionary.xlsx"
)



class XGBoostModelTrainer:
    def __init__(self, seed: int = 42, device: str = 'cuda'):
        self.seed = seed
        self.device = device

        self.model: Optional[XGBClassifier] = None
        self.best_params: Optional[Dict] = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.target_subfolder: Optional[str] = None
        self.dataset_name: Optional[str] = None
        self.performance: Optional[Dict] = None

        # 特征名映射（代码 -> 友好名）；_feature_display_name 为列名转友好名的函数
        self.feature_name_map: Dict[str, str] = {}
        self._feature_display_name = None

    # ==================== 训练主流程 ====================
    def train(self, dataset_path: str, output_folder: str, parameters=None):
        self._setup_output_folder(dataset_path, output_folder)
        df = self._load_and_preprocess_data(dataset_path)

        # 自动尝试加载数据字典：input/InterRAI data dictionary
        self._try_load_feature_dictionary(FEATURE_DICT_PATH_DEFAULT)

        self._split_data(df, dataset_path)
        self._perform_grid_search(parameters)
        self._evaluate_model()
        self._save_results()
        self._generate_feature_importance_plots()
        self._generate_shap_plots()
        return self.model

    # ==================== 目录/数据处理 ====================
    def _setup_output_folder(self, dataset_path: str, output_folder: str):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        self.dataset_name = os.path.basename(dataset_path).split('.')[0]
        self.target_subfolder = os.path.join(output_folder, 'xgb_' + self.dataset_name)
        if not os.path.exists(self.target_subfolder):
            os.makedirs(self.target_subfolder)

    def _load_and_preprocess_data(self, dataset_path: str) -> pd.DataFrame:
        np.random.seed(self.seed)
        df = pd.read_pickle(dataset_path)
        print(f"Dataset shape: {df.shape}")
        drop_cols = [c for c in ['IDno', 'Assessment_Date'] if c in df.columns]
        return df.drop(columns=drop_cols)

    def _split_data(self, df: pd.DataFrame, dataset_path: str):
        if 'MAL' in dataset_path:
            target = "Malnutrition"
        elif 'CAP' in dataset_path:
            target = "CAP_Nutrition"
        else:
            raise ValueError("Unknown dataset format. Please provide a dataset file starting with CAP or MAL.")

        X = df.drop(columns=[target])
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")

    # ==================== 网格搜索 ====================
    def _perform_grid_search(self, parameters):
        params = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
        } if parameters is None else parameters

        # 为了兼容性与可复现，使用 CPU + hist（GPU 环境不一致时容易出问题）
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            tree_method='hist',
            device='cpu',
            random_state=self.seed
        )

        scoring_method = 'recall_macro' if len(np.unique(self.y_train)) > 2 else 'recall'
        min_class_count = int(self.y_train.value_counts().min())
        cv_folds = max(2, min(5, min_class_count))  # 至少2折

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

    # ==================== 评估/保存 ====================
    def _evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred, zero_division=0))

        self.performance = {
            "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "macro_f1": round(f1_score(self.y_test, y_pred, average='macro', zero_division=0), 4),
            "macro_recall": round(recall_score(self.y_test, y_pred, average='macro', zero_division=0), 4),
            "macro_precision": round(precision_score(self.y_test, y_pred, average='macro', zero_division=0), 4),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
            "classification_report": classification_report(self.y_test, y_pred, zero_division=0, output_dict=True)
        }

        print("\n🔍 Multi-class Evaluation Metrics (macro average):")
        print(f"Accuracy       : {self.performance['accuracy']:.4f}")
        print(f"Sensitivity    : {self.performance['macro_recall']:.4f}")
        print(f"F1-score       : {self.performance['macro_f1']:.4f}")
        print(f"Precision      : {self.performance['macro_precision']:.4f}")

    def _save_results(self):
        perf_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_performance.json')
        with open(perf_path, 'w') as f:
            json.dump(self.performance, f, indent=4)
        print(f"Model performance saved to {perf_path}")

        model_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        params_path = os.path.join(self.target_subfolder, f'{self.dataset_name}_xgb_best_params.json')
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        print(f"Best parameters saved to {params_path}")

    # ==================== 特征重要性 ====================
    def _generate_feature_importance_plots(self, max_display: int = 12):
        """保存 XGBoost 特征重要性（gain优先）。图表按“占比(%)”显示，并在柱子上标百分比；CSV 同步输出占比。"""
        from matplotlib.ticker import PercentFormatter

        if self.model is None:
            print("[FI] model is None, skip.")
            return

        fi_series = None
        # 1) booster gain
        try:
            booster = getattr(self.model, "get_booster", lambda: None)()
            if booster is not None:
                score = booster.get_score(importance_type="gain")  # {'f0': val, ...}
                if score:
                    fmap = {f"f{i}": col for i, col in enumerate(self.X_train.columns)}
                    fi_series = pd.Series({fmap.get(k, k): v for k, v in score.items()}, dtype=float)
        except Exception as e:
            print(f"[FI] booster gain importance failed: {e}")

        # 2) 回退 sklearn 的 feature_importances_
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

        # 归一化为“占比”（总和=1），再取 Top-k
        fi_series = fi_series.fillna(0.0)
        total = fi_series.sum()
        if total <= 0:
            print("[FI] Sum of importance is 0, skip.")
            return
        fi_prop = (fi_series / total).sort_values(ascending=False)
        top = fi_prop.head(max_display)

        # CSV：代码 + 名称 + 占比
        top_df = top.reset_index()
        top_df.columns = ["feature_code", "importance_prop"]
        top_df["feature_name"] = self._map_feature_names(top_df["feature_code"])
        top_df["importance_percent"] = (top_df["importance_prop"] * 100).round(2)

        csv_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_feature_importance_top{max_display}.csv")
        top_df[["feature_code", "feature_name", "importance_prop", "importance_percent"]].to_csv(csv_path, index=False)
        print(f"[FI] Saved CSV: {csv_path}")

        # 画布更大 + 标签显示“名称（代码）”
        labels = self._compose_display_labels(top_df["feature_name"], top_df["feature_code"])

        # —— 关键：放大画布（宽 13~14，更好容纳长名称；高随条数自适应）——
        fig_h = max(5, 0.6 * len(labels) + 1.5)
        plt.figure(figsize=(13.5, fig_h))

        # 柱状图用“占比”，x 轴显示百分比
        values = top_df["importance_prop"].values
        bars = plt.barh(labels[::-1], values[::-1])

        # x 轴百分比格式
        ax = plt.gca()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.xlabel("Relative Importance (%)")
        plt.title(f"Top-{max_display} Feature Importance — {self.dataset_name}")

        # 在每根柱子右边对齐标注百分比
        xmax = ax.get_xlim()[1]   # 取坐标轴的右边界
        for bar, v in zip(bars, values[::-1]):
            y = bar.get_y() + bar.get_height()/2
            plt.text(
                xmax, y, f"{v*100:.1f}%", 
                va='center', ha='right',  # 竖直居中，水平靠右
                fontsize=10
            )


        plt.tight_layout()
        out_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_feature_importance_top{max_display}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FI] Saved bar plot: {out_path}")


    # ==================== SHAP 图与 CSV ====================
    def _generate_shap_plots(self):
        print("[SHAP] start: per-class Positives beeswarm only")

        # —— 数值化清洗，避免 object/None/Inf 报错 ——
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

        # beeswarm 用「可读列名」的副本
        X_num_disp = self._with_readable_columns(X_num)

        # 解释器
        try:
            explainer = shap.TreeExplainer(self.model, model_output="raw")
            sv = explainer.shap_values(X_num)
        except Exception as e:
            print(f"[SHAP] Explainer failed: {e}")
            return

        # 统一成 list[n_classes]，每个 (n_samples, n_features)
        if isinstance(sv, list):
            sv_list = sv
        else:
            arr = np.asarray(sv)
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
            elif arr.ndim == 2:
                # 二分类时只出一张
                plt.figure()
                shap.summary_plot(arr, X_num_disp, plot_type="dot", max_display=12, show=False)
                plt.gcf().set_size_inches(9.5, 8); plt.tight_layout()
                out_path = os.path.join(self.target_subfolder, f"{self.dataset_name}_shap_beeswarm_binary_top12.png")
                plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
                print(f"[SHAP] Saved: {out_path}")
                return
            else:
                print(f"[SHAP] Unexpected shape: {arr.shape}")
                return

        class_ids = getattr(self.model, "classes_", list(range(len(sv_list))))
        y_true = np.asarray(self.y_test)
        max_display = 12
        h = max(6, 0.58 * max_display + 1.5)

        # 仅保存「每类的 Positives」beeswarm
        for ci, sv_c in enumerate(sv_list):
            mask_pos = (y_true == class_ids[ci])
            if mask_pos.sum() < 5:
                print(f"[SHAP] Skip class {class_ids[ci]} Positives: only {mask_pos.sum()} samples.")
                continue

            plt.figure()
            shap.summary_plot(sv_c[mask_pos], X_num_disp.iloc[mask_pos],
                            plot_type="dot", max_display=max_display, show=False)
            plt.gcf().set_size_inches(13.5, max(h, 8))  # 更宽；高度至少 8
            plt.title(f"SHAP Beeswarm — Class {class_ids[ci]} (Positives)")
            plt.tight_layout()
            out_bee = os.path.join(
                self.target_subfolder,
                f"{self.dataset_name}_shap_beeswarm_class{class_ids[ci]}_positives_top{max_display}.png"
            )
            plt.savefig(out_bee, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SHAP] Saved: {out_bee}")


    # ==================== 字典加载/名称映射 ====================
    def _try_load_feature_dictionary(self, path: str):
        """尝试从目录/文件加载数据字典；失败则静默跳过。"""
        try:
            if not os.path.exists(path):
                print(f"[DICT] Skip: path not found -> {path}")
                return
            self.feature_name_map = self._load_feature_name_map(path)
            if self.feature_name_map:
                print(f"[DICT] Loaded feature name map: {len(self.feature_name_map)} entries")
        except Exception as e:
            print(f"[DICT] Load dictionary failed: {e}")

    def _load_feature_name_map(self, dict_path: str, sheet: str = "Questions"):
        """
        支持传目录或单个文件：
        - 如果是目录：找名字里带 question 的文件
        - 如果是文件：直接用这个文件
        """
        import pandas as pd, os

        # 如果传进来的是目录，就找文件
        if os.path.isdir(dict_path):
            found = None
            for fn in os.listdir(dict_path):
                if fn.lower().endswith((".xlsx", ".xls", ".csv")) and "question" in fn.lower():
                    found = os.path.join(dict_path, fn)
                    break
            if found is None:
                print(f"[DICT] No Question file found in {dict_path}")
                return {}
            dict_path = found
        elif not os.path.isfile(dict_path):
            print(f"[DICT] Path not found: {dict_path}")
            return {}

        # ---- 读文件 ----
        if dict_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(dict_path, sheet_name=sheet)
        else:
            df = pd.read_csv(dict_path)

        # ---- 检查列 ----
        if not {"iCode", "Question"}.issubset(df.columns):
            print(f"[DICT] File {dict_path} missing iCode/Question columns, got {list(df.columns)}")
            return {}

        mapping = dict(zip(df["iCode"].astype(str).str.strip(),
                        df["Question"].astype(str).str.strip()))

        def _display_name(col: str) -> str:
            col_str = str(col)
            base = re.sub(r"(_scaled|_std|_lag\d+|_zscore)$", "", col_str)
            return mapping.get(base, col_str)

        self._feature_display_name = _display_name
        return mapping


    def _norm_colname(self, c):
        s = str(c).strip()
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"[^0-9a-zA-Z_]", "", s)
        return s.lower()

    def _map_feature_names(self, cols):
        if self._feature_display_name is None:
            return list(cols)
        return [self._feature_display_name(c) for c in cols]

    def _compose_display_labels(self, names, codes):
        """名称与代码合并为唯一标签：Name (Code)。若两者一致则不重复显示。"""
        labels = []
        for n, c in zip(names, codes):
            if str(n) == str(c):
                labels.append(str(n))
            else:
                labels.append(f"{n} ({c})")
        # 保证唯一性（避免重复）
        uniq = []
        seen = {}
        for s in labels:
            if s not in seen:
                seen[s] = 1
                uniq.append(s)
            else:
                seen[s] += 1
                uniq.append(f"{s} #{seen[s]}")
        return uniq

    def _with_readable_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回用于绘图的列名副本（可读名称），并确保唯一，避免 beeswarm y 轴重复。"""
        names = self._map_feature_names(X.columns)
        labels = self._compose_display_labels(names, X.columns)
        X_disp = X.copy()
        X_disp.columns = labels
        return X_disp
