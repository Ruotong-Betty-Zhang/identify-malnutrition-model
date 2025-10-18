# -*- coding: utf-8 -*-
import os
import re
import json
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from typing import Optional, Dict
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix,
    f1_score, recall_score, precision_score
)
from sklearn.utils.class_weight import compute_sample_weight

import shap

# Warnings and backend setup
warnings.filterwarnings("ignore", category=UserWarning)
plt.switch_backend("agg")

# Default path for the feature dictionary (if present)
load_dotenv(dotenv_path=".env", override=True)
FEATURE_DICT_PATH_DEFAULT = os.getenv("FEATURE_DICT_PATH")


class XGBoostModelTrainer:
    def __init__(self, seed: int = 42, device: str = "cuda"):
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

        # Feature code -> readable name mapping
        self.feature_name_map: Dict[str, str] = {}
        self._feature_display_name = None

    # ==================== Public API ====================

    def train(self, dataset_path: str, output_folder: str, parameters=None):
        """Main training pipeline: load, split, grid search, evaluate, save, and plot."""
        self._setup_output_folder(dataset_path, output_folder)
        df = self._load_and_preprocess_data(dataset_path)

        # Try loading an optional feature dictionary
        self._try_load_feature_dictionary(FEATURE_DICT_PATH_DEFAULT)

        self._split_data(df, dataset_path)
        self._perform_grid_search(parameters)
        self._evaluate_model()
        self._save_results()
        self._generate_feature_importance_plots()
        self._generate_shap_plots()
        return self.model

    # ==================== IO / Data Prep ====================

    def _setup_output_folder(self, dataset_path: str, output_folder: str):
        os.makedirs(output_folder, exist_ok=True)
        self.dataset_name = os.path.basename(dataset_path).split(".")[0]
        self.target_subfolder = os.path.join(output_folder, "xgb_" + self.dataset_name)
        os.makedirs(self.target_subfolder, exist_ok=True)

    def _load_and_preprocess_data(self, dataset_path: str) -> pd.DataFrame:
        np.random.seed(self.seed)
        df = pd.read_pickle(dataset_path)
        print(f"Dataset shape: {df.shape}")
        drop_cols = [c for c in ["IDno", "Assessment_Date"] if c in df.columns]
        return df.drop(columns=drop_cols)

    def _split_data(self, df: pd.DataFrame, dataset_path: str):
        if "MAL" in dataset_path:
            target = "Malnutrition"
        elif "CAP" in dataset_path:
            target = "CAP_Nutrition"
        else:
            raise ValueError("Unknown dataset format. Use file starting with CAP or MAL.")

        X = df.drop(columns=[target])
        y = df[target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )
        print(f"Training set size: {self.X_train.shape[0]}")
        print(f"Test set size: {self.X_test.shape[0]}")

    # ==================== Grid Search ====================

    def _perform_grid_search(self, parameters):
        params = (
            {
                "max_depth": [5, 7, 10],
                "learning_rate": [0.01, 0.1],
                "n_estimators": [50, 100, 200],
                "subsample": [0.6, 0.8, 1],
                "colsample_bytree": [0.6, 0.8, 1],
            }
            if parameters is None
            else parameters
        )

        # Use CPU hist for consistency/reproducibility across environments
        xgb = XGBClassifier(
            eval_metric="mlogloss",
            tree_method="hist",
            device="cpu",
            random_state=self.seed,
        )

        scoring_method = (
            "recall_macro" if len(np.unique(self.y_train)) > 2 else "recall"
        )
        min_class_count = int(self.y_train.value_counts().min())
        cv_folds = max(2, min(5, min_class_count))

        grid = GridSearchCV(
            xgb,
            param_grid=params,
            scoring=scoring_method,
            cv=cv_folds,
            verbose=1,
            n_jobs=1,
        )

        sample_weights = compute_sample_weight(
            class_weight="balanced", y=self.y_train
        )
        grid.fit(self.X_train, self.y_train, sample_weight=sample_weights)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

    # ==================== Evaluation / Save ====================

    def _evaluate_model(self):
        y_pred = self.model.predict(self.X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(self.y_test, y_pred))
        print(
            "Classification Report:\n",
            classification_report(self.y_test, y_pred, zero_division=0),
        )

        self.performance = {
            "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
            "macro_f1": round(
                f1_score(self.y_test, y_pred, average="macro", zero_division=0), 4
            ),
            "macro_recall": round(
                recall_score(self.y_test, y_pred, average="macro", zero_division=0), 4
            ),
            "macro_precision": round(
                precision_score(
                    self.y_test, y_pred, average="macro", zero_division=0
                ),
                4,
            ),
            "confusion_matrix": confusion_matrix(self.y_test, y_pred).tolist(),
            "classification_report": classification_report(
                self.y_test, y_pred, zero_division=0, output_dict=True
            ),
        }

        print("\nMulti-class evaluation (macro-average):")
        print(f"Accuracy    : {self.performance['accuracy']:.4f}")
        print(f"Recall      : {self.performance['macro_recall']:.4f}")
        print(f"F1-score    : {self.performance['macro_f1']:.4f}")
        print(f"Precision   : {self.performance['macro_precision']:.4f}")

    def _save_results(self):
        perf_path = os.path.join(
            self.target_subfolder, f"{self.dataset_name}_xgb_performance.json"
        )
        with open(perf_path, "w") as f:
            json.dump(self.performance, f, indent=4)
        print(f"Saved performance: {perf_path}")

        model_path = os.path.join(
            self.target_subfolder, f"{self.dataset_name}_xgb_model.pkl"
        )
        joblib.dump(self.model, model_path)
        print(f"Saved model: {model_path}")

        params_path = os.path.join(
            self.target_subfolder, f"{self.dataset_name}_xgb_best_params.json"
        )
        with open(params_path, "w") as f:
            json.dump(self.best_params, f, indent=4)
        print(f"Saved best parameters: {params_path}")

    # ==================== Feature Importance ====================

    def _generate_feature_importance_plots(self, max_display: int = 12):
        """Save top-k feature importance (gain preferred), with CSV and bar plot."""
        from matplotlib.ticker import PercentFormatter

        if self.model is None:
            print("[FI] model is None, skip.")
            return

        fi_series = None
        # 1) Booster gain
        try:
            booster = getattr(self.model, "get_booster", lambda: None)()
            if booster is not None:
                score = booster.get_score(importance_type="gain")
                if score:
                    fmap = {f"f{i}": col for i, col in enumerate(self.X_train.columns)}
                    fi_series = pd.Series(
                        {fmap.get(k, k): v for k, v in score.items()}, dtype=float
                    )
        except Exception as e:
            print(f"[FI] booster gain importance failed: {e}")

        # 2) Fallback: sklearn feature_importances_
        if fi_series is None or fi_series.empty:
            try:
                importances = getattr(self.model, "feature_importances_", None)
                if importances is not None and len(importances) == self.X_train.shape[1]:
                    fi_series = pd.Series(
                        importances, index=self.X_train.columns, dtype=float
                    )
            except Exception as e:
                print(f"[FI] feature_importances_ failed: {e}")

        if fi_series is None or fi_series.empty:
            print("[FI] No feature importance available, skip.")
            return

        # Normalize to proportions and select Top-k
        fi_series = fi_series.fillna(0.0)
        total = fi_series.sum()
        if total <= 0:
            print("[FI] Sum of importance is 0, skip.")
            return
        fi_prop = (fi_series / total).sort_values(ascending=False)
        top = fi_prop.head(max_display)

        # CSV with code + readable name + percentage
        top_df = top.reset_index()
        top_df.columns = ["feature_code", "importance_prop"]
        top_df["feature_name"] = self._map_feature_names(top_df["feature_code"])
        top_df["importance_percent"] = (top_df["importance_prop"] * 100).round(2)

        csv_path = os.path.join(
            self.target_subfolder,
            f"{self.dataset_name}_feature_importance_top{max_display}.csv",
        )
        top_df[
            [
                "feature_code",
                "feature_name",
                "importance_prop",
                "importance_percent",
            ]
        ].to_csv(csv_path, index=False)
        print(f"[FI] Saved CSV: {csv_path}")

        labels = self._compose_display_labels(
            top_df["feature_name"], top_df["feature_code"]
        )

        # Wide figure to accommodate long labels
        fig_h = max(5, 0.6 * len(labels) + 1.5)
        plt.figure(figsize=(13.5, fig_h))

        values = top_df["importance_prop"].values
        bars = plt.barh(labels[::-1], values[::-1])

        ax = plt.gca()
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        plt.xlabel("Relative Importance (%)")
        plt.title(f"Top-{max_display} Feature Importance — {self.dataset_name}")

        # Percentage labels on the right edge
        xmax = ax.get_xlim()[1]
        for bar, v in zip(bars, values[::-1]):
            y = bar.get_y() + bar.get_height() / 2
            plt.text(xmax, y, f"{v*100:.1f}%", va="center", ha="right", fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(
            self.target_subfolder,
            f"{self.dataset_name}_feature_importance_top{max_display}.png",
        )
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[FI] Saved bar plot: {out_path}")

    # ==================== SHAP ====================

    def _generate_shap_plots(self):
        print("[SHAP] start: per-class positives beeswarm only")

        # Sanitize input to numeric/codes to avoid dtype issues
        def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
            out = df.copy()
            out = out.replace({None: np.nan}).replace([np.inf, -np.inf], np.nan)
            for c in out.columns:
                s = out[c]
                if s.dtype == bool:
                    out[c] = s.astype(np.int8)
                    continue
                if pd.api.types.is_categorical_dtype(s):
                    out[c] = (
                        s.cat.add_categories(["__NA__"])
                        .fillna("__NA__")
                        .cat.codes.astype(np.int32)
                    )
                    continue
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
            X_num = pd.DataFrame(
                X_num, columns=[f"Feature_{i}" for i in range(X_num.shape[1])]
            )

        X_num_disp = self._with_readable_columns(X_num)

        try:
            explainer = shap.TreeExplainer(self.model, model_output="raw")
            sv = explainer.shap_values(X_num)
        except Exception as e:
            print(f"[SHAP] Explainer failed: {e}")
            return

        # Normalize the shape to list[n_classes] -> (n_samples, n_features)
        if isinstance(sv, list):
            sv_list = sv
        else:
            arr = np.asarray(sv)
            if arr.ndim == 3:
                n, a, b = arr.shape
                if a == getattr(self.model, "n_classes_", a) and b == X_num.shape[1]:
                    arr = np.transpose(arr, (1, 0, 2))
                elif n == getattr(self.model, "n_classes_", n) and a == X_num.shape[0]:
                    pass
                elif b == getattr(self.model, "n_classes_", b) and a == X_num.shape[1]:
                    arr = np.transpose(arr, (2, 0, 1))
                else:
                    arr = np.transpose(arr, (1, 0, 2))
                sv_list = [arr[i] for i in range(arr.shape[0])]
            elif arr.ndim == 2:
                # Binary case: single beeswarm
                plt.figure()
                shap.summary_plot(
                    arr, X_num_disp, plot_type="dot", max_display=12, show=False
                )
                plt.gcf().set_size_inches(9.5, 8)
                plt.tight_layout()
                out_path = os.path.join(
                    self.target_subfolder,
                    f"{self.dataset_name}_shap_beeswarm_binary_top12.png",
                )
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                plt.close()
                print(f"[SHAP] Saved: {out_path}")
                return
            else:
                print(f"[SHAP] Unexpected shape: {arr.shape}")
                return

        class_ids = getattr(self.model, "classes_", list(range(len(sv_list))))
        y_true = np.asarray(self.y_test)
        max_display = 12
        h = max(6, 0.58 * max_display + 1.5)

        # Beeswarm for positives of each class
        for ci, sv_c in enumerate(sv_list):
            mask_pos = y_true == class_ids[ci]
            if mask_pos.sum() < 5:
                print(
                    f"[SHAP] Skip class {class_ids[ci]} positives: only {mask_pos.sum()} samples."
                )
                continue

            plt.figure()
            shap.summary_plot(
                sv_c[mask_pos],
                X_num_disp.iloc[mask_pos],
                plot_type="dot",
                max_display=max_display,
                show=False,
            )
            plt.gcf().set_size_inches(13.5, max(h, 8))
            plt.title(f"SHAP Beeswarm — Class {class_ids[ci]} (Positives)")
            plt.tight_layout()
            out_bee = os.path.join(
                self.target_subfolder,
                f"{self.dataset_name}_shap_beeswarm_class{class_ids[ci]}_positives_top{max_display}.png",
            )
            plt.savefig(out_bee, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[SHAP] Saved: {out_bee}")

    # ==================== Dictionary / Name Mapping ====================

    def _try_load_feature_dictionary(self, path: str):
        """Try to load the feature dictionary; silently skip if not found or invalid."""
        try:
            if not os.path.exists(path):
                print(f"[DICT] Skip: path not found -> {path}")
                return
            self.feature_name_map = self._load_feature_name_map(path)
            if self.feature_name_map:
                print(
                    f"[DICT] Loaded feature name map: {len(self.feature_name_map)} entries"
                )
        except Exception as e:
            print(f"[DICT] Load dictionary failed: {e}")

    def _load_feature_name_map(self, dict_path: str, sheet: str = "Questions"):
        """
        Supports a directory or a single file.
        - Directory: pick a file whose name contains 'question'
        - File: use it directly
        """
        import os

        if os.path.isdir(dict_path):
            found = None
            for fn in os.listdir(dict_path):
                if fn.lower().endswith((".xlsx", ".xls", ".csv")) and "question" in fn.lower():
                    found = os.path.join(dict_path, fn)
                    break
            if found is None:
                print(f"[DICT] No 'question' file found in {dict_path}")
                return {}
            dict_path = found
        elif not os.path.isfile(dict_path):
            print(f"[DICT] Path not found: {dict_path}")
            return {}

        if dict_path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(dict_path, sheet_name=sheet)
        else:
            df = pd.read_csv(dict_path)

        if not {"iCode", "Question"}.issubset(df.columns):
            print(
                f"[DICT] Missing iCode/Question columns, got {list(df.columns)}"
            )
            return {}

        mapping = dict(
            zip(
                df["iCode"].astype(str).str.strip(),
                df["Question"].astype(str).str.strip(),
            )
        )

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
        """Compose unique labels 'Name (Code)'; avoid duplicates."""
        labels = []
        for n, c in zip(names, codes):
            if str(n) == str(c):
                labels.append(str(n))
            else:
                labels.append(f"{n} ({c})")
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
        """Return a copy of X with readable, unique column names for plotting."""
        names = self._map_feature_names(X.columns)
        labels = self._compose_display_labels(names, X.columns)
        X_disp = X.copy()
        X_disp.columns = labels
        return X_disp
