import pandas as pd
import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import json
from sklearn.metrics import f1_score, recall_score, precision_score

class XGBoostModelTrainer:
    def __init__(self, seed=42, device='cuda'):
        self.seed = seed
        self.device = device
        self.model = None
        self.best_params = None

    def train(self, dataset_path: str, output_folder: str, parameters=None, test_on_original=False):
        # Make sure output folder exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Set random seed
        np.random.seed(self.seed)

        # Load data
        df = pd.read_pickle(dataset_path)
        print(f"Dataset shape: {df.shape}")

        # Identify target column
        if 'MAL' in dataset_path:
            target_col = "Malnutrition"
        elif 'CAP' in dataset_path:
            target_col = "CAP_Nutrition"
        else:
            print("Unknown dataset format. Please provide a valid dataset start with CAP or MAL.")
            return None

        # According to test_on_original flag, split data
        if test_on_original:
            if 'IDno' not in df.columns:
                raise ValueError("IDno column is missing, cannot test only on original data. Please ensure IDno is retained before augmentation.")
            orig_mask = df['IDno'].notna()
            df_orig = df.loc[orig_mask]
            if df_orig.empty:
                raise ValueError("Original data subset is empty (IDno is all NaN).")

            stratify_y = df_orig[target_col] if df_orig[target_col].nunique() > 1 else None
            train_idx_orig, test_idx = train_test_split(
                df_orig.index,
                test_size=0.2,
                random_state=self.seed,
                stratify=stratify_y
            )
            train_df = df.drop(index=test_idx)   # Training set = full data minus original test indices (includes original train + all augmentations)
            test_df = df.loc[test_idx]           # Test set = only original samples
            print(f"[Split] test_on_original=True -> Test only from original samples (IDno notna).")
        else:
            stratify_y = df[target_col] if df[target_col].nunique() > 1 else None
            train_idx, test_idx = train_test_split(
                df.index,
                test_size=0.2,
                random_state=self.seed,
                stratify=stratify_y
            )
            train_df = df.loc[train_idx]
            test_df  = df.loc[test_idx]
            print(f"[Split] test_on_original=False -> Standard random split on full dataset.")

        # Drop irrelevant columns and build X/y
        drop_cols = [c for c in ['IDno', 'Assessment_Date'] if c in df.columns]
        X_train = train_df.drop(columns=drop_cols + [target_col])
        y_train = train_df[target_col]
        X_test  = test_df.drop(columns=drop_cols + [target_col])
        y_test  = test_df[target_col]

        print(f"Training set size: {X_train.shape[0]}"
            f"({'Include augmentations' if test_on_original else 'Could include augmentations'})")
        print(f"Test set size: {X_test.shape[0]}"
            f"({'Only original samples' if test_on_original else 'Original + augmented mix'})")


        # Parameter search space (can be replaced with a larger space)
        params = {
            'max_depth': [5, 7, 10],
            'learning_rate': [0.01, 0.1],
            'n_estimators': [50, 100, 200],
            'subsample': [0.6, 0.8, 1],
            'colsample_bytree': [0.6, 0.8, 1],
        }

        if parameters:
            params = parameters

        # Define model
        xgb = XGBClassifier(
            eval_metric='mlogloss',
            tree_method='hist',
            device=self.device,
            random_state=self.seed
        )

        # Grid search
        grid = GridSearchCV(xgb, params, cv=5, verbose=1, n_jobs=1)
        grid.fit(X_train, y_train)

        print("Best parameters found:", grid.best_params_)
        self.model = grid.best_estimator_
        self.best_params = grid.best_params_

        # Test set prediction and evaluation
        y_test_pred = self.model.predict(X_test)
        print("Test Results:")
        print("Accuracy:", accuracy_score(y_test, y_test_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
        print("Classification Report:\n", classification_report(y_test, y_test_pred))

        # Compute multi-class metrics (macro averages are computed per class and then averaged, unaffected by class imbalance)
        accuracy = accuracy_score(y_test, y_test_pred)
        f1_macro = f1_score(y_test, y_test_pred, average='macro')
        recall_macro = recall_score(y_test, y_test_pred, average='macro')  # Sensitivity
        precision_macro = precision_score(y_test, y_test_pred, average='macro')

        print("\n🔍 Multi-class Evaluation Metrics (macro average):")
        print(f"Accuracy       : {accuracy:.4f}")
        print(f"Sensitivity    : {recall_macro:.4f}")      # i.e. macro recall
        print(f"F1-score       : {f1_macro:.4f}")
        print(f"Precision      : {precision_macro:.4f}")


        # Subfolder naming
        dataset_name = os.path.basename(dataset_path).split('.')[0]
        target_subfolder = os.path.join(output_folder, 'xgb_' + dataset_name)
        if not os.path.exists(target_subfolder):
            os.makedirs(target_subfolder)

        # Save model
        model_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_model.pkl')
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

        # Save best parameters
        self.save_best_params(target_subfolder, dataset_name)

        # Save feature importance plot
        importances = self.model.feature_importances_

        # Normalize importances to percentage (sum to 100)
        importances_pct = importances / importances.sum() * 100

        # Get top 12 feature indices and values
        indices = np.argsort(importances_pct)[::-1]
        top_n = 10
        top_indices = indices[:top_n]
        top_features = X_train.columns[top_indices]
        top_importances = importances_pct[top_indices]
        # Print the top 12 features in one line
        top_features_str = ', '.join([f"{feat}" for feat, imp in zip(top_features, top_importances)])
        print(f"Top 12 features: {top_features_str}")

        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(top_n), top_importances[::-1], align='center')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel("Feature Importance (%)")
        plt.title("Top " + str(top_n) + " Most Important Features (xgb_" + dataset_name + ")")
        plt.tight_layout()

        # Add white value labels next to each bar
        for i, (value, bar) in enumerate(zip(top_importances[::-1], bars)):
            plt.text(0.1, bar.get_y() + bar.get_height() / 2,
                    f'{value:.2f}%', va='center', color='white')

        # Save plot
        plot_path = os.path.join(target_subfolder, f'{dataset_name}_xgb_importance.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()

        print(f"Top 12 feature importance plot saved to {plot_path}")

        # Print the top 12 features in one line
        top_features_str = ', '.join([f"{feat}" for feat, imp in zip(top_features, top_importances)])
        print(f"Top 12 features: {top_features_str}")

        return self.model

    def save_best_params(self, output_folder, dataset_name):
        if self.best_params is not None:
            params_path = os.path.join(output_folder, f'{dataset_name}_xgb_best_params.json')
            with open(params_path, 'w') as f:
                json.dump(self.best_params, f, indent=4)
            print(f"Best parameters saved to {params_path}")
        else:
            print("No best parameters to save.")