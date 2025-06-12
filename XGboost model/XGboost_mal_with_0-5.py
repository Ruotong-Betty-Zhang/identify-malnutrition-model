import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def split_mal_data(df):
    cols_to_drop = [df.columns[0], df.columns[1], 'Malnutrition']
    X = df.drop(columns=cols_to_drop)
    y = df['Malnutrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test


def train_xgboost_with_grid_search(X_train, y_train):
    xgb_model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42
    )
    
    param_grid = {
        'max_depth': [7, 10],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.6, 0.8],
    }
    
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='f1_weighted',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    
    return grid_search.best_estimator_

def get_and_print_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def plot_feature_importance(clf, X_train, top_n=20):
    feature_importances = clf.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    importance_df = importance_df.head(top_n)
    print(importance_df.head(top_n))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plt.savefig("./image/xgboost_mal_feature_importance_score_0_5.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    df = pd.read_pickle("./datasets/mal_data.pkl")
    X_train, X_test, y_train, y_test = split_mal_data(df)
    
    # Train the XGBoost model with grid search
    clf = train_xgboost_with_grid_search(X_train, y_train)
    
    # Get and print accuracy
    accuracy = get_and_print_accuracy(clf, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(clf, X_train, top_n=20)
    
    # Save the model
    joblib.dump(clf, "./model/xgboost_malnutrition_model_0_5.pkl")
    print("Model saved as xgboost_malnutrition_model_0_5.pkl")
    