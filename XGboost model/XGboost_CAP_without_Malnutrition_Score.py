import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import shap

def split_cap_data(df):
    cols_to_drop = [df.columns[0], df.columns[1], 'CAP_Nutrition', 'Scale_BMI', 'Cap_cognitive']
    X = df.drop(columns=cols_to_drop)
    if 'Malnutrition' in X.columns:
        X = X.drop(columns=['Malnutrition'])
    
    y = df['CAP_Nutrition']
    
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
        # 'gamma': [0, 0.1, 0.2],
        # 'min_child_weight': [1, 3, 5]
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

def train_xgboost_simple(X_train, y_train):
    xgb_model = xgb.XGBClassifier( 
        eval_metric='logloss',
        max_depth=10,          
        learning_rate=0.01,    
        n_estimators=200,     
        subsample=0.8,     
        colsample_bytree=0.8, 
        gamma=0.1,            
        min_child_weight=1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def get_and_print_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print(f"XGBoost Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return accuracy

def plot_feature_importance(clf, X_train, top_n=20):
    importances = clf.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False).head(top_n)
    
    print(importance_df)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plt.savefig("./image/xgboost_cap_feature_importance_score_0_5.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def save_model(clf, X, filename):
    plt.figure(figsize=(40, 20))
    xgb.plot_tree(clf, num_trees=0, rankdir='LR', fmap='')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def use_shap_values(clf, X_train):
    explainer = shap.Explainer(clf)
    shap_values = explainer(X_train)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_train, plot_type="bar")
    plt.savefig("./image/xgboost_cap_shap_summary_plot.png", dpi=300, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    df = pd.read_pickle("./datasets/cap_data.pkl")
    X_train, X_test, y_train, y_test = split_cap_data(df)
    
    # Train XGBoost model with grid search
    # clf = train_xgboost_simple(X_train, y_train)
    clf = train_xgboost_with_grid_search(X_train, y_train)
    
    # Get and print accuracy
    accuracy = get_and_print_accuracy(clf, X_test, y_test)
    
    # Plot feature importance
    plot_feature_importance(clf, X_train, top_n=20)
    # Use SHAP values for model interpretation
    use_shap_values(clf, X_train)
    
    # Save the model to a file
    model_path = "./model/xgboost_cap_model.pkl"
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")
    
    # Save the decision tree visualization
    save_model(clf, X_train, "./tree/cap_without_Mal_score_xgboost_tree_model.pdf")