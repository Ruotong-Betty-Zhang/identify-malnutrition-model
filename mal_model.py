import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier

def split_mal_data(df):
    cols_to_drop = [df.columns[0], df.columns[1], 'Malnutrition']
    X = df.drop(columns=cols_to_drop)
    y = df['Malnutrition']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def get_and_print_accuracy(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return accuracy

def plot_feature_importance(clf, X_train, top_n=20):
    feature_importances = clf.feature_importances_
    features = X_train.columns
    importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    importance_df = importance_df.head(top_n)
    #print top n features
    print(importance_df.head(top_n))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plt.savefig("./image/mal_feature_importance_score_0_5.png", dpi=300, bbox_inches='tight')
    plt.show()
    
def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=30)
    clf.fit(X_train, y_train)
    return clf
    
def save_model_to_pdf(clf, X, filename="decision_tree_model.pdf"):
    plt.figure(figsize=(40, 20))  
    plot_tree(clf, 
              feature_names=X.columns, 
              class_names=True, 
              filled=True, 
              fontsize=10)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = pd.read_pickle("./datasets/mal_data.pkl")
    X_train, X_test, y_train, y_test = split_mal_data(df)
    # Train the decision tree classifier
    # clf = train_decision_tree(X_train, y_train)
    # accuracy = get_and_print_accuracy(clf, X_test, y_test)
    # plot_feature_importance(clf, X_train, top_n=20)
    # save_model_to_pdf(clf, X_train, "./tree/mal_decision_tree_model_score_0_5.pdf")
    
    # Train the random forest classifier
    rf_clf = train_random_forest(X_train, y_train)
    rf_accuracy = get_and_print_accuracy(rf_clf, X_test, y_test)
    plot_feature_importance(rf_clf, X_train, top_n=20)
    save_model_to_pdf(rf_clf, X_train, "./tree/mal_random_forest_model_score_0_5.pdf")
    