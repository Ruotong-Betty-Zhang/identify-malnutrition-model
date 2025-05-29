import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def split_cap_data(df):
    cols_to_drop = [df.columns[0], df.columns[1], 'CAP_Nutrition']
    X = df.drop(columns=cols_to_drop)
    y = df['CAP_Nutrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train):
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    return clf

def get_and_print_accuracy(clf, X_test, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
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
    #print top n features
    print(importance_df.head(top_n))

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    plt.savefig("./image/cap_without_Mal_score_feature_importance.png", dpi=300, bbox_inches='tight')
    plt.show()

def save_model_to_pdf(clf, X, filename="decision_tree_model.pdf"):
    plt.figure(figsize=(40, 20))  
    plot_tree(clf, 
            feature_names=X.columns, 
            class_names=True, 
            filled=True, 
            max_depth=9)  # max_depth limit

    plt.tight_layout()
    plt.savefig(filename)  # save to PDF
    plt.close()

if __name__ == "__main__":
    df = pd.read_pickle("./datasets/cap_data.pkl")
    X_train, X_test, y_train, y_test = split_cap_data(df)
    clf = train_decision_tree(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = get_and_print_accuracy(clf, X_test, y_test)
    plot_feature_importance(clf, X_train, top_n=20)
    save_model_to_pdf(clf, X_train, "./tree/decision_tree_model.pdf")
    
    
    
    
    