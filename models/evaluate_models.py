import os
import json
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score


def find_best_models(output_root):
    best_models = {"MAL": None, "CAP": None}
    best_scores = {"MAL": -1, "CAP": -1}

    for folder in os.listdir(output_root):
        if folder.startswith("xgb_"):
            dataset_name = folder[4:]  # 去掉 xgb_
            model_folder = os.path.join(output_root, folder)
            performance_file = os.path.join(model_folder, f"{dataset_name}_xgb_performance.json")

            if os.path.exists(performance_file):
                with open(performance_file, 'r') as f:
                    perf = json.load(f)
                    score = perf.get("macro_recall", -1)

                if "MAL" in dataset_name and score > best_scores["MAL"]:
                    best_scores["MAL"] = score
                    best_models["MAL"] = {"name": dataset_name, "folder": model_folder}

                if "CAP" in dataset_name and score > best_scores["CAP"]:
                    best_scores["CAP"] = score
                    best_models["CAP"] = {"name": dataset_name, "folder": model_folder}
    print("best models:", best_models)
    return best_models


def load_top12_features(model_info):
    feature_path = os.path.join(model_info["folder"], f'{model_info["name"]}_top12_features.json')
    if os.path.exists(feature_path):
        with open(feature_path, 'r') as f:
            return json.load(f)
    else:
        print(f" Features not found: {feature_path}")
        return None

def split_dataset_by_label(dataset_path, label_col):
    print(f"📦 Loading dataset: {dataset_path}")
    df = pd.read_pickle(dataset_path)

    # 删除无关列（存在则删）
    df = df.drop(columns=["IDno", "Assessment_Date"], errors="ignore")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_with_selected_features(X_train, X_test, y_train, y_test, selected_features, model_type='logistic', return_report=False):
    print(f"\n Training model using features: {selected_features}")

    # 训练集与测试集只使用选定特征
    X_train_sel = X_train[selected_features]
    X_test_sel = X_test[selected_features]

    if model_type == 'logistic':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'svm':
        model = SVC(kernel='rbf', probability=True)
    elif model_type == 'knn':
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_type == 'nb':
        model = GaussianNB()
    elif model_type == 'mlp':
        model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=42)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # 模型训练与预测
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)

    # 评估指标
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)

    print(" Evaluation Metrics:")
    print(f"  - Accuracy : {acc:.4f}")
    print(f"  - Macro F1 : {f1:.4f}")
    print(f"  - Recall   : {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print("\n Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    if return_report:
        report = {
            "accuracy": acc,
            "macro_f1": f1,
            "macro_recall": recall,
            "macro_precision": precision,
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        }
        return model, report

    return model

def train_multiple_models_and_select_best(X_train, X_test, y_train, y_test, selected_features, model_types=None, sort_by='macro_recall'):
    if model_types is None:
        model_types = ['logistic', 'svm', 'knn', 'nb', 'mlp']

    results = []

    for model_type in model_types:
        print(f"\n🧪 Training model: {model_type.upper()}")
        model, report = train_with_selected_features(
            X_train, X_test, y_train, y_test,
            selected_features=selected_features,
            model_type=model_type,
            return_report=True
        )

        # 存储结果
        results.append({
            "model_type": model_type,
            "model": model,
            "report": report
        })

    # 选出最好的模型
    best_model = max(results, key=lambda x: x["report"][sort_by])

    print(f"\n✅ Best model: {best_model['model_type'].upper()} ({sort_by} = {best_model['report'][sort_by]:.4f})")
    return best_model




if __name__ == "__main__":
    import os

    # 设置路径
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_folder = os.path.join(BASE_DIR, "outputs")
    dataset_folder = os.path.join(BASE_DIR, "datasets")

    # 找出表现最好的模型（一个 CAP，一个 MAL）
    best_models = find_best_models(output_folder)

    # 遍历两个数据集
    for dataset_type in ["MAL", "CAP"]:
        model_info = best_models.get(dataset_type)
        if not model_info:
            print(f"⚠️ No best model found for {dataset_type}")
            continue

        print(f"\n✅ Best {dataset_type} model: {model_info['name']}")
        top_features = load_top12_features(model_info)
        if not top_features:
            print(f"⚠️ No top 12 features found for {dataset_type}")
            continue

        # 定义路径和标签列
        dataset_path = os.path.join(dataset_folder, f"{model_info['name']}.pkl")
        label_col = "Malnutrition" if dataset_type == "MAL" else "CAP_Nutrition"

        # Step 1: 划分数据
        X_train, X_test, y_train, y_test = split_dataset_by_label(dataset_path, label_col)

        # Step 2: 使用多个模型训练，并根据 macro_recall 选择最优
        best_model_result = train_multiple_models_and_select_best(
            X_train, X_test, y_train, y_test,
            selected_features=top_features,
            sort_by='macro_recall'
        )

        # 可选：输出结果
        print(f"\n {dataset_type} best model: {best_model_result['model_type'].upper()}")
        print(f"Macro Recall: {best_model_result['report']['macro_recall']:.4f}")
