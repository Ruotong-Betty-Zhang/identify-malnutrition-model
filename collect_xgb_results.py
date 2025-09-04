import os, json, pandas as pd
from dotenv import load_dotenv

def collect_xgb_results(model_output: str, save_csv: bool = True, csv_name: str = "xgb_results_summary_with_knn_classifier.csv") -> pd.DataFrame:
    rows = []
    if not os.path.isdir(model_output):
        raise FileNotFoundError(f"MODEL_OUTPUT not found: {model_output}")

    for sub in sorted(os.listdir(model_output)):
        subdir = os.path.join(model_output, sub)
        if not (os.path.isdir(subdir) and sub.startswith("xgb_")):
            continue

        # 例如：xgb_CAP_2 -> 数据集名 CAP_2
        dataset = sub.replace("xgb_", "", 1)

        # 期望文件名：CAP_2_xgb_performance.json / CAP_2_xgb_best_params.json
        perf_path = os.path.join(subdir, f"{dataset}_xgb_performance.json")
        params_path = os.path.join(subdir, f"{dataset}_xgb_best_params.json")

        if not os.path.isfile(perf_path):
            # 兼容：有时目录名是 xgb_CAP_2，但内部文件用整个原始名（如 CAP_2.pkl 去掉后缀）
            # 兜底再找一次任何 *_xgb_performance.json
            candidates = [f for f in os.listdir(subdir) if f.endswith("_xgb_performance.json")]
            if candidates:
                perf_path = os.path.join(subdir, candidates[0])
                dataset = candidates[0].replace("_xgb_performance.json", "")
        if not os.path.isfile(perf_path):
            print(f"[Skip] No performance JSON in {subdir}")
            continue

        # 读取 performance
        with open(perf_path, "r") as f:
            perf = json.load(f)

        # 读取 best params（可选）
        best_params = {}
        if os.path.isfile(params_path):
            with open(params_path, "r") as f:
                best_params = json.load(f)
        else:
            # 兜底：寻找任何 *_xgb_best_params.json
            candidates = [f for f in os.listdir(subdir) if f.endswith("_xgb_best_params.json")]
            if candidates:
                with open(os.path.join(subdir, candidates[0]), "r") as f:
                    best_params = json.load(f)

        # 整理一行
        row = {
            "dataset": dataset,
            "accuracy": perf.get("accuracy"),
            "macro_recall": perf.get("macro_recall"),
            "macro_f1": perf.get("macro_f1"),
            "macro_precision": perf.get("macro_precision"),
            # 关键超参（缺了就 None）
            "max_depth": best_params.get("max_depth"),
            "learning_rate": best_params.get("learning_rate"),
            "n_estimators": best_params.get("n_estimators"),
            "subsample": best_params.get("subsample"),
            "colsample_bytree": best_params.get("colsample_bytree"),
        }
        rows.append(row)

    if not rows:
        raise RuntimeError("No XGBoost results found. Make sure you've run training and results exist in MODEL_OUTPUT/xgb_*.")

    df = pd.DataFrame(rows).sort_values(by=["dataset"]).reset_index(drop=True)

    # 保留 4 位小数显示更整齐
    for c in ["accuracy", "macro_recall", "macro_f1", "macro_precision"]:
        if c in df.columns:
            df[c] = df[c].map(lambda x: round(x, 4) if isinstance(x, (int, float)) else x)

    if save_csv:
        out_csv = os.path.join(model_output, csv_name)
        df.to_csv(out_csv, index=False)
        print(f"[OK] Summary CSV saved to: {out_csv}")

    # 也打印一下方便快速查看
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df)

    return df

if __name__ == "__main__":
    load_dotenv(dotenv_path=".env", override=True)
    model_output = os.getenv("MODEL_OUTPUT")
    if not model_output:
        raise EnvironmentError("Missing env var MODEL_OUTPUT")
    collect_xgb_results(model_output)
