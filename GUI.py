import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import numpy as np
import os

# 全局数据载入
DATA_PATH = "./datasets/mal_data.pkl"
df = pd.read_pickle(DATA_PATH)
df['IDno'] = df['IDno'].astype(str)  # ✅ 强制 ID 为字符串，保留前导零
df['Malnutrition'] = df['Malnutrition'].apply(lambda x: 0 if x in [0, 1, 2] else 1)
df = df.set_index("IDno")  # ✅ 把 IDno 设为索引
feature_columns = df.columns.difference(["Malnutrition", "Assessment_Date"])  # ✅ 放在 set_index 后
id_to_data = df

# 特征列
feature_columns = df.columns.difference(["Malnutrition", "Assessment_Date"])

# GUI 应用
class ModelComparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Comparison Tool")

        # 用户输入
        tk.Label(root, text="Please Enter User ID:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.id_entry = tk.Entry(root)
        self.id_entry.grid(row=0, column=1, padx=5, pady=5)

        # 模型选择按钮
        self.model1_path = tk.StringVar()
        self.model2_path = tk.StringVar()

        tk.Button(root, text="Select model 1", command=self.load_model1).grid(row=1, column=0, padx=5, pady=5)
        tk.Label(root, textvariable=self.model1_path, fg="blue").grid(row=1, column=1)

        tk.Button(root, text="Select model 2", command=self.load_model2).grid(row=2, column=0, padx=5, pady=5)
        tk.Label(root, textvariable=self.model2_path, fg="green").grid(row=2, column=1)

        # 对比按钮
        tk.Button(root, text="Result comparison", command=self.compare_models).grid(row=3, column=0, columnspan=2, pady=10)

        # 输出框
        self.output_text = tk.Text(root, height=10, width=50)
        self.output_text.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

    def load_model1(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
            self.model1_path.set(path)

    def load_model2(self):
        path = filedialog.askopenfilename(filetypes=[("Pickle files", "*.pkl")])
        if path:
            self.model2_path.set(path)

    def compare_models(self):
        user_id = self.id_entry.get()
        if not user_id:
            messagebox.showwarning("Error", "Please enter a user ID")
            return

        try:
            row = id_to_data.loc[user_id]
        except:
            messagebox.showerror("Error", f"Cannot find user with id: {user_id}")
            return

        try:
            model1 = joblib.load(self.model1_path.get())
            model2 = joblib.load(self.model2_path.get())
        except:
            messagebox.showerror("Load error", "Please load both models before comparing")
            return

        x = row[feature_columns].values.reshape(1, -1)

        pred1 = model1.predict(x)[0]
        prob1 = model1.predict_proba(x)[0][1] if hasattr(model1, "predict_proba") else "N/A"

        pred2 = model2.predict(x)[0]
        prob2 = model2.predict_proba(x)[0][1] if hasattr(model2, "predict_proba") else "N/A"

        result = f"""User ID: {user_id}

Model 1 ({os.path.basename(self.model1_path.get())}):
  Result: {pred1}
  Probability: {prob1}

Model 2 ({os.path.basename(self.model2_path.get())}):
  Result: {pred2}
  Probability: {prob2}
"""
        self.output_text.delete("1.0", tk.END)
        self.output_text.insert(tk.END, result)

# 启动 GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = ModelComparatorApp(root)
    root.mainloop()
