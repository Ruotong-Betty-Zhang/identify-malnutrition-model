import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import joblib
import shap
import threading
import warnings
import re
from matplotlib.figure import Figure

warnings.filterwarnings('ignore')

class ModelEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model Evaluation Tool")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        self._mpl_refs = [] 
        
        # Initialize variables
        self.dataset = None
        self.model = None
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.feature_names = None
        self.target_name = None
        self.label_encoders = {}
        self.model_feature_names = None
        self.feature_importances = None
        self.alignment_info = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure row and column weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Dataset upload section
        ttk.Label(main_frame, text="Upload Dataset:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.dataset_path, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_dataset).grid(row=0, column=2, padx=5)
        
        # Data format selection
        ttk.Label(main_frame, text="Data Format:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.data_format = tk.StringVar(value="pkl")
        format_frame = ttk.Frame(main_frame)
        format_frame.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Radiobutton(format_frame, text="PKL", variable=self.data_format, value="pkl").pack(side=tk.LEFT)
        ttk.Radiobutton(format_frame, text="CSV", variable=self.data_format, value="csv").pack(side=tk.LEFT)
        
        # Data preprocessing options
        ttk.Label(main_frame, text="Preprocessing:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.preprocess_frame = ttk.Frame(main_frame)
        self.preprocess_frame.grid(row=2, column=1, sticky=tk.W, padx=5)
        
        self.handle_categorical = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.preprocess_frame, text="Encode categorical", variable=self.handle_categorical).pack(side=tk.LEFT)
        
        self.handle_datetime = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.preprocess_frame, text="Process datetime", variable=self.handle_datetime).pack(side=tk.LEFT)
        
        self.remove_id = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.preprocess_frame, text="Remove ID columns", variable=self.remove_id).pack(side=tk.LEFT)
        
        # Target variable selection
        ttk.Label(main_frame, text="Target Variable:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(main_frame, textvariable=self.target_var, state="readonly", width=20)
        self.target_combo.grid(row=3, column=1, sticky=tk.W, padx=5)
        
        # Model upload section
        ttk.Label(main_frame, text="Upload Model:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.model_path = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.model_path, width=50).grid(row=4, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_model).grid(row=4, column=2, padx=5)
        
        # Execute button
        ttk.Button(main_frame, text="Run Evaluation", command=self.run_evaluation).grid(row=5, column=1, pady=10)
        
        # Create notebook to display results
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Accuracy tab
        self.accuracy_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.accuracy_frame, text="Accuracy")
        
        # Confusion Matrix tab
        self.cm_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.cm_frame, text="Confusion Matrix")
        
        # Classification Report tab
        self.report_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.report_frame, text="Classification Report")
        
        # Feature Importance tab
        self.fi_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.fi_frame, text="Feature Importance")
        
        # SHAP Analysis tab
        self.shap_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.shap_frame, text="SHAP Analysis")
        
        # Data Preview tab
        self.preview_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.preview_frame, text="Data Preview")
        
        # Feature Alignment Info tab
        self.alignment_frame = ttk.Frame(self.notebook, padding="10")
        self.notebook.add(self.alignment_frame, text="Feature Alignment")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        ttk.Label(main_frame, textvariable=self.status_var).grid(row=7, column=0, columnspan=3, sticky=tk.W)
        
        # Configure weights
        main_frame.rowconfigure(6, weight=1)
        main_frame.columnconfigure(1, weight=1)
    
    def get_feature_importances(self):
        """获取特征重要性"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # scikit-learn 模型
                return self.model.feature_importances_
            elif hasattr(self.model, 'get_booster'):
                # XGBoost 模型
                return self.model.get_booster().get_score(importance_type='weight')
            elif hasattr(self.model, 'coef_'):
                # 线性模型系数
                if len(self.model.coef_.shape) == 1:
                    return np.abs(self.model.coef_)
                else:
                    return np.mean(np.abs(self.model.coef_), axis=0)
            else:
                return None
        except:
            return None
    
    def align_features_with_model(self, X, model_feature_names):
        """确保数据特征与模型训练时的特征匹配"""
        X_aligned = X.copy()
        
        # 获取当前数据的特征名称
        if hasattr(X_aligned, 'columns'):
            current_features = list(X_aligned.columns)
        else:
            current_features = [f"feature_{i}" for i in range(X_aligned.shape[1])]
            # 如果模型有特征名称，使用模型的名称
            if model_feature_names and len(model_feature_names) == X_aligned.shape[1]:
                X_aligned = pd.DataFrame(X_aligned, columns=model_feature_names)
                current_features = model_feature_names
        
        alignment_info = {
            "missing_features": [],
            "extra_features": [],
            "current_features": current_features,
            "model_features": model_feature_names,
            "aligned": False,
            "details": ""
        }
        
        if model_feature_names:
            # 找出缺失的特征
            missing_features = set(model_feature_names) - set(current_features)
            # 找出多余的特征
            extra_features = set(current_features) - set(model_feature_names)
            
            alignment_info["missing_features"] = list(missing_features)
            alignment_info["extra_features"] = list(extra_features)
            
            if missing_features:
                # 为缺失的特征添加默认值（0）
                for feature in missing_features:
                    X_aligned[feature] = 0
                alignment_info["details"] += f"Added {len(missing_features)} missing features with default values (0)\n"
                alignment_info["details"] += f"Missing features: {list(missing_features)}\n"
            
            if extra_features:
                # 移除多余的特征
                X_aligned = X_aligned.drop(columns=list(extra_features))
                alignment_info["details"] += f"Removed {len(extra_features)} extra features not used in model training\n"
                alignment_info["details"] += f"Extra features: {list(extra_features)}\n"
            
            # 确保特征顺序与模型一致
            try:
                X_aligned = X_aligned[model_feature_names]
                alignment_info["aligned"] = True
                alignment_info["details"] += "Features successfully aligned with model\n"
            except Exception as e:
                alignment_info["details"] += f"Feature alignment error: {str(e)}\n"
        else:
            alignment_info["details"] = "No model feature names available, using original features\n"
            alignment_info["aligned"] = True
        
        self.alignment_info = alignment_info
        return X_aligned, alignment_info
    
    def get_model_feature_names(self):
        """获取模型训练时使用的特征名称"""
        try:
            # 尝试从模型属性获取特征名称
            if hasattr(self.model, 'feature_names_in_'):
                return list(self.model.feature_names_in_)
            elif hasattr(self.model, 'get_booster'):
                # XGBoost模型
                return self.model.get_booster().feature_names
            elif hasattr(self.model, 'feature_importances_'):
                # 如果有特征重要性但没名称，创建默认名称
                n_features = len(self.model.feature_importances_)
                return [f"feature_{i}" for i in range(n_features)]
            else:
                return None
        except:
            return None
    
    def preprocess_data(self, X):
        """预处理数据"""
        X_processed = X.copy()
        preprocessing_info = {"removed_columns": [], "encoded_columns": []}
        
        # 处理每列的数据类型
        for col in list(X_processed.columns):
            if col not in X_processed.columns:
                continue
                
            col_dtype = X_processed[col].dtype
            
            # 更智能的ID列识别 - 只移除明确的ID列，不移除iD1这样的特征
            if self.remove_id.get():
                # 只移除明确的ID标识列，不移除看起来像特征名的列
                is_real_id = (col.lower() in ['id', 'idno', 'id_num', 'id_number', 'patient_id', 'sample_id'] or
                            col.lower().endswith('_id') or
                            col.lower().startswith('id_'))
                
                # 特别保留 iD1, iD2, iD3a, iD3b, iD4a, iD4b 等特征
                is_important_feature = (col in ['iD1', 'iD2', 'iD3a', 'iD3b', 'iD4a', 'iD4b'] or
                                    col.startswith('iD') and len(col) > 2 and col[2:].isdigit())
                
                if is_real_id and not is_important_feature:
                    X_processed = X_processed.drop(columns=[col])
                    preprocessing_info["removed_columns"].append(col)
                    continue
            
            # 处理日期时间列
            from pandas.api.types import is_datetime64_any_dtype

            # 处理日期时间列
            if self.handle_datetime.get() and is_datetime64_any_dtype(X_processed[col]):
                X_processed = X_processed.drop(columns=[col])
                preprocessing_info["removed_columns"].append(col)
                continue

            
            # 处理分类变量
            elif self.handle_categorical.get() and col_dtype == 'object':
                try:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                    self.label_encoders[col] = le
                    preprocessing_info["encoded_columns"].append(col)
                except:
                    X_processed = X_processed.drop(columns=[col])
                    preprocessing_info["removed_columns"].append(col)
        
        return X_processed, preprocessing_info
    
    def browse_dataset(self):
        file_types = [("PKL files", "*.pkl"), ("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if file_path:
            self.dataset_path.set(file_path)
            try:
                if self.data_format.get() == "pkl":
                    self.dataset = joblib.load(file_path)
                else:
                    self.dataset = pd.read_csv(file_path)
                
                if isinstance(self.dataset, pd.DataFrame):
                    self.target_combo['values'] = list(self.dataset.columns)
                    if self.dataset.columns.size > 0:
                        self.target_var.set(self.dataset.columns[-1])
                
                self.update_data_preview()
                self.status_var.set("Dataset loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
                self.status_var.set("Failed to load dataset")
    
    def browse_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl;*.joblib"), ("All files", "*.*")])
        if file_path:
            self.model_path.set(file_path)
            try:
                self.model = joblib.load(file_path)
                self.model_feature_names = self.get_model_feature_names()
                self.feature_importances = self.get_feature_importances()
                self.status_var.set("Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Error loading model: {str(e)}")
                self.status_var.set("Failed to load model")
    
    def run_evaluation(self):
        if self.dataset is None or self.model is None:
            messagebox.showerror("Error", "Please upload both dataset and model first")
            return
        
        thread = threading.Thread(target=self._run_evaluation_thread)
        thread.daemon = True
        thread.start()
    
    def _run_evaluation_thread(self):
        self.status_var.set("Running evaluation...")
        try:
            if not isinstance(self.dataset, pd.DataFrame):
                messagebox.showerror("Error", "Unsupported data format")
                return
            
            target = self.target_var.get()
            if not target:
                messagebox.showerror("Error", "Please select target variable")
                return
            
            X = self.dataset.drop(columns=[target])
            y = self.dataset[target]
            
            # 预处理数据
            X_processed, preprocessing_info = self.preprocess_data(X)
            
            # 确保特征与模型匹配
            X_aligned, alignment_info = self.align_features_with_model(X_processed, self.model_feature_names)
            if not alignment_info["aligned"]:
                raise ValueError("Feature alignment failed")
            
            # 显示详细的匹配信息
            if alignment_info["missing_features"] or alignment_info["extra_features"]:
                detailed_message = "Feature Alignment Details:\n\n"
                detailed_message += alignment_info["details"]
                detailed_message += f"\nCurrent features: {len(alignment_info['current_features'])}"
                detailed_message += f"\nModel features: {len(alignment_info['model_features'])}"
                detailed_message += f"\nFinal features: {len(X_aligned.columns)}"
                
                self.root.after(0, lambda: messagebox.showinfo("Feature Alignment Details", detailed_message))
            
            # 分割数据
            X_train, X_test, y_train, y_test = train_test_split(X_aligned, y, test_size=0.2, random_state=42)
            self.X_test = X_test
            self.y_test = y_test
            
            # 进行预测
            self.y_pred = self.model.predict(X_test)
            
            # 计算指标
            # 统一类标签（基于全量 y）
            classes = np.unique(y)

            accuracy = accuracy_score(y_test, self.y_pred)
            cm = confusion_matrix(y_test, self.y_pred, labels=classes)
            report = classification_report(y_test, self.y_pred, labels=classes, zero_division=0)

            self.root.after(0, self.update_accuracy_tab, accuracy, preprocessing_info)
            self.root.after(0, self.update_cm_tab, cm, classes)  # 传同一份 classes
            self.root.after(0, self.update_report_tab, report)
            self.root.after(0, self.update_feature_importance_tab, X_aligned.columns)
            self.root.after(0, self.update_shap_tab, X_test)
            self.root.after(0, self.update_alignment_info_tab, alignment_info)

            
            self.status_var.set("Evaluation completed")
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error during evaluation: {str(e)}"))
            self.status_var.set("Evaluation failed")
    
    def update_alignment_info_tab(self, alignment_info):
        for widget in self.alignment_frame.winfo_children():
            widget.destroy()
        
        text_widget = scrolledtext.ScrolledText(self.alignment_frame, wrap=tk.WORD, height=15)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_text = "=== FEATURE ALIGNMENT DETAILS ===\n\n"
        info_text += alignment_info["details"] + "\n"
        
        info_text += f"Total current features: {len(alignment_info['current_features'])}\n"
        info_text += f"Total model features: {len(alignment_info['model_features']) if alignment_info['model_features'] else 'Unknown'}\n"
        info_text += f"Missing features: {len(alignment_info['missing_features'])}\n"
        info_text += f"Extra features: {len(alignment_info['extra_features'])}\n\n"
        
        if alignment_info["missing_features"]:
            info_text += "MISSING FEATURES (added with default values):\n"
            for i, feature in enumerate(alignment_info["missing_features"]):
                info_text += f"  {i+1}. {feature}\n"
            info_text += "\n"
        
        if alignment_info["extra_features"]:
            info_text += "EXTRA FEATURES (removed):\n"
            for i, feature in enumerate(alignment_info["extra_features"]):
                info_text += f"  {i+1}. {feature}\n"
            info_text += "\n"
        
        info_text += "FINAL FEATURE LIST:\n"
        if hasattr(self.X_test, 'columns'):
            for i, feature in enumerate(self.X_test.columns):
                info_text += f"  {i+1}. {feature}\n"
        
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)
    
    def update_feature_importance_tab(self, feature_names):
        for widget in self.fi_frame.winfo_children():
            widget.destroy()
        
        if self.feature_importances is None:
            ttk.Label(self.fi_frame, text="Feature importance not available for this model").pack(pady=12)
            return
        
        try:
            # 创建特征重要性数据
            if isinstance(self.feature_importances, dict):
                # XGBoost格式的特征重要性
                fi_data = []
                for feature, importance in self.feature_importances.items():
                    fi_data.append((feature, importance))
                fi_data.sort(key=lambda x: x[1], reverse=True)
            else:
                # 数组格式的特征重要性
                fi_data = list(zip(feature_names, self.feature_importances))
                fi_data.sort(key=lambda x: x[1], reverse=True)
            
            # 创建特征重要性图表
            fig = Figure(figsize=(14, 8))              # 新
            ax = fig.add_subplot(111)
            top_k = 12
            features = [x[0] for x in fi_data[:top_k]]
            importances = [x[1] for x in fi_data[:top_k]]
                        
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 12 Feature Importances')
            
            canvas = FigureCanvasTkAgg(fig, master=self.fi_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)  
            
            # 添加详细的特征重要性表格
            frame = ttk.Frame(self.fi_frame)
            frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            tree = ttk.Treeview(frame, columns=('Feature', 'Importance'), show='headings', height=10)
            tree.heading('Feature', text='Feature')
            tree.heading('Importance', text='Importance')
            tree.column('Feature', width=300)
            tree.column('Importance', width=100)
            
            for feature, importance in fi_data:
                tree.insert('', 'end', values=(feature, f"{importance:.6f}"))
            
            scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            ttk.Label(self.fi_frame, text=f"Feature importance error: {str(e)}").pack(pady=20)
    
    def update_accuracy_tab(self, accuracy, preprocessing_info):
        for widget in self.accuracy_frame.winfo_children():
            widget.destroy()
        
        # 显示准确率
        ttk.Label(self.accuracy_frame, text=f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", 
                 font=("Arial", 16)).pack(pady=10)
        
        # 显示预处理信息
        if preprocessing_info["removed_columns"]:
            info_text = f"Removed columns: {', '.join(preprocessing_info['removed_columns'])}"
            ttk.Label(self.accuracy_frame, text=info_text, wraplength=600).pack(pady=5)
        
        if preprocessing_info["encoded_columns"]:
            info_text = f"Encoded columns: {', '.join(preprocessing_info['encoded_columns'])}"
            ttk.Label(self.accuracy_frame, text=info_text, wraplength=600).pack(pady=5)
    
    def update_cm_tab(self, cm, class_names):
        for widget in self.cm_frame.winfo_children():
            widget.destroy()
        
        # fig, ax = plt.subplots(figsize=(8, 6))
        fig = Figure(figsize=(8, 6))                # 新
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._mpl_refs.append(canvas)
    
    def update_report_tab(self, report):
        for widget in self.report_frame.winfo_children():
            widget.destroy()
        
        text_widget = scrolledtext.ScrolledText(self.report_frame, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        text_widget.insert(tk.END, "Classification Report:\n\n")
        text_widget.insert(tk.END, report)
        text_widget.config(state=tk.DISABLED)
    
    def update_shap_tab(self, X_test):
        for widget in self.shap_frame.winfo_children():
            widget.destroy()
        
        try:
            # 使用您XGBoost代码中的数值化清洗方法
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

            X_num = _sanitize(X_test)
            if not isinstance(X_num, pd.DataFrame):
                X_num = pd.DataFrame(X_num, columns=[f"Feature_{i}" for i in range(X_num.shape[1])])

            # 解释器
            try:
                explainer = shap.TreeExplainer(self.model, model_output="raw")
                sv = explainer.shap_values(X_num)
            except Exception:
                try:
                    explainer = shap.LinearExplainer(self.model, X_num, feature_dependence="independent")
                    sv = explainer.shap_values(X_num)
                except Exception as e2:
                    print(f"[SHAP] Explainer failed: {e2}")
                    return

            # ---- 统一成 list[n_classes]，每个元素形状为 (n_samples, n_features) ----
            def _split_sv_to_list(sv_val, n_classes_hint=None):
                # sv 可能是 list、(n_samples,n_features)、(n_samples,n_classes,n_features)
                # 或 (n_samples,n_features,n_classes)
                if isinstance(sv_val, list):
                    return sv_val

                arr = np.asarray(sv_val)
                if arr.ndim == 2:
                    # 二分类/回归：一个输出
                    return [arr]

                if arr.ndim == 3:
                    # 猜测类别所在维度：优先用 model.classes_ 或 y_test 的唯一类数
                    if n_classes_hint is None:
                        n_classes_hint = len(getattr(self.model, "classes_", [])) or len(np.unique(self.y_test))

                    # 优先匹配“某一维 == n_classes_hint”
                    for axis, size in enumerate(arr.shape):
                        if n_classes_hint and size == n_classes_hint:
                            return [np.take(arr, i, axis=axis) for i in range(size)]

                    # 次优：若最后一维比较小（<=10）当作类别维
                    if arr.shape[-1] <= 10:
                        return [arr[..., i] for i in range(arr.shape[-1])]
                    # 再次优：若中间维比较小（<=10）当作类别维
                    if arr.shape[1] <= 10:
                        return [arr[:, i, :] for i in range(arr.shape[1])]
                    # 兜底：按第一维切（多数树模型的旧返回）
                    return [arr[i] for i in range(arr.shape[0])]

                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

            # 基于 hint 进行拆分
            n_classes_hint = len(getattr(self.model, "classes_", [])) or len(np.unique(self.y_test))
            sv_list = _split_sv_to_list(sv, n_classes_hint=n_classes_hint)

            # ---- 类别标签对齐：以 sv_list 的长度为准，避免越界 ----
            model_classes = getattr(self.model, "classes_", None)
            if isinstance(model_classes, (list, np.ndarray)) and len(model_classes) == len(sv_list):
                class_ids = list(model_classes)
            else:
                # 用测试集出现的类与 sv_list 对齐，否则退化为 0..n-1
                test_classes = list(np.unique(self.y_test))
                if len(test_classes) == len(sv_list):
                    class_ids = test_classes
                else:
                    class_ids = list(range(len(sv_list)))
            
            # 创建主选项卡
            notebook = ttk.Notebook(self.shap_frame)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # 计算每个类别的平均SHAP绝对值（特征重要性）
            shap_importance_by_class = {}
            for ci, sv_c in enumerate(sv_list):
                # 平均绝对SHAP值作为特征重要性
                mean_abs_shap = np.mean(np.abs(sv_c), axis=0)
                shap_importance_by_class[class_ids[ci]] = mean_abs_shap

            # 为每个类别创建选项卡
            for ci, sv_c in enumerate(sv_list):
                class_frame = ttk.Frame(notebook)
                notebook.add(class_frame, text=f"Class {class_ids[ci]}")

                # 创建子选项卡（Beeswarm和雷达图）
                class_notebook = ttk.Notebook(class_frame)
                class_notebook.pack(fill=tk.BOTH, expand=True)

                # Beeswarm图标签页
                beeswarm_frame = ttk.Frame(class_notebook)
                class_notebook.add(beeswarm_frame, text="Beeswarm Plot")

                mask_pos = (np.array(self.y_test) == class_ids[ci])
                if mask_pos.sum() >= 5:
                    X_num_disp = self._with_readable_columns(X_num)

                    # —— 关键：为 SHAP 专门创建一个新的 pyplot figure —— #
                    fig_beeswarm = plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        sv_c[mask_pos], X_num_disp.iloc[mask_pos],
                        plot_type="dot", max_display=12, show=False
                    )
                    fig_beeswarm.tight_layout()

                    canvas_beeswarm = FigureCanvasTkAgg(fig_beeswarm, master=beeswarm_frame)
                    canvas_beeswarm.draw()
                    canvas_beeswarm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                    # 保存引用并从 pyplot 注销，避免影响后续图形
                    self._mpl_refs.append(canvas_beeswarm)
                    plt.close(fig_beeswarm)
                else:
                    ttk.Label(
                        beeswarm_frame,
                        text=f"Not enough samples ({mask_pos.sum()}) for beeswarm plot"
                    ).pack(pady=20)

                # 雷达图标签页
                radar_frame = ttk.Frame(class_notebook)
                class_notebook.add(radar_frame, text="Radar Chart")

                # 创建雷达图
                self._create_shap_radar_chart(radar_frame, shap_importance_by_class[class_ids[ci]], 
                                            X_num.columns, class_ids[ci])

            # # 添加比较所有类别的雷达图标签页
            # comparison_frame = ttk.Frame(notebook)
            # notebook.add(comparison_frame, text="Class Comparison")

            # 创建比较所有类别的雷达图
            # shap_importance_by_class = {}
            # for ci, sv_c in enumerate(sv_list):
            #     mean_abs_shap = np.mean(np.abs(sv_c), axis=0)  # (n_features,)
            #     shap_importance_by_class[class_ids[ci]] = mean_abs_shap

            comparison_frame = ttk.Frame(notebook)
            notebook.add(comparison_frame, text="Class Comparison")

            available_classes = list(shap_importance_by_class.keys())
            self._create_comparison_radar_chart(
                comparison_frame,
                shap_importance_by_class,
                X_num.columns,
                available_classes
            )
                
        except Exception as e:
            error_label = ttk.Label(
                self.shap_frame, 
                text=f"SHAP analysis error: {str(e)}",
                justify=tk.LEFT
            )
            error_label.pack(pady=20)
            print(f"SHAP error: {e}")

    def _with_readable_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """返回用于绘图的列名副本（可读名称）"""
        # 如果没有特征名称映射，使用原始列名
        if not hasattr(self, 'feature_name_map') or not self.feature_name_map:
            return X.copy()
        
        # 创建可读的列名
        readable_columns = []
        for col in X.columns:
            base_col = re.sub(r"(_scaled|_std|_lag\d+|_zscore)$", "", str(col))
            readable_name = self.feature_name_map.get(base_col, col)
            readable_columns.append(readable_name)
        
        X_disp = X.copy()
        X_disp.columns = readable_columns
        return X_disp

    def _compose_display_labels(self, names, codes):
        """名称与代码合并为唯一标签：Name (Code)"""
        labels = []
        for n, c in zip(names, codes):
            if str(n) == str(c):
                labels.append(str(n))
            else:
                labels.append(f"{n} ({c})")
        return labels

    def _map_feature_names(self, cols):
        """将特征代码映射为可读名称"""
        if not hasattr(self, 'feature_name_map') or not self.feature_name_map:
            return list(cols)
        
        readable_names = []
        for col in cols:
            base_col = re.sub(r"(_scaled|_std|_lag\d+|_zscore)$", "", str(col))
            readable_names.append(self.feature_name_map.get(base_col, col))
    
        return readable_names
        
    def update_data_preview(self):
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
        
        if self.dataset is None:
            return
        
        try:
            if isinstance(self.dataset, pd.DataFrame):
                info_text = f"Shape: {self.dataset.shape}\nColumns: {list(self.dataset.columns)}\n\nFirst 5 rows:\n{str(self.dataset.head())}"
            
            text_widget = scrolledtext.ScrolledText(self.preview_frame, wrap=tk.WORD, height=15)
            text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            text_widget.insert(tk.END, info_text)
            text_widget.config(state=tk.DISABLED)
            
        except Exception as e:
            ttk.Label(self.preview_frame, text=f"Preview error: {str(e)}").pack(pady=20)
    
    def _create_shap_radar_chart(self, parent, shap_importance, feature_names, class_id, top_n=10):
        """为单个类别创建SHAP重要性雷达图"""
        try:
            # 获取最重要的top_n个特征
            top_indices = np.argsort(shap_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = shap_importance[top_indices]
            
            # 归一化到0-1范围
            max_val = np.max(top_importance)
            if max_val > 0:
                normalized_importance = top_importance / max_val
            else:
                normalized_importance = top_importance
            
            # 雷达图需要闭合（第一个和最后一个点相同）
            values = np.concatenate([normalized_importance, [normalized_importance[0]]])
            features = list(top_features) + [top_features[0]]
            
            # 创建雷达图
            fig = Figure(figsize=(10, 8))                                           # 新
            ax = fig.add_subplot(111, polar=True)
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, len(features), endpoint=True).tolist()
            
            # 绘制雷达图
            ax.fill(angles, values, color='red', alpha=0.25)
            ax.plot(angles, values, color='red', linewidth=2)
            
            # 设置特征标签
            ax.set_xticks(angles[:-1])
            feature_labels = self._map_feature_names(features[:-1])
            ax.set_xticklabels(feature_labels, fontsize=9)
            
            # 设置标题和网格
            ax.set_title(f'Top {top_n} Features - Class {class_id}\n(by mean |SHAP value|)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            
            # 设置y轴标签
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            # 嵌入到GUI
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)
            
            # 添加数值表格
            table_frame = ttk.Frame(parent)
            table_frame.pack(fill=tk.X, padx=10, pady=5)

            top_n = min(top_n, len(feature_names))
            
            # 创建表格数据
            table_data = []
            for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
                display_name = self._map_feature_names([feature])[0]
                table_data.append((i+1, display_name, f"{importance:.6f}"))
            
            # 创建表格
            tree = ttk.Treeview(table_frame, columns=('Rank', 'Feature', 'Importance'), 
                            show='headings', height=min(6, len(table_data)))
            tree.heading('Rank', text='Rank')
            tree.heading('Feature', text='Feature')
            tree.heading('Importance', text='Mean |SHAP|')
            tree.column('Rank', width=50, anchor='center')
            tree.column('Feature', width=300)
            tree.column('Importance', width=100, anchor='center')
            
            for row in table_data:
                tree.insert('', 'end', values=row)
            
            scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
        except Exception as e:
            ttk.Label(parent, text=f"Radar chart error: {str(e)}").pack(pady=20)

    def _create_comparison_radar_chart(self, parent, shap_importance_dict, feature_names, class_ids, top_n=8):
        """创建比较所有类别的雷达图"""
        try:
            # 获取所有类别共同的最重要特征
            all_importances = np.stack(list(shap_importance_dict.values()))
            mean_importance = np.mean(all_importances, axis=0)
            top_indices = np.argsort(mean_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            
            # 准备数据
            categories = top_features
            N = len(categories)
            
            # 计算角度
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # 闭合
            
            # 创建雷达图
            fig = Figure(figsize=(12, 9))                                           # 新
            ax = fig.add_subplot(111, polar=True)
            
            # 为每个类别绘制线
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
            for i, class_id in enumerate(class_ids):
                importance = shap_importance_dict[class_id][top_indices]
                max_val = np.max(importance)
                if max_val > 0:
                    normalized = importance / max_val
                else:
                    normalized = importance
                
                # 闭合数据
                values = np.concatenate([normalized, [normalized[0]]])
                
                ax.plot(angles, values, linewidth=2, linestyle='solid', 
                    label=f'Class {class_id}', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.1)
            
            # 设置特征标签
            feature_labels = self._map_feature_names(categories)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=9)
            
            # 设置标题和图例
            ax.set_title(f'Top {top_n} Features Comparison Across Classes\n(Normalized by class max)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            # 设置网格
            ax.grid(True)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
            ax.set_ylim(0, 1.1)
            
            plt.tight_layout()
            
            # 嵌入到GUI
            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)
            
        except Exception as e:
            ttk.Label(parent, text=f"Comparison radar chart error: {str(e)}").pack(pady=20)

def main():
    root = tk.Tk()
    app = ModelEvaluationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()