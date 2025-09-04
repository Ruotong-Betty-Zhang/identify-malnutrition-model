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
warnings.filterwarnings('ignore')

class ModelEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model Evaluation Tool")
        self.root.geometry("1200x800")
        self.root.resizable(True, True)
        
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
            if self.handle_datetime.get() and (col_dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower()):
                try:
                    X_processed = X_processed.drop(columns=[col])
                    preprocessing_info["removed_columns"].append(col)
                except:
                    X_processed = X_processed.drop(columns=[col])
                    preprocessing_info["removed_columns"].append(col)
            
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
            accuracy = accuracy_score(y_test, self.y_pred)
            cm = confusion_matrix(y_test, self.y_pred)
            report = classification_report(y_test, self.y_pred)
            
            # 更新UI
            self.root.after(0, self.update_accuracy_tab, accuracy, preprocessing_info)
            self.root.after(0, self.update_cm_tab, cm, np.unique(y))
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
            ttk.Label(self.fi_frame, text="Feature importance not available for this model").pack(pady=20)
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
            fig, ax = plt.subplots(figsize=(12, 8))
            features = [x[0] for x in fi_data[:20]]  # 显示前20个最重要的特征
            importances = [x[1] for x in fi_data[:20]]
            
            y_pos = np.arange(len(features))
            ax.barh(y_pos, importances, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 20 Feature Importances')
            
            canvas = FigureCanvasTkAgg(fig, master=self.fi_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
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
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        
        canvas = FigureCanvasTkAgg(fig, master=self.cm_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
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
            if hasattr(self.model, 'predict_proba'):
                explainer = shap.Explainer(self.model, X_test)
                shap_values = explainer(X_test)
                
                fig, ax = plt.subplots(figsize=(12, 8))
                shap.summary_plot(shap_values, X_test, show=False)
                
                canvas = FigureCanvasTkAgg(fig, master=self.shap_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            else:
                ttk.Label(self.shap_frame, text="SHAP analysis not supported").pack(pady=20)
                
        except Exception as e:
            ttk.Label(self.shap_frame, text=f"SHAP error: {str(e)}").pack(pady=20)
    
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

def main():
    root = tk.Tk()
    app = ModelEvaluationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()