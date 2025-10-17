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

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)

        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        # Only bind mouse wheel when hovering; unbind when leaving
        for w in (self, self.canvas, self.inner):
            w.bind("<Enter>", self._bind_mousewheel)
            w.bind("<Leave>", self._unbind_mousewheel)

    @property
    def body(self):
        return self.inner

    # Windows / macOS: <MouseWheel> (works with macOS's small delta values too)
    def _on_mousewheel(self, event):
        step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")  # Adjust 3 to control speed

    # Linux: Button-4/5
    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")

    def _bind_mousewheel(self, _=None):
        # Only take over the mouse wheel when hovering over this widget
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _=None):
        # Mouse leaves: release the mouse wheel, don't affect other pages/widgets
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


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
        """Get feature importances"""
        try:
            if hasattr(self.model, 'feature_importances_'):
                # scikit-learn model
                return self.model.feature_importances_
            elif hasattr(self.model, 'get_booster'):
                # XGBoost model
                return self.model.get_booster().get_score(importance_type='weight')
            elif hasattr(self.model, 'coef_'):
                # Linear model coefficients
                if len(self.model.coef_.shape) == 1:
                    return np.abs(self.model.coef_)
                else:
                    return np.mean(np.abs(self.model.coef_), axis=0)
            else:
                return None
        except:
            return None
    
    def align_features_with_model(self, X, model_feature_names):
        """Ensure data features match those used during model training"""
        X_aligned = X.copy()
        
        # Get feature names from current data
        if hasattr(X_aligned, 'columns'):
            current_features = list(X_aligned.columns)
        else:
            current_features = [f"feature_{i}" for i in range(X_aligned.shape[1])]
            # If model has feature names, use them
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
            # Find missing features
            missing_features = set(model_feature_names) - set(current_features)
            # Find extra features
            extra_features = set(current_features) - set(model_feature_names)
            
            alignment_info["missing_features"] = list(missing_features)
            alignment_info["extra_features"] = list(extra_features)
            
            if missing_features:
                # Add default values (0) for missing features
                for feature in missing_features:
                    X_aligned[feature] = 0
                alignment_info["details"] += f"Added {len(missing_features)} missing features with default values (0)\n"
                alignment_info["details"] += f"Missing features: {list(missing_features)}\n"
            
            if extra_features:
                # Remove extra features
                X_aligned = X_aligned.drop(columns=list(extra_features))
                alignment_info["details"] += f"Removed {len(extra_features)} extra features not used in model training\n"
                alignment_info["details"] += f"Extra features: {list(extra_features)}\n"
            
            # Ensure feature order matches model
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
        """Get feature names used during model training"""
        try:
            # Try to get feature names from model attributes
            if hasattr(self.model, 'feature_names_in_'):
                return list(self.model.feature_names_in_)
            elif hasattr(self.model, 'get_booster'):
                # XGBoost model
                return self.model.get_booster().feature_names
            elif hasattr(self.model, 'feature_importances_'):
                # If feature importances exist but no names, create default names
                n_features = len(self.model.feature_importances_)
                return [f"feature_{i}" for i in range(n_features)]
            else:
                return None
        except:
            return None
    
    def preprocess_data(self, X):
        """Preprocess data"""
        X_processed = X.copy()
        preprocessing_info = {"removed_columns": [], "encoded_columns": []}
        
        # Process each column's data type
        for col in list(X_processed.columns):
            if col not in X_processed.columns:
                continue
                
            col_dtype = X_processed[col].dtype
            
            # Smarter ID column detection - only remove explicit ID columns, not features like iD1
            if self.remove_id.get():
                # Only remove explicit ID columns, not columns that look like feature names
                is_real_id = (col.lower() in ['id', 'idno', 'id_num', 'id_number', 'patient_id', 'sample_id'] or
                            col.lower().endswith('_id') or
                            col.lower().startswith('id_'))
                
                # Specifically preserve features like iD1, iD2, iD3a, iD3b, iD4a, iD4b
                is_important_feature = (col in ['iD1', 'iD2', 'iD3a', 'iD3b', 'iD4a', 'iD4b'] or
                                    col.startswith('iD') and len(col) > 2 and col[2:].isdigit())
                
                if is_real_id and not is_important_feature:
                    X_processed = X_processed.drop(columns=[col])
                    preprocessing_info["removed_columns"].append(col)
                    continue
            
            # Process datetime columns
            from pandas.api.types import is_datetime64_any_dtype

            # Process datetime columns
            if self.handle_datetime.get() and is_datetime64_any_dtype(X_processed[col]):
                X_processed = X_processed.drop(columns=[col])
                preprocessing_info["removed_columns"].append(col)
                continue

            
            # Process categorical variables
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
            
            # Preprocess data
            X_processed, preprocessing_info = self.preprocess_data(X)
            
            # Ensure features match model
            X_aligned, alignment_info = self.align_features_with_model(X_processed, self.model_feature_names)
            if not alignment_info["aligned"]:
                raise ValueError("Feature alignment failed")
            
            # Display detailed matching information
            if alignment_info["missing_features"] or alignment_info["extra_features"]:
                detailed_message = "Feature Alignment Details:\n\n"
                detailed_message += alignment_info["details"]
                detailed_message += f"\nCurrent features: {len(alignment_info['current_features'])}"
                detailed_message += f"\nModel features: {len(alignment_info['model_features'])}"
                detailed_message += f"\nFinal features: {len(X_aligned.columns)}"
                
                self.root.after(0, lambda: messagebox.showinfo("Feature Alignment Details", detailed_message))
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X_aligned, y, test_size=0.2, random_state=42)
            self.X_test = X_test
            self.y_test = y_test
            
            # Make predictions
            self.y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            # Unify class labels (based on full y)
            classes = np.unique(y)

            accuracy = accuracy_score(y_test, self.y_pred)
            cm = confusion_matrix(y_test, self.y_pred, labels=classes)
            report = classification_report(y_test, self.y_pred, labels=classes, zero_division=0)

            self.root.after(0, self.update_accuracy_tab, accuracy, preprocessing_info)
            self.root.after(0, self.update_cm_tab, cm, classes)  # Pass the same classes
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
            # Create feature importance data
            if isinstance(self.feature_importances, dict):
                # XGBoost format feature importances
                fi_data = []
                for feature, importance in self.feature_importances.items():
                    fi_data.append((feature, importance))
                fi_data.sort(key=lambda x: x[1], reverse=True)
            else:
                # Array format feature importances
                fi_data = list(zip(feature_names, self.feature_importances))
                fi_data.sort(key=lambda x: x[1], reverse=True)
            
            # Create feature importance chart
            fig = Figure(figsize=(14, 8))              # New
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
            
            # Add detailed feature importance table
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
        
        # Display accuracy
        ttk.Label(self.accuracy_frame, text=f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", 
                 font=("Arial", 16)).pack(pady=10)
        
        # Display preprocessing information
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
        fig = Figure(figsize=(8, 6))                # New
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
        
        scroll = ScrollableFrame(self.shap_frame)
        scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        parent_for_content = scroll.body

        try:
            # Use numeric cleaning method from your XGBoost code
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

            # Explainer
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

            # ---- Unify into list[n_classes], each element has shape (n_samples, n_features) ----
            def _split_sv_to_list(sv_val, n_classes_hint=None):
                # sv could be list, (n_samples,n_features), (n_samples,n_classes,n_features)
                # or (n_samples,n_features,n_classes)
                if isinstance(sv_val, list):
                    return sv_val

                arr = np.asarray(sv_val)
                if arr.ndim == 2:
                    # Binary classification/regression: one output
                    return [arr]

                if arr.ndim == 3:
                    # Guess which dimension contains classes: prefer model.classes_ or unique class count from y_test
                    if n_classes_hint is None:
                        n_classes_hint = len(getattr(self.model, "classes_", [])) or len(np.unique(self.y_test))

                    # First try to match any dimension == n_classes_hint
                    for axis, size in enumerate(arr.shape):
                        if n_classes_hint and size == n_classes_hint:
                            return [np.take(arr, i, axis=axis) for i in range(size)]

                    # Second option: if last dimension is small (<=10), treat as class dimension
                    if arr.shape[-1] <= 10:
                        return [arr[..., i] for i in range(arr.shape[-1])]
                    # Third option: if middle dimension is small (<=10), treat as class dimension
                    if arr.shape[1] <= 10:
                        return [arr[:, i, :] for i in range(arr.shape[1])]
                    # Fallback: split by first dimension (legacy return from most tree models)
                    return [arr[i] for i in range(arr.shape[0])]

                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

            # Split based on hint
            n_classes_hint = len(getattr(self.model, "classes_", [])) or len(np.unique(self.y_test))
            sv_list = _split_sv_to_list(sv, n_classes_hint=n_classes_hint)

            # ---- Align class labels: use sv_list length to avoid out-of-bounds ----
            model_classes = getattr(self.model, "classes_", None)
            if isinstance(model_classes, (list, np.ndarray)) and len(model_classes) == len(sv_list):
                class_ids = list(model_classes)
            else:
                # Align with classes present in test set and sv_list, otherwise fallback to 0..n-1
                test_classes = list(np.unique(self.y_test))
                if len(test_classes) == len(sv_list):
                    class_ids = test_classes
                else:
                    class_ids = list(range(len(sv_list)))
            
            # Create main tabs
            notebook = ttk.Notebook(parent_for_content)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Calculate mean absolute SHAP values (feature importance) for each class
            shap_importance_by_class = {}
            for ci, sv_c in enumerate(sv_list):
                # Mean absolute SHAP value as feature importance
                mean_abs_shap = np.mean(np.abs(sv_c), axis=0)
                shap_importance_by_class[class_ids[ci]] = mean_abs_shap

            # Create tabs for each class
            for ci, sv_c in enumerate(sv_list):
                class_frame = ttk.Frame(notebook)
                notebook.add(class_frame, text=f"Class {class_ids[ci]}")

                # Create sub-tabs (Beeswarm and Radar chart)
                class_notebook = ttk.Notebook(class_frame)
                class_notebook.pack(fill=tk.BOTH, expand=True)

                # Beeswarm plot tab
                beeswarm_frame = ttk.Frame(class_notebook)
                class_notebook.add(beeswarm_frame, text="Beeswarm Plot")

                mask_pos = (np.array(self.y_test) == class_ids[ci])
                if mask_pos.sum() >= 5:
                    X_num_disp = self._with_readable_columns(X_num)

                    # —— Important: Create a new pyplot figure specifically for SHAP —— #
                    fig_beeswarm = plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        sv_c[mask_pos], X_num_disp.iloc[mask_pos],
                        plot_type="dot", max_display=12, show=False
                    )
                    fig_beeswarm.tight_layout()

                    canvas_beeswarm = FigureCanvasTkAgg(fig_beeswarm, master=beeswarm_frame)
                    canvas_beeswarm.draw()
                    canvas_beeswarm.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                    # Save reference and unregister from pyplot to avoid affecting subsequent plots
                    self._mpl_refs.append(canvas_beeswarm)
                    plt.close(fig_beeswarm)
                else:
                    ttk.Label(
                        beeswarm_frame,
                        text=f"Not enough samples ({mask_pos.sum()}) for beeswarm plot"
                    ).pack(pady=20)

                # Radar chart tab
                radar_frame = ttk.Frame(class_notebook)
                class_notebook.add(radar_frame, text="Radar Chart")

                # Create radar chart
                self._create_shap_radar_chart(radar_frame, shap_importance_by_class[class_ids[ci]], 
                                            X_num.columns, class_ids[ci])

            # # Add comparison radar chart tab for all classes
            # comparison_frame = ttk.Frame(notebook)
            # notebook.add(comparison_frame, text="Class Comparison")

            # Create comparison radar chart for all classes
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
        """Return a copy with readable column names for plotting"""
        # If no feature name mapping exists, use original column names
        if not hasattr(self, 'feature_name_map') or not self.feature_name_map:
            return X.copy()
        
        # Create readable column names
        readable_columns = []
        for col in X.columns:
            base_col = re.sub(r"(_scaled|_std|_lag\d+|_zscore)$", "", str(col))
            readable_name = self.feature_name_map.get(base_col, col)
            readable_columns.append(readable_name)
        
        X_disp = X.copy()
        X_disp.columns = readable_columns
        return X_disp

    def _compose_display_labels(self, names, codes):
        """Combine names and codes into unique labels: Name (Code)"""
        labels = []
        for n, c in zip(names, codes):
            if str(n) == str(c):
                labels.append(str(n))
            else:
                labels.append(f"{n} ({c})")
        return labels

    def _map_feature_names(self, cols):
        """Map feature codes to readable names"""
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
    
    def _create_shap_radar_chart(self, parent, shap_importance, feature_names, class_id, top_n=12):
        """Create SHAP importance radar chart for a single class (absolute scale, no normalization)"""
        try:
            # 1) Get Top-N features for this class (sorted by mean|SHAP|)
            top_n = min(top_n, len(feature_names))
            top_indices = np.argsort(shap_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = shap_importance[top_indices]  # Real mean|SHAP| values

            # 2) Polar angles (closed polygon)
            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]  # Close

            # 3) Prepare values and close the polygon
            values = np.concatenate([top_importance, [top_importance[0]]])

            # 4) Plot (no normalization, using absolute scale)
            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values, color='red', linewidth=2)
            ax.fill(angles, values, color='red', alpha=0.25)

            # 5) Axes and annotations
            feature_labels = self._map_feature_names(top_features)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=9)

            ymax = float(top_importance.max()) if top_importance.size else 1.0
            if ymax <= 0:
                ymax = 1.0
            ax.set_ylim(0, ymax * 1.05)
            yticks = np.linspace(0, ymax, 5)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.3g}" for v in yticks], fontsize=8)  # Display real SHAP values

            ax.set_title(f"Top {top_n} Features - Class {class_id}\n(by absolute mean |SHAP|)",
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)

            fig.tight_layout()
            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)

            # 6) Table below: show real mean|SHAP| values for Top-N features
            table_frame = ttk.Frame(parent)
            table_frame.pack(fill=tk.X, padx=10, pady=5)

            table_data = []
            for i, (feature, importance) in enumerate(zip(top_features, top_importance), start=1):
                display_name = self._map_feature_names([feature])[0]
                table_data.append((i, display_name, f"{importance:.6f}"))

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


    def _create_comparison_radar_chart(self, parent, shap_importance_dict, feature_names, class_ids, top_n=12):
        """Compare mean|SHAP| across classes: use unified absolute scale (no normalization) to ensure comparable magnitudes"""
        try:
            # 1) Select feature indices for comparison: prefer model FI, otherwise use cross-class mean|SHAP|
            if self.feature_importances is not None:
                if isinstance(self.feature_importances, dict):
                    fi_vec = np.array([self.feature_importances.get(str(f), 0.0) for f in feature_names], dtype=float)
                else:
                    fi_vec = np.asarray(self.feature_importances, dtype=float)
                    if fi_vec.shape[0] != len(feature_names):
                        fi_vec = np.resize(fi_vec, len(feature_names))
                top_indices = np.argsort(fi_vec)[-top_n:][::-1]
            else:
                all_importances_for_pick = np.stack(list(shap_importance_dict.values()))  # (n_classes, n_features)
                mean_importance = np.mean(all_importances_for_pick, axis=0)
                top_indices = np.argsort(mean_importance)[-top_n:][::-1]

            top_features = [feature_names[i] for i in top_indices]
            N = len(top_features)

            # 2) Polar angles (closed)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]

            # 3) Unified absolute scale: find global maximum across all "classes × selected Top-N features"
            all_importances = np.stack([shap_importance_dict[c] for c in class_ids])  # (n_classes, n_features)
            global_max = float(np.max(all_importances[:, top_indices])) if top_indices is not None else float(np.max(all_importances))
            if global_max <= 0:
                global_max = 1.0  # Prevent division by zero/empty plot when all zeros

            # 4) Plot (no normalization at all!)
            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111, polar=True)

            colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
            for i, class_id in enumerate(class_ids):
                importance = shap_importance_dict[class_id][top_indices]      # Real mean|SHAP| (absolute values)
                values = np.concatenate([importance, [importance[0]]])        # Close
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Class {class_id}', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.10)

            # 5) Axes and annotations (same absolute scale)
            feature_labels = self._map_feature_names(top_features)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, fontsize=9)

            # Set y-axis upper limit with global maximum to ensure comparability across classes
            ax.set_ylim(0, global_max * 1.05)
            yticks = np.linspace(0, global_max, 5)  # 0, 25%, 50%, 75%, 100% of "absolute value" scale
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.3g}" for v in yticks], fontsize=8)  # Display real SHAP values directly (not percentages)

            ax.set_title(f"Top {top_n} Features Comparison Across Classes\n(Absolute mean |SHAP|, same scale)", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.grid(True)

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