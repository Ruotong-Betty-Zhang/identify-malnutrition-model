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

# ------------------------------
# 可滚动容器（沿用你的实现）
# ------------------------------
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

        for w in (self, self.canvas, self.inner):
            w.bind("<Enter>", self._bind_mousewheel)
            w.bind("<Leave>", self._unbind_mousewheel)

    @property
    def body(self):
        return self.inner

    def _on_mousewheel(self, event):
        step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")

    def _bind_mousewheel(self, _=None):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux)

    def _unbind_mousewheel(self, _=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")


# ------------------------------
# 结果面板：渲染单个模型的所有结果
# ------------------------------
class ResultPanel:
    def __init__(self, parent):
        self._mpl_refs = []
        self.parent = parent
        self.state = {
            'X_test': None,
            'y_test': None,
            'y_pred': None,
            'accuracy': None,
            'cm': None,
            'classes': None,
            'report': None,
            'feature_importances': None,
            'feature_names': None,
            'alignment_info': None,
            'feature_name_map': None,
            'model': None,
        }
        self._build_ui()

    def _build_ui(self):
        self.nb = ttk.Notebook(self.parent)
        self.nb.pack(fill=tk.BOTH, expand=True)

        self.tab_accuracy = ttk.Frame(self.nb, padding=10)
        self.tab_cm = ttk.Frame(self.nb, padding=10)
        self.tab_report = ttk.Frame(self.nb, padding=10)
        self.tab_fi = ttk.Frame(self.nb, padding=10)
        self.tab_shap = ttk.Frame(self.nb, padding=10)
        self.tab_preview = ttk.Frame(self.nb, padding=10)
        self.tab_align = ttk.Frame(self.nb, padding=10)

        self.nb.add(self.tab_accuracy, text="Accuracy")
        self.nb.add(self.tab_cm, text="Confusion Matrix")
        self.nb.add(self.tab_report, text="Classification Report")
        self.nb.add(self.tab_fi, text="Feature Importance")
        self.nb.add(self.tab_shap, text="SHAP Analysis")
        self.nb.add(self.tab_align, text="Feature Alignment")

    # ---------- 渲染函数 ----------
    def clear_tab(self, tab):
        for w in tab.winfo_children():
            w.destroy()

    def render_accuracy(self, accuracy, preprocessing_info):
        self.clear_tab(self.tab_accuracy)
        ttk.Label(self.tab_accuracy, text=f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)", font=("Arial", 16)).pack(pady=10)
        if preprocessing_info.get('removed_columns'):
            ttk.Label(self.tab_accuracy, text=f"Removed columns: {', '.join(preprocessing_info['removed_columns'])}").pack(pady=5)
        if preprocessing_info.get('encoded_columns'):
            ttk.Label(self.tab_accuracy, text=f"Encoded columns: {', '.join(preprocessing_info['encoded_columns'])}").pack(pady=5)

    def render_cm(self, cm, classes):
        self.clear_tab(self.tab_cm)
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix')
        canvas = FigureCanvasTkAgg(fig, master=self.tab_cm)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._mpl_refs.append(canvas)

    def render_report(self, report_text):
        self.clear_tab(self.tab_report)
        text = scrolledtext.ScrolledText(self.tab_report, wrap=tk.WORD)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, "Classification Report:\n\n")
        text.insert(tk.END, report_text)
        text.config(state=tk.DISABLED)

    def render_fi(self, feature_importances, feature_names):
        self.clear_tab(self.tab_fi)
        if feature_importances is None:
            ttk.Label(self.tab_fi, text="Feature importance not available for this model").pack(pady=12)
            return
        if isinstance(feature_importances, dict):
            fi_data = sorted(list(feature_importances.items()), key=lambda x: x[1], reverse=True)
        else:
            fi_data = list(zip(feature_names, feature_importances))
            fi_data.sort(key=lambda x: x[1], reverse=True)
        fig = Figure(figsize=(14, 8))
        ax = fig.add_subplot(111)
        top_k = min(12, len(fi_data))
        feats = [x[0] for x in fi_data[:top_k]]
        vals = [x[1] for x in fi_data[:top_k]]
        y = np.arange(len(feats))
        ax.barh(y, vals, align='center')
        ax.set_yticks(y); ax.set_yticklabels(feats); ax.invert_yaxis()
        ax.set_xlabel('Feature Importance'); ax.set_title('Top 12 Feature Importances')
        canvas = FigureCanvasTkAgg(fig, master=self.tab_fi)
        canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self._mpl_refs.append(canvas)

        frame = ttk.Frame(self.tab_fi); frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        tree = ttk.Treeview(frame, columns=('Feature','Importance'), show='headings', height=10)
        tree.heading('Feature', text='Feature'); tree.heading('Importance', text='Importance')
        tree.column('Feature', width=300); tree.column('Importance', width=120)
        for f,v in fi_data:
            if isinstance(v, (int,float,np.floating)):
                tree.insert('', 'end', values=(f, f"{v:.6f}"))
            else:
                tree.insert('', 'end', values=(f, str(v)))
        sb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)

    def render_align(self, alignment_info, X_cols=None):
        self.clear_tab(self.tab_align)
        text_widget = scrolledtext.ScrolledText(self.tab_align, wrap=tk.WORD, height=15)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        info_text = "=== FEATURE ALIGNMENT DETAILS ===\n\n"
        info_text += alignment_info.get("details", "") + "\n"
        info_text += f"Missing: {len(alignment_info.get('missing_features', []))}\n"
        info_text += f"Extra: {len(alignment_info.get('extra_features', []))}\n\n"
        if alignment_info.get('missing_features'):
            info_text += "MISSING (added with 0):\n" + "\n".join([f"  - {x}" for x in alignment_info['missing_features']]) + "\n\n"
        if alignment_info.get('extra_features'):
            info_text += "EXTRA (removed):\n" + "\n".join([f"  - {x}" for x in alignment_info['extra_features']]) + "\n\n"
        if X_cols is not None:
            info_text += "FINAL FEATURES ORDER:\n" + "\n".join([f"  {i+1}. {c}" for i,c in enumerate(X_cols)])
        text_widget.insert(tk.END, info_text)
        text_widget.config(state=tk.DISABLED)

    def render_shap(self, model, X_test, y_test):
        self.clear_tab(self.tab_shap)
        scroll = ScrollableFrame(self.tab_shap); scroll.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        parent = scroll.body
        try:
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

            try:
                explainer = shap.TreeExplainer(model, model_output="raw")
                sv = explainer.shap_values(X_num)
            except Exception:
                explainer = shap.LinearExplainer(model, X_num, feature_dependence="independent")
                sv = explainer.shap_values(X_num)

            def _split_sv_to_list(sv_val, n_classes_hint=None):
                if isinstance(sv_val, list):
                    return sv_val
                arr = np.asarray(sv_val)
                if arr.ndim == 2:
                    return [arr]
                if arr.ndim == 3:
                    if n_classes_hint is None:
                        n_classes_hint = len(getattr(model, "classes_", [])) or len(np.unique(y_test))
                    for axis, size in enumerate(arr.shape):
                        if n_classes_hint and size == n_classes_hint:
                            return [np.take(arr, i, axis=axis) for i in range(size)]
                    if arr.shape[-1] <= 10:
                        return [arr[..., i] for i in range(arr.shape[-1])]
                    if arr.shape[1] <= 10:
                        return [arr[:, i, :] for i in range(arr.shape[1])]
                    return [arr[i] for i in range(arr.shape[0])]
                raise ValueError(f"Unexpected SHAP shape: {arr.shape}")

            n_classes_hint = len(getattr(model, "classes_", [])) or len(np.unique(y_test))
            sv_list = _split_sv_to_list(sv, n_classes_hint=n_classes_hint)

            model_classes = getattr(model, "classes_", None)
            if isinstance(model_classes, (list, np.ndarray)) and len(model_classes) == len(sv_list):
                class_ids = list(model_classes)
            else:
                test_classes = list(np.unique(y_test))
                if len(test_classes) == len(sv_list):
                    class_ids = test_classes
                else:
                    class_ids = list(range(len(sv_list)))

            notebook = ttk.Notebook(parent); notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            shap_importance_by_class = {}
            for ci, sv_c in enumerate(sv_list):
                mean_abs_shap = np.mean(np.abs(sv_c), axis=0)
                shap_importance_by_class[class_ids[ci]] = mean_abs_shap

            for ci, sv_c in enumerate(sv_list):
                class_frame = ttk.Frame(notebook); notebook.add(class_frame, text=f"Class {class_ids[ci]}")
                class_nb = ttk.Notebook(class_frame); class_nb.pack(fill=tk.BOTH, expand=True)

                beeswarm_frame = ttk.Frame(class_nb); class_nb.add(beeswarm_frame, text="Beeswarm Plot")
                mask_pos = (np.array(y_test) == class_ids[ci])
                if mask_pos.sum() >= 5:
                    fig_beeswarm = plt.figure(figsize=(12, 8))
                    shap.summary_plot(sv_c[mask_pos], X_num.iloc[mask_pos], plot_type="dot", max_display=12, show=False)
                    fig_beeswarm.tight_layout()
                    canvas = FigureCanvasTkAgg(fig_beeswarm, master=beeswarm_frame)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    self._mpl_refs.append(canvas); plt.close(fig_beeswarm)
                else:
                    ttk.Label(beeswarm_frame, text=f"Not enough samples ({mask_pos.sum()}) for beeswarm plot").pack(pady=20)

                radar_frame = ttk.Frame(class_nb); class_nb.add(radar_frame, text="Radar Chart")
                self._create_shap_radar_chart(radar_frame, shap_importance_by_class[class_ids[ci]], X_num.columns, class_ids[ci])

            comp_frame = ttk.Frame(notebook); notebook.add(comp_frame, text="Class Comparison")
            available_classes = list(shap_importance_by_class.keys())
            self._create_comparison_radar_chart(comp_frame, shap_importance_by_class, X_num.columns, available_classes)

        except Exception as e:
            ttk.Label(self.tab_shap, text=f"SHAP analysis error: {str(e)}").pack(pady=20)

    def _map_feature_names(self, cols):
        return list(cols)

    def _create_shap_radar_chart(self, parent, shap_importance, feature_names, class_id, top_n=12):
        try:
            top_n = min(top_n, len(feature_names))
            top_indices = np.argsort(shap_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = shap_importance[top_indices]
            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
            values = np.concatenate([top_importance, [top_importance[0]]])

            fig = Figure(figsize=(10, 8)); ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.25)
            feature_labels = self._map_feature_names(top_features)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(feature_labels, fontsize=9)
            ymax = float(top_importance.max()) if top_importance.size else 1.0
            if ymax <= 0: ymax = 1.0
            ax.set_ylim(0, ymax * 1.05)
            yticks = np.linspace(0, ymax, 5)
            ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.3g}" for v in yticks], fontsize=8)
            ax.set_title(f"Top {top_n} Features - Class {class_id}", fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)
            canvas = FigureCanvasTkAgg(fig, master=parent); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)

            # 表格
            table_frame = ttk.Frame(parent); table_frame.pack(fill=tk.X, padx=10, pady=5)
            tree = ttk.Treeview(table_frame, columns=('Rank','Feature','Mean|SHAP|'), show='headings', height=min(6, len(top_features)))
            tree.heading('Rank', text='Rank'); tree.heading('Feature', text='Feature'); tree.heading('Mean|SHAP|', text='Mean |SHAP|')
            tree.column('Rank', width=50, anchor='center'); tree.column('Feature', width=300); tree.column('Mean|SHAP|', width=120, anchor='center')
            for i,(f,imp) in enumerate(zip(top_features, top_importance), start=1):
                tree.insert('', 'end', values=(i, f, f"{imp:.6f}"))
            sb = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)
        except Exception as e:
            ttk.Label(parent, text=f"Radar chart error: {str(e)}").pack(pady=20)

    def _create_comparison_radar_chart(self, parent, shap_importance_dict, feature_names, class_ids, top_n=12):
        try:
            all_importances = np.stack([shap_importance_dict[c] for c in class_ids])
            mean_importance = np.mean(all_importances, axis=0)
            top_indices = np.argsort(mean_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]
            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
            global_max = float(np.max(all_importances[:, top_indices]))
            if global_max <= 0: global_max = 1.0
            fig = Figure(figsize=(12, 9)); ax = fig.add_subplot(111, polar=True)
            colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
            for i, cid in enumerate(class_ids):
                importance = shap_importance_dict[cid][top_indices]
                values = np.concatenate([importance, [importance[0]]])
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Class {cid}', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.10)
            feature_labels = top_features
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(feature_labels, fontsize=9)
            ax.set_ylim(0, global_max * 1.05)
            yticks = np.linspace(0, global_max, 5)
            ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.3g}" for v in yticks], fontsize=8)
            ax.set_title(f"Top {top_n} Features Comparison Across Classes", fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.0)); ax.grid(True)
            canvas = FigureCanvasTkAgg(fig, master=parent); canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)
        except Exception as e:
            ttk.Label(parent, text=f"Comparison radar chart error: {str(e)}").pack(pady=20)


# ------------------------------
# 主应用：支持 A / B 两套数据+模型，并提供对比页
# ------------------------------
class ModelComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Evaluation & Comparison Tool")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # 共享预处理选项
        self.handle_categorical = tk.BooleanVar(value=True)
        self.handle_datetime = tk.BooleanVar(value=True)
        self.remove_id = tk.BooleanVar(value=True)

        # A / B 路径与格式
        self.dataset_path = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.model_path = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.data_format = {'A': tk.StringVar(value='pkl'), 'B': tk.StringVar(value='pkl')}
        self.target_var = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.dataset_df = {'A': None, 'B': None}
        self.model = {'A': None, 'B': None}
        self.model_feature_names = {'A': None, 'B': None}
        self.feature_importances = {'A': None, 'B': None}

        # 评估结果缓存（便于对比页复用）
        self.results = {
            'A': {},
            'B': {},
        }

        self.label_encoders = {}
        self.status_var = tk.StringVar(value='Ready')

        self._build_ui()

    # ---------- UI ----------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky='nsew')
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        # Top: A / B 两列设置区
        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self._build_side_controls(top, side='A', col=0, label='Model A & Dataset A')
        self._build_side_controls(top, side='B', col=1, label='Model B & Dataset B')

        # 预处理选项（共享）
        pre = ttk.LabelFrame(main, text='Preprocessing (applies to both A & B)')
        pre.grid(row=1, column=0, sticky='ew', pady=(8,4))
        ttk.Checkbutton(pre, text="Encode categorical", variable=self.handle_categorical).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(pre, text="Process datetime", variable=self.handle_datetime).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(pre, text="Remove ID columns", variable=self.remove_id).pack(side=tk.LEFT, padx=8)

        # 按钮区
        btns = ttk.Frame(main)
        btns.grid(row=2, column=0, sticky='ew', pady=(6,6))
        ttk.Button(btns, text='Run A', command=lambda: self.run_evaluation('A')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text='Run B', command=lambda: self.run_evaluation('B')).pack(side=tk.LEFT, padx=5)
        ttk.Button(btns, text='Run A & B and Compare', command=self.run_compare).pack(side=tk.LEFT, padx=12)

        # 结果区：上层 Notebook，包含三个页：A/B 详情 + Comparison 对比
        self.top_nb = ttk.Notebook(main)
        self.top_nb.grid(row=3, column=0, sticky='nsew', pady=(8,8))
        main.rowconfigure(3, weight=1)

        self.tab_A = ttk.Frame(self.top_nb, padding=6)
        self.tab_B = ttk.Frame(self.top_nb, padding=6)
        self.tab_cmp = ttk.Frame(self.top_nb, padding=6)
        self.top_nb.add(self.tab_A, text='Model A Details')
        self.top_nb.add(self.tab_B, text='Model B Details')
        self.top_nb.add(self.tab_cmp, text='Comparison')

        self.panelA = ResultPanel(self.tab_A)
        self.panelB = ResultPanel(self.tab_B)

        # Comparison 页内部布局
        self._build_comparison_tab()

        # 状态栏
        ttk.Label(main, textvariable=self.status_var).grid(row=4, column=0, sticky='w')

    def _build_side_controls(self, parent, side: str, col: int, label: str):
        box = ttk.LabelFrame(parent, text=label, padding=8)
        box.grid(row=0, column=col, sticky='nsew', padx=6)
        parent.columnconfigure(col, weight=1)

        # 数据集
        ttk.Label(box, text='Dataset Path:').grid(row=0, column=0, sticky='w', pady=3)
        ttk.Entry(box, textvariable=self.dataset_path[side], width=48).grid(row=0, column=1, sticky='ew', padx=6)
        ttk.Button(box, text='Browse', command=lambda s=side: self.browse_dataset(s)).grid(row=0, column=2, padx=4)

        ttk.Label(box, text='Format:').grid(row=1, column=0, sticky='w', pady=3)
        fmt = ttk.Frame(box); fmt.grid(row=1, column=1, sticky='w')
        ttk.Radiobutton(fmt, text='PKL', variable=self.data_format[side], value='pkl').pack(side=tk.LEFT)
        ttk.Radiobutton(fmt, text='CSV', variable=self.data_format[side], value='csv').pack(side=tk.LEFT)

        ttk.Label(box, text='Target Column:').grid(row=2, column=0, sticky='w', pady=3)
        combo = ttk.Combobox(box, textvariable=self.target_var[side], state='readonly', width=20)
        combo.grid(row=2, column=1, sticky='w', padx=6)
        setattr(self, f'target_combo_{side}', combo)

        # 模型
        ttk.Label(box, text='Model Path:').grid(row=3, column=0, sticky='w', pady=3)
        ttk.Entry(box, textvariable=self.model_path[side], width=48).grid(row=3, column=1, sticky='ew', padx=6)
        ttk.Button(box, text='Browse', command=lambda s=side: self.browse_model(s)).grid(row=3, column=2, padx=4)

        for c in (0,1,2):
            box.columnconfigure(c, weight=1)

    def _build_comparison_tab(self):
        """Comparison 页面：使用内部 Notebook 分页展示不同对比项目"""
        # 创建分页
        self.cmp_nb = ttk.Notebook(self.tab_cmp)
        self.cmp_nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 1) Metrics 汇总
        self.cmp_tab_metrics = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_metrics, text='Metrics')
        self.cmp_metrics_tree = ttk.Treeview(
            self.cmp_tab_metrics,
            columns=('Metric', 'Model A', 'Model B', 'Δ (B-A)'),
            show='headings', height=8
        )
        for i, (col, w) in enumerate([
            ('Metric', 220), ('Model A', 140), ('Model B', 140), ('Δ (B-A)', 140)
        ]):
            self.cmp_metrics_tree.heading(col, text=col)
            self.cmp_metrics_tree.column(col, width=w, anchor='center' if i else 'w')
        self.cmp_metrics_tree.pack(fill=tk.X, expand=True)

        # 2) Confusion Matrices（并排）
        self.cmp_tab_cm = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_cm, text='Confusion Matrices')
        self.cmp_cm_left = ttk.Frame(self.cmp_tab_cm)
        self.cmp_cm_right = ttk.Frame(self.cmp_tab_cm)
        self.cmp_cm_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_cm_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 3) Feature Importance（并排）
        self.cmp_tab_fi = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_fi, text='Feature Importance')
        self.cmp_fi_left = ttk.Frame(self.cmp_tab_fi)
        self.cmp_fi_right = ttk.Frame(self.cmp_tab_fi)
        self.cmp_fi_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_fi_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 4) Classification Reports（并排文本）
        self.cmp_tab_reports = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_reports, text='Reports')
        self.cmp_rep_left = ttk.Frame(self.cmp_tab_reports)
        self.cmp_rep_right = ttk.Frame(self.cmp_tab_reports)
        self.cmp_rep_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_rep_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 5) SHAP Top-N Comparison（雷达或提示）
        self.cmp_tab_shap = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_shap, text='SHAP (Top-N)')
        self.cmp_shap_container = ttk.Frame(self.cmp_tab_shap)
        self.cmp_shap_container.pack(fill=tk.BOTH, expand=True)


    # ---------- 数据/模型加载 ----------
    def browse_dataset(self, side: str):
        file_types = [("PKL files", "*.pkl"), ("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if not file_path: return
        self.dataset_path[side].set(file_path)
        try:
            if self.data_format[side].get() == 'pkl':
                df = joblib.load(file_path)
            else:
                df = pd.read_csv(file_path)
            if not isinstance(df, pd.DataFrame):
                raise ValueError('Unsupported data format (expect DataFrame).')
            self.dataset_df[side] = df
            combo: ttk.Combobox = getattr(self, f'target_combo_{side}')
            combo['values'] = list(df.columns)
            if df.columns.size > 0:
                self.target_var[side].set(df.columns[-1])
            self.status_var.set(f"Dataset {side} loaded: {df.shape}")
        except Exception as e:
            messagebox.showerror('Error', f'Error loading dataset {side}: {e}')

    def browse_model(self, side: str):
        file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.pkl;*.joblib"), ("All files", "*.*")])
        if not file_path: return
        self.model_path[side].set(file_path)
        try:
            model = joblib.load(file_path)
            self.model[side] = model
            self.model_feature_names[side] = self._get_model_feature_names(model)
            self.feature_importances[side] = self._get_feature_importances(model)
            self.status_var.set(f"Model {side} loaded")
        except Exception as e:
            messagebox.showerror('Error', f'Error loading model {side}: {e}')

    # ---------- 运行评估 & 对比 ----------
    def run_evaluation(self, side: str):
        t = threading.Thread(target=lambda: self._run_eval_thread(side), daemon=True)
        t.start()

    def run_compare(self):
        def _go():
            self._run_eval_thread('A')
            self._run_eval_thread('B')
            self.root.after(0, self._render_comparison)
        t = threading.Thread(target=_go, daemon=True)
        t.start()

    def _run_eval_thread(self, side: str):
        try:
            self.status_var.set(f"Running evaluation for {side}...")
            df = self.dataset_df[side]
            model = self.model[side]
            target = self.target_var[side].get()
            if df is None or model is None:
                messagebox.showerror('Error', f'Please load both dataset and model for {side}.'); return
            if not target:
                messagebox.showerror('Error', f'Please select target column for {side}.'); return

            X = df.drop(columns=[target]); y = df[target]
            X_proc, prep_info = self._preprocess_data(X)
            X_align, align_info = self._align_features_with_model(X_proc, self.model_feature_names[side])
            if not align_info.get('aligned', False):
                raise ValueError('Feature alignment failed')

            classes = np.unique(y)
            X_train, X_test, y_train, y_test = train_test_split(X_align, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            report = classification_report(y_test, y_pred, labels=classes, zero_division=0)

            # 保存结果
            self.results[side] = {
                'X_test': X_test,
                'y_test': y_test,
                'y_pred': y_pred,
                'accuracy': acc,
                'cm': cm,
                'classes': classes,
                'report': report,
                'feature_importances': self.feature_importances[side],
                'feature_names': list(X_align.columns),
                'alignment_info': align_info,
                'model': model,
                'preprocess_info': prep_info,
            }

            # 渲染到对应面板
            panel = self.panelA if side=='A' else self.panelB
            self.root.after(0, lambda: panel.render_accuracy(acc, prep_info))
            self.root.after(0, lambda: panel.render_cm(cm, classes))
            self.root.after(0, lambda: panel.render_report(report))
            self.root.after(0, lambda: panel.render_fi(self.feature_importances[side], list(X_align.columns)))
            self.root.after(0, lambda: panel.render_align(align_info, X_cols=list(X_test.columns)))
            self.root.after(0, lambda: panel.render_shap(model, X_test, y_test))

            self.status_var.set(f"Evaluation {side} completed")
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror('Error', f'Error during evaluation {side}: {e}'))
            self.status_var.set(f"Evaluation {side} failed")

    def _render_comparison(self):
        # --- 读取缓存 ---
        a = self.results.get('A', {})
        b = self.results.get('B', {})
        if not a or not b:
            return

        # ========== 1) Metrics 页 ==========
        for i in self.cmp_metrics_tree.get_children():
            self.cmp_metrics_tree.delete(i)

        def _fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else str(x)

        rows = [
            ('Accuracy', a.get('accuracy', np.nan), b.get('accuracy', np.nan)),
            # 需要的话可在此追加 ('F1 (macro)', ...), ('Precision (macro)', ...), ('Recall (macro)', ...)
        ]
        for name, va, vb in rows:
            delta = (vb - va) if isinstance(va, (int, float, np.floating)) and isinstance(vb, (int, float, np.floating)) else ''
            self.cmp_metrics_tree.insert('', 'end', values=(name, _fmt(va), _fmt(vb), _fmt(delta) if delta != '' else ''))

        # ========== 2) Confusion Matrices 页 ==========
        for w in self.cmp_cm_left.winfo_children():
            w.destroy()
        for w in self.cmp_cm_right.winfo_children():
            w.destroy()

        figA = Figure(figsize=(6, 5)); axA = figA.add_subplot(111)
        sns.heatmap(a['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=a['classes'], yticklabels=a['classes'], ax=axA)
        axA.set_title('Model A - Confusion Matrix'); axA.set_xlabel('Predicted'); axA.set_ylabel('True')
        FigureCanvasTkAgg(figA, master=self.cmp_cm_left).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        figB = Figure(figsize=(6, 5)); axB = figB.add_subplot(111)
        sns.heatmap(b['cm'], annot=True, fmt='d', cmap='Greens',
                    xticklabels=b['classes'], yticklabels=b['classes'], ax=axB)
        axB.set_title('Model B - Confusion Matrix'); axB.set_xlabel('Predicted'); axB.set_ylabel('True')
        FigureCanvasTkAgg(figB, master=self.cmp_cm_right).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ========== 3) Feature Importance 页（Top-K 并排） ==========
        for w in self.cmp_fi_left.winfo_children():
            w.destroy()
        for w in self.cmp_fi_right.winfo_children():
            w.destroy()

        top_k = 12
        # A
        if a.get('feature_importances') is not None:
            if isinstance(a['feature_importances'], dict):
                fiA = sorted(a['feature_importances'].items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                fiA = list(zip(a['feature_names'], a['feature_importances']))
                fiA.sort(key=lambda x: x[1], reverse=True); fiA = fiA[:top_k]
            fig_fiA = Figure(figsize=(6, 5)); ax_fiA = fig_fiA.add_subplot(111)
            yA = np.arange(len(fiA))
            ax_fiA.barh(yA, [v for _, v in fiA])
            ax_fiA.set_yticks(yA); ax_fiA.set_yticklabels([f for f, _ in fiA]); ax_fiA.invert_yaxis()
            ax_fiA.set_title('Model A - Top Feature Importances'); ax_fiA.set_xlabel('Importance')
            FigureCanvasTkAgg(fig_fiA, master=self.cmp_fi_left).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.cmp_fi_left, text='Model A: Feature importance not available').pack(pady=12)

        # B
        if b.get('feature_importances') is not None:
            if isinstance(b['feature_importances'], dict):
                fiB = sorted(b['feature_importances'].items(), key=lambda x: x[1], reverse=True)[:top_k]
            else:
                fiB = list(zip(b['feature_names'], b['feature_importances']))
                fiB.sort(key=lambda x: x[1], reverse=True); fiB = fiB[:top_k]
            fig_fiB = Figure(figsize=(6, 5)); ax_fiB = fig_fiB.add_subplot(111)
            yB = np.arange(len(fiB))
            ax_fiB.barh(yB, [v for _, v in fiB])
            ax_fiB.set_yticks(yB); ax_fiB.set_yticklabels([f for f, _ in fiB]); ax_fiB.invert_yaxis()
            ax_fiB.set_title('Model B - Top Feature Importances'); ax_fiB.set_xlabel('Importance')
            FigureCanvasTkAgg(fig_fiB, master=self.cmp_fi_right).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.cmp_fi_right, text='Model B: Feature importance not available').pack(pady=12)

        # ========== 4) Reports 页（并排文本） ==========
        for w in self.cmp_rep_left.winfo_children():
            w.destroy()
        for w in self.cmp_rep_right.winfo_children():
            w.destroy()

        txtA = scrolledtext.ScrolledText(self.cmp_rep_left, wrap=tk.WORD)
        txtA.pack(fill=tk.BOTH, expand=True)
        txtA.insert(tk.END, 'Model A - Classification Report\n\n')
        txtA.insert(tk.END, a.get('report', 'N/A'))
        txtA.config(state=tk.DISABLED)

        txtB = scrolledtext.ScrolledText(self.cmp_rep_right, wrap=tk.WORD)
        txtB.pack(fill=tk.BOTH, expand=True)
        txtB.insert(tk.END, 'Model B - Classification Report\n\n')
        txtB.insert(tk.END, b.get('report', 'N/A'))
        txtB.config(state=tk.DISABLED)

        # ========== 5) SHAP Top-N Comparison 页 ==========
        for w in self.cmp_shap_container.winfo_children():
            w.destroy()
        # 简洁处理：提示去各自 SHAP 页查看细节；如需，我可以把“跨模型同特征 Top-N 雷达叠加”也放到这里
        ttk.Label(
            self.cmp_shap_container,
            text=("Use each model's SHAP tabs for detailed per-class plots.\n"
                "This page summarizes top features via each model's FI as a quick proxy."),
            justify=tk.LEFT
        ).pack(pady=6)

        self.status_var.set('Comparison refreshed')

    # ---------- 工具：特征名/重要性/预处理/对齐 ----------
    def _get_feature_importances(self, model):
        try:
            if hasattr(model, 'feature_importances_'):
                return model.feature_importances_
            elif hasattr(model, 'get_booster'):
                return model.get_booster().get_score(importance_type='weight')
            elif hasattr(model, 'coef_'):
                coef = model.coef_
                return np.abs(coef) if coef.ndim==1 else np.mean(np.abs(coef), axis=0)
        except: pass
        return None

    def _get_model_feature_names(self, model):
        try:
            if hasattr(model, 'feature_names_in_'):
                return list(model.feature_names_in_)
            elif hasattr(model, 'get_booster'):
                return model.get_booster().feature_names
            elif hasattr(model, 'feature_importances_'):
                n = len(model.feature_importances_)
                return [f"feature_{i}" for i in range(n)]
        except: pass
        return None

    def _preprocess_data(self, X: pd.DataFrame):
        Xp = X.copy()
        info = {"removed_columns": [], "encoded_columns": []}
        from pandas.api.types import is_datetime64_any_dtype
        for col in list(Xp.columns):
            if col not in Xp.columns: continue
            col_dtype = Xp[col].dtype
            if self.remove_id.get():
                is_real_id = (col.lower() in ['id','idno','id_num','id_number','patient_id','sample_id'] or col.lower().endswith('_id') or col.lower().startswith('id_'))
                is_important_feature = (col in ['iD1','iD2','iD3a','iD3b','iD4a','iD4b'] or (col.startswith('iD') and len(col)>2 and col[2:].isdigit()))
                if is_real_id and not is_important_feature:
                    Xp = Xp.drop(columns=[col]); info['removed_columns'].append(col); continue
            if self.handle_datetime.get() and is_datetime64_any_dtype(Xp[col]):
                Xp = Xp.drop(columns=[col]); info['removed_columns'].append(col); continue
            if self.handle_categorical.get() and col_dtype == 'object':
                try:
                    le = LabelEncoder(); Xp[col] = le.fit_transform(Xp[col].astype(str))
                    self.label_encoders[col] = le; info['encoded_columns'].append(col)
                except:
                    Xp = Xp.drop(columns=[col]); info['removed_columns'].append(col)
        return Xp, info

    def _align_features_with_model(self, X: pd.DataFrame, model_feature_names):
        X_aligned = X.copy()
        if hasattr(X_aligned, 'columns'):
            current_features = list(X_aligned.columns)
        else:
            current_features = [f"feature_{i}" for i in range(X_aligned.shape[1])]
            if model_feature_names and len(model_feature_names) == X_aligned.shape[1]:
                X_aligned = pd.DataFrame(X_aligned, columns=model_feature_names)
                current_features = model_feature_names
        info = {
            "missing_features": [],
            "extra_features": [],
            "current_features": current_features,
            "model_features": model_feature_names,
            "aligned": False,
            "details": ""
        }
        if model_feature_names:
            missing = set(model_feature_names) - set(current_features)
            extra = set(current_features) - set(model_feature_names)
            info['missing_features'] = list(missing); info['extra_features'] = list(extra)
            if missing:
                for f in missing: X_aligned[f] = 0
                info['details'] += f"Added {len(missing)} missing features with default 0\n"
            if extra:
                X_aligned = X_aligned.drop(columns=list(extra))
                info['details'] += f"Removed {len(extra)} extra features not used by model\n"
            try:
                X_aligned = X_aligned[model_feature_names]
                info['aligned'] = True; info['details'] += 'Features successfully aligned with model\n'
            except Exception as e:
                info['details'] += f"Feature alignment error: {e}\n"
        else:
            info['details'] = 'No model feature names available, using original features\n'
            info['aligned'] = True
        return X_aligned, info


# ------------------------------
# 入口
# ------------------------------
def main():
    root = tk.Tk()
    app = ModelComparisonApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
