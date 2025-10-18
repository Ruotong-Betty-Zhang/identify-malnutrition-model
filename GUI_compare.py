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
import os

warnings.filterwarnings('ignore')


def show_error(message: str) -> None:
    """Centralized error dialog."""
    try:
        messagebox.showerror("Error", message)
    except Exception:
        # Fallback for rare cases where Tk messagebox is not available
        print(f"[Error] {message}")

def _is_probably_dataframe(obj) -> bool:
    """Return True if obj looks like a pandas DataFrame."""
    return isinstance(obj, pd.DataFrame)

def _is_probably_model(obj) -> bool:
    """Return True if obj looks like a trained model/estimator."""
    # Has a predict method; common for sklearn/xgboost etc.
    return hasattr(obj, "predict") and callable(getattr(obj, "predict", None))

# ------------------------------
# Scrollable container
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

        # Bind mouse wheel only when hovering over this widget
        for w in (self, self.canvas, self.inner):
            w.bind("<Enter>", self._bind_mousewheel)
            w.bind("<Leave>", self._unbind_mousewheel)

    @property
    def body(self):
        return self.inner

    # Windows/macOS: <MouseWheel>
    def _on_mousewheel(self, event):
        step = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(step * 3, "units")

    # Linux: Button-4/5
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
# Result panel: render all outputs for a single model
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



    # ---------- Renderers ----------
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
            if isinstance(model_classes, (list, np.ndarray)):
                model_classes = list(model_classes)
            uniq_y = list(np.unique(y_test))

            # ========== Binary classification: Show only one tab (SHAP for positive class) ==========
            if (model_classes and len(model_classes) == 2) or (not model_classes and len(uniq_y) == 2):
                # Determine positive and negative class labels
                if model_classes and len(model_classes) == 2:
                    neg_label, pos_label = model_classes[0], model_classes[1]
                else:
                    # No classes_, use the larger sorted value from y_test as positive class
                    neg_label, pos_label = sorted(uniq_y)[0], sorted(uniq_y)[1]

                # Get SHAP values for the 'positive class'
                if len(sv_list) == 1:
                    sv_pos = np.asarray(sv_list[0])  # Common case: only positive class returned
                else:
                    # When two groups exist, take the second group by classes order, otherwise use index=1 as fallback
                    idx_pos = 1 if len(sv_list) > 1 else 0
                    sv_pos = np.asarray(sv_list[idx_pos])

                if sv_pos.ndim != 2:
                    sv_pos = sv_pos.reshape(sv_pos.shape[0], -1)

                # Mean |SHAP| for radar chart usage
                mean_abs = np.mean(np.abs(sv_pos), axis=0)

                # --- Single tab ---
                notebook = ttk.Notebook(parent); notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                class_frame = ttk.Frame(notebook)
                notebook.add(class_frame, text=f"Binary SHAP (pos={pos_label}, neg={neg_label})")

                class_nb = ttk.Notebook(class_frame); class_nb.pack(fill=tk.BOTH, expand=True)

                # 1) Beeswarm (all samples, no class filtering)
                beeswarm_frame = ttk.Frame(class_nb); class_nb.add(beeswarm_frame, text="Beeswarm Plot")
                fig_beeswarm = plt.figure(figsize=(12, 8))
                shap.summary_plot(sv_pos, X_num, plot_type="dot", max_display=12, show=False)
                fig_beeswarm.tight_layout()
                canvas = FigureCanvasTkAgg(fig_beeswarm, master=beeswarm_frame)
                canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                self._mpl_refs.append(canvas)
                plt.close(fig_beeswarm)

                # 2) Radar chart (include positive class in title)
                radar_frame = ttk.Frame(class_nb); class_nb.add(radar_frame, text="Radar Chart")
                self._create_shap_radar_chart(
                    radar_frame,
                    mean_abs,
                    X_num.columns,
                    class_id=f"pos={pos_label}"
                )
                return  # Binary classification ends here

            # ========== Multi-class classification: Keep original "one tab per class + comparison" ==========
            notebook = ttk.Notebook(parent); notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Calculate mean|SHAP| for each class
            shap_importance_by_class = {}
            # Determine class_ids
            if model_classes and len(model_classes) == len(sv_list):
                class_ids = model_classes
            else:
                if len(uniq_y) == len(sv_list):
                    class_ids = uniq_y
                else:
                    class_ids = list(range(len(sv_list)))

            for ci, sv_c in enumerate(sv_list):
                mean_abs_shap = np.mean(np.abs(sv_c), axis=0)
                shap_importance_by_class[class_ids[ci]] = mean_abs_shap

            # One tab per class
            for ci, sv_c in enumerate(sv_list):
                class_frame = ttk.Frame(notebook); notebook.add(class_frame, text=f"Class {class_ids[ci]}")
                class_nb = ttk.Notebook(class_frame); class_nb.pack(fill=tk.BOTH, expand=True)

                # 1) Beeswarm (filtered by samples of this class, show warning if insufficient samples)
                beeswarm_frame = ttk.Frame(class_nb); class_nb.add(beeswarm_frame, text="Beeswarm Plot")
                y_arr = np.asarray(y_test); mask_cur = (y_arr == class_ids[ci])
                if mask_cur.sum() >= 5:
                    fig_beeswarm = plt.figure(figsize=(12, 8))
                    shap.summary_plot(sv_c[mask_cur], X_num.iloc[mask_cur], plot_type="dot", max_display=12, show=False)
                    fig_beeswarm.tight_layout()
                    canvas = FigureCanvasTkAgg(fig_beeswarm, master=beeswarm_frame)
                    canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                    self._mpl_refs.append(canvas)
                    plt.close(fig_beeswarm)
                else:
                    ttk.Label(beeswarm_frame, text=f"Not enough samples for class {class_ids[ci]} (got {mask_cur.sum()}, need ≥5)").pack(pady=20)

                # 2) Radar chart
                radar_frame = ttk.Frame(class_nb); class_nb.add(radar_frame, text="Radar Chart")
                self._create_shap_radar_chart(
                    radar_frame,
                    shap_importance_by_class[class_ids[ci]],
                    X_num.columns,
                    class_ids[ci]
                )

            # Class comparison radar
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

            # ---- Key: Normalize radius to 0~1 for stable display ----
            denom = float(top_importance.max()) if top_importance.size else 1.0
            if denom <= 0: denom = 1.0
            plot_vals = top_importance / denom

            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
            values = np.concatenate([plot_vals, [plot_vals[0]]])

            fig = Figure(figsize=(10, 8))
            ax = fig.add_subplot(111, polar=True)

            # Draw the line
            ax.plot(angles, values, linewidth=2)
            ax.fill(angles, values, alpha=0.25)

            # ---- Key: Provide larger margins to avoid cutting ----
            fig.subplots_adjust(top=0.86, bottom=0.20, left=0.08, right=0.95)

            # Wrap labels automatically to avoid overflow
            import textwrap
            feature_labels = ['\n'.join(textwrap.wrap(str(f), width=16)) for f in top_features]
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(feature_labels, fontsize=9)

            # ---- Key: Fix radius to 1.05 (after normalization) ----
            ax.set_ylim(0, 1.05)
            yticks = np.linspace(0, 1.0, 5)
            ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=8)

            ax.set_title(f"Top {top_n} Features - Class {class_id} (scaled by max |SHAP|= {denom:.3g})",
                        fontsize=14, fontweight='bold', pad=20)
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)

            # Table (keep original values)
            table_frame = ttk.Frame(parent); table_frame.pack(fill=tk.X, padx=10, pady=5)
            tree = ttk.Treeview(table_frame, columns=('Rank','Feature','Mean|SHAP|'), show='headings',
                                height=min(6, len(top_features)))
            tree.heading('Rank', text='Rank'); tree.heading('Feature', text='Feature'); tree.heading('Mean|SHAP|', text='Mean |SHAP|')
            tree.column('Rank', width=50, anchor='center'); tree.column('Feature', width=300)
            tree.column('Mean|SHAP|', width=120, anchor='center')
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

            # ---- Key: Normalize using global maximum across all classes ----
            global_max = float(np.max(all_importances[:, top_indices]))
            if global_max <= 0: global_max = 1.0

            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]

            fig = Figure(figsize=(12, 9))
            ax = fig.add_subplot(111, polar=True)
            fig.subplots_adjust(top=0.86, bottom=0.20, left=0.06, right=0.95)

            colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
            for i, cid in enumerate(class_ids):
                importance = shap_importance_dict[cid][top_indices] / global_max
                values = np.concatenate([importance, [importance[0]]])
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Class {cid}', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.10)

            import textwrap
            feature_labels = ['\n'.join(textwrap.wrap(str(f), width=16)) for f in top_features]
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(feature_labels, fontsize=9)

            ax.set_ylim(0, 1.05)
            yticks = np.linspace(0, 1.0, 5)
            ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=8)

            ax.set_title(f"Top {top_n} Features Comparison Across Classes (scaled by global max |SHAP|={global_max:.3g})",
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.02))
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._mpl_refs.append(canvas)
        except Exception as e:
            ttk.Label(parent, text=f"Comparison radar chart error: {str(e)}").pack(pady=20)

# ------------------------------
# Main app: supports A/B datasets + models and a comparison page
# ------------------------------
class ModelComparisonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Model Evaluation & Comparison Tool")
        self.root.geometry("1400x900")
        self.root.resizable(True, True)

        # Shared preprocessing options
        self.handle_categorical = tk.BooleanVar(value=True)
        self.handle_datetime = tk.BooleanVar(value=True)
        self.remove_id = tk.BooleanVar(value=True)

        # A / B paths and formats
        self.dataset_path = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.model_path = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.data_format = {'A': tk.StringVar(value='pkl'), 'B': tk.StringVar(value='pkl')}
        self.target_var = {'A': tk.StringVar(), 'B': tk.StringVar()}
        self.dataset_df = {'A': None, 'B': None}
        self.model = {'A': None, 'B': None}
        self.model_feature_names = {'A': None, 'B': None}
        self.feature_importances = {'A': None, 'B': None}

        # Evaluation results cache (for reuse in comparison tab)
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

        # Top: A / B two-column settings area
        top = ttk.Frame(main)
        top.grid(row=0, column=0, sticky='ew')
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        self._build_side_controls(top, side='A', col=0, label='Model A & Dataset A')
        self._build_side_controls(top, side='B', col=1, label='Model B & Dataset B')

        # Preprocessing options (shared)
        pre = ttk.LabelFrame(main, text='Preprocessing (applies to both A & B)')
        pre.grid(row=1, column=0, sticky='ew', pady=(8,4))
        ttk.Checkbutton(pre, text="Encode categorical", variable=self.handle_categorical).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(pre, text="Process datetime", variable=self.handle_datetime).pack(side=tk.LEFT, padx=8)
        ttk.Checkbutton(pre, text="Remove ID columns", variable=self.remove_id).pack(side=tk.LEFT, padx=8)

        # Action buttons
        btns = ttk.Frame(main)
        btns.grid(row=2, column=0, sticky='ew', pady=(6, 6))

        self.btn_run_a  = ttk.Button(btns, text='Run A', command=lambda: self.run_evaluation('A'))
        self.btn_run_b  = ttk.Button(btns, text='Run B', command=lambda: self.run_evaluation('B'))
        self.btn_run_ab = ttk.Button(btns, text='Run A & B and Compare', command=self.run_compare)

        for b in (self.btn_run_a, self.btn_run_b, self.btn_run_ab):
            b.pack(side=tk.LEFT, padx=5)


        # Results area: top notebook with A/B details and Comparison
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

        # Comparison tab layout
        self._build_comparison_tab()

        # Status bar
        ttk.Label(main, textvariable=self.status_var).grid(row=4, column=0, sticky='w')

    def _set_busy(self, busy: bool):
        state = 'disabled' if busy else 'normal'
        for b in (self.btn_run_a, self.btn_run_b, self.btn_run_ab):
            b.configure(state=state)

    def browse_model(self, side: str):
        file_path = filedialog.askopenfilename(
            filetypes=[("Model files", "*.pkl;*.joblib"), ("All files", "*.*")]
        )
        if not file_path:
            return

        self.model_path[side].set(file_path)

        try:
            obj = joblib.load(file_path)

            # Guard: if user selected a dataset DataFrame here, block and explain.
            if _is_probably_dataframe(obj) and not _is_probably_model(obj):
                self.model_path[side].set("")          # clear the entry box
                self.model[side] = None
                self.model_feature_names[side] = None
                self.feature_importances[side] = None
                show_error(
                    "This file looks like a dataset (pandas DataFrame), not a trained model.\n"
                    "Please select a model file in the Model Path picker."
                )
                self.status_var.set(f"Model {side} load failed (wrong file type)")
                return

            if not _is_probably_model(obj):
                raise ValueError("Unsupported file content: expected a trained model with a 'predict' method.")

            self.model[side] = obj
            self.model_feature_names[side] = self._get_model_feature_names(obj)
            self.feature_importances[side] = self._get_feature_importances(obj)
            self.status_var.set(f"Model {side} loaded")

        except Exception as e:
            # Clean up state on failure
            self.model_path[side].set("")
            self.model[side] = None
            self.model_feature_names[side] = None
            self.feature_importances[side] = None
            show_error(f"Error loading model {side}: {e}")
            self.status_var.set(f"Model {side} load failed")

    def _build_side_controls(self, parent, side: str, col: int, label: str):
        box = ttk.LabelFrame(parent, text=label, padding=8)
        box.grid(row=0, column=col, sticky='nsew', padx=6)
        parent.columnconfigure(col, weight=1)

        # Dataset
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

        # Model
        ttk.Label(box, text='Model Path:').grid(row=3, column=0, sticky='w', pady=3)
        ttk.Entry(box, textvariable=self.model_path[side], width=48).grid(row=3, column=1, sticky='ew', padx=6)
        ttk.Button(box, text='Browse', command=lambda s=side: self.browse_model(s)).grid(row=3, column=2, padx=4)

        for c in (0,1,2):
            box.columnconfigure(c, weight=1)

    def _build_comparison_tab(self):
        """Comparison tab: use internal Notebook to display different comparison items"""
        # Create sub-tabs
        self.cmp_nb = ttk.Notebook(self.tab_cmp)
        self.cmp_nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        # 1) Metrics summary
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

        # 2) Confusion Matrices
        self.cmp_tab_cm = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_cm, text='Confusion Matrices')
        self.cmp_cm_left = ttk.Frame(self.cmp_tab_cm)
        self.cmp_cm_right = ttk.Frame(self.cmp_tab_cm)
        self.cmp_cm_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_cm_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 3) Feature Importance
        self.cmp_tab_fi = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_fi, text='Feature Importance')
        self.cmp_fi_left = ttk.Frame(self.cmp_tab_fi)
        self.cmp_fi_right = ttk.Frame(self.cmp_tab_fi)
        self.cmp_fi_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_fi_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 4) Classification Reports (side-by-side)
        self.cmp_tab_reports = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_reports, text='Reports')
        self.cmp_rep_left = ttk.Frame(self.cmp_tab_reports)
        self.cmp_rep_right = ttk.Frame(self.cmp_tab_reports)
        self.cmp_rep_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        self.cmp_rep_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        # 5) SHAP Top-N Comparison (scrollable)
        self.cmp_tab_shap = ttk.Frame(self.cmp_nb, padding=8)
        self.cmp_nb.add(self.cmp_tab_shap, text='SHAP (Top-N)')

        self.cmp_shap_scroll = ScrollableFrame(self.cmp_tab_shap)
        self.cmp_shap_scroll.pack(fill=tk.BOTH, expand=True)

        # All future content will be placed in this container
        self.cmp_shap_container = self.cmp_shap_scroll.body


    # ---------- Data / Model loading ----------
    def browse_dataset(self, side: str):
        file_types = [("PKL files", "*.pkl"), ("CSV files", "*.csv"), ("All files", "*.*")]
        file_path = filedialog.askopenfilename(filetypes=file_types)
        if not file_path:
            return

        # If the extension suggests a different format, auto-adjust the radio selection.
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            self.data_format[side].set("csv")
        elif ext == ".pkl":
            self.data_format[side].set("pkl")

        self.dataset_path[side].set(file_path)

        try:
            fmt = self.data_format[side].get()
            if fmt == "pkl":
                obj = joblib.load(file_path)

                # Guard: if user selected a model file here, block and explain.
                if _is_probably_model(obj) and not _is_probably_dataframe(obj):
                    self.dataset_path[side].set("")           # clear the entry box
                    self.dataset_df[side] = None
                    show_error(
                        "This file looks like a model (has a predict method), not a dataset.\n"
                        "Please select a dataset file in the Dataset Path picker."
                    )
                    self.status_var.set(f"Dataset {side} load failed (wrong file type)")
                    return

                # Accept DataFrame; also accept objects that round-trip to DataFrame
                if _is_probably_dataframe(obj):
                    df = obj
                else:
                    raise ValueError("Unsupported PKL content: expected a pandas DataFrame.")

            else:  # csv
                df = pd.read_csv(file_path, low_memory=False)

            if not _is_probably_dataframe(df):
                raise ValueError("Unsupported data format (expected a pandas DataFrame).")

            self.dataset_df[side] = df
            combo: ttk.Combobox = getattr(self, f"target_combo_{side}")
            combo["values"] = list(df.columns)
            if df.columns.size > 0:
                self.target_var[side].set(df.columns[-1])
            self.status_var.set(f"Dataset {side} loaded: {df.shape}")

        except Exception as e:
            # Clean up state on failure
            self.dataset_path[side].set("")
            self.dataset_df[side] = None
            show_error(f"Error loading dataset {side}: {e}")
            self.status_var.set(f"Dataset {side} load failed")


    # ---------- Run evaluation & comparison ----------
    def run_evaluation(self, side: str):
        self._set_busy(True)
        def _task():
            try:
                self._run_eval_thread(side)
            finally:
                self.root.after(0, lambda: self._set_busy(False))
        threading.Thread(target=_task, daemon=True).start()

    def run_compare(self):
        self._set_busy(True)
        def _go():
            try:
                self._run_eval_thread('A')
                self._run_eval_thread('B')
                self.root.after(0, self._render_comparison)
            finally:
                self.root.after(0, lambda: self._set_busy(False))
        threading.Thread(target=_go, daemon=True).start()


    def _run_eval_thread(self, side: str):
        try:
            self.status_var.set(f"Running evaluation for {side}...")
            df = self.dataset_df[side]
            model = self.model[side]
            target = self.target_var[side].get()
            if df is None or model is None:
                self.root.after(0, lambda: show_error(f'Please load both dataset and model for {side}.'))
                return
            if not target:
                self.root.after(0, lambda: show_error(f'Please select target column for {side}.'))
                return
            X = df.drop(columns=[target]); y = df[target]
            X_proc, prep_info = self._preprocess_data(X)
            X_align, align_info = self._align_features_with_model(X_proc, self.model_feature_names[side])
            if not align_info.get('aligned', False):
                # Enhanced error message for feature alignment failure
                detailed_message = f"Feature Alignment Failed for {side}:\n\n"
                detailed_message += align_info.get('details', '')
                detailed_message += f"\nCurrent features: {len(align_info.get('current_features', []))}"
                detailed_message += f"\nModel features: {len(align_info.get('model_features', []))}"
                self.root.after(0, lambda: show_error(detailed_message))
                self.status_var.set(f"Evaluation {side} failed - Feature alignment error")
                return
            
            # Show feature alignment details if there are missing or extra features
            if align_info.get('missing_features') or align_info.get('extra_features'):
                detailed_message = f"Feature Alignment Details for {side}:\n\n"
                detailed_message += align_info.get('details', '')
                detailed_message += f"\nCurrent features: {len(align_info.get('current_features', []))}"
                detailed_message += f"\nModel features: {len(align_info.get('model_features', []))}"
                detailed_message += f"\nFinal features: {len(X_align.columns)}"
                self.root.after(0, lambda: messagebox.showinfo("Feature Alignment Details", detailed_message))

            classes = np.unique(y)
            X_train, X_test, y_train, y_test = train_test_split(X_align, y, test_size=0.2, random_state=42)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred, labels=classes)
            report = classification_report(y_test, y_pred, labels=classes, zero_division=0)

            # Cache results
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

            # # Compute and cache SHAP summary for the comparison tab (per-class mean |SHAP|)
            shap_stats = self._compute_shap_summary(model, X_test, y_test)
            self.results[side]['shap_by_class'] = shap_stats["by_class"]
            self.results[side]['shap_feature_names'] = shap_stats["feature_names"]


            # Render to the corresponding panel
            panel = self.panelA if side=='A' else self.panelB
            self.root.after(0, lambda: panel.render_accuracy(acc, prep_info))
            self.root.after(0, lambda: panel.render_cm(cm, classes))
            self.root.after(0, lambda: panel.render_report(report))
            self.root.after(0, lambda: panel.render_fi(self.feature_importances[side], list(X_align.columns)))
            self.root.after(0, lambda: panel.render_align(align_info, X_cols=list(X_test.columns)))
            self.root.after(0, lambda: panel.render_shap(model, X_test, y_test))

            self.status_var.set(f"Evaluation {side} completed")
        except Exception as e:
            self.root.after(0, lambda: show_error(f'Error during evaluation {side}: {e}'))
            self.status_var.set(f"Evaluation {side} failed")


    def _render_comparison(self):
        # --- Read cached results ---
        a = self.results.get('A', {})
        b = self.results.get('B', {})
        if not a or not b:
            return

        # === Derive readable names from selected model paths ===
        name_a = os.path.basename(self.model_path['A'].get()) or 'Model A'
        name_b = os.path.basename(self.model_path['B'].get()) or 'Model B'

        # ========== 1) Metrics tab ==========
        for i in self.cmp_metrics_tree.get_children():
            self.cmp_metrics_tree.delete(i)

        # Update table headers dynamically
        self.cmp_metrics_tree.heading('Model A', text=name_a)
        self.cmp_metrics_tree.heading('Model B', text=name_b)
        self.cmp_metrics_tree.heading('Δ (B-A)', text=f'Δ ({name_b} - {name_a})')

        def _fmt(x):
            return f"{x:.4f}" if isinstance(x, (int, float, np.floating)) else str(x)

        rows = [
            ('Accuracy', a.get('accuracy', np.nan), b.get('accuracy', np.nan)),
        ]
        for name, va, vb in rows:
            delta = (vb - va) if isinstance(va, (int, float, np.floating)) and isinstance(vb, (int, float, np.floating)) else ''
            self.cmp_metrics_tree.insert('', 'end', values=(name, _fmt(va), _fmt(vb), _fmt(delta) if delta != '' else ''))

        # ========== 2) Confusion Matrices tab ==========
        for w in self.cmp_cm_left.winfo_children():
            w.destroy()
        for w in self.cmp_cm_right.winfo_children():
            w.destroy()

        figA = Figure(figsize=(6, 5)); axA = figA.add_subplot(111)
        sns.heatmap(a['cm'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=a['classes'], yticklabels=a['classes'], ax=axA)
        axA.set_title(f'{name_a} - Confusion Matrix'); axA.set_xlabel('Predicted'); axA.set_ylabel('True')
        FigureCanvasTkAgg(figA, master=self.cmp_cm_left).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        figB = Figure(figsize=(6, 5)); axB = figB.add_subplot(111)
        sns.heatmap(b['cm'], annot=True, fmt='d', cmap='Greens',
                    xticklabels=b['classes'], yticklabels=b['classes'], ax=axB)
        axB.set_title(f'{name_b} - Confusion Matrix'); axB.set_xlabel('Predicted'); axB.set_ylabel('True')
        FigureCanvasTkAgg(figB, master=self.cmp_cm_right).get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ========== 3) Feature Importance tab ==========
        for w in self.cmp_fi_left.winfo_children():
            w.destroy()
        for w in self.cmp_fi_right.winfo_children():
            w.destroy()

        top_k = 12
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
            ax_fiA.set_title(f'{name_a} - Top Feature Importances'); ax_fiA.set_xlabel('Importance')
            FigureCanvasTkAgg(fig_fiA, master=self.cmp_fi_left).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.cmp_fi_left, text=f'{name_a}: Feature importance not available').pack(pady=12)

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
            ax_fiB.set_title(f'{name_b} - Top Feature Importances'); ax_fiB.set_xlabel('Importance')
            FigureCanvasTkAgg(fig_fiB, master=self.cmp_fi_right).get_tk_widget().pack(fill=tk.BOTH, expand=True)
        else:
            ttk.Label(self.cmp_fi_right, text=f'{name_b}: Feature importance not available').pack(pady=12)

        # ========== 4) Reports tab ==========
        for w in self.cmp_rep_left.winfo_children():
            w.destroy()
        for w in self.cmp_rep_right.winfo_children():
            w.destroy()

        txtA = scrolledtext.ScrolledText(self.cmp_rep_left, wrap=tk.WORD)
        txtA.pack(fill=tk.BOTH, expand=True)
        txtA.insert(tk.END, f'{name_a} - Classification Report\n\n')
        txtA.insert(tk.END, a.get('report', 'N/A'))
        txtA.config(state=tk.DISABLED)

        txtB = scrolledtext.ScrolledText(self.cmp_rep_right, wrap=tk.WORD)
        txtB.pack(fill=tk.BOTH, expand=True)
        txtB.insert(tk.END, f'{name_b} - Classification Report\n\n')
        txtB.insert(tk.END, b.get('report', 'N/A'))
        txtB.config(state=tk.DISABLED)

        # ========== 5) SHAP Top-N Comparison tab ==========
        for w in self.cmp_shap_container.winfo_children():
            w.destroy()

        ttk.Label(self.cmp_shap_container,
                text='Class Comparison Radar (Top-N by global mean |SHAP|)',
                font=('Arial', 14, 'bold')).pack(pady=6)
        wrap = ttk.Frame(self.cmp_shap_container)
        wrap.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(wrap); left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 4))
        right = ttk.Frame(wrap); right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))

        if a.get('shap_by_class') and a.get('shap_feature_names') is not None:
            self._plot_class_comparison_radar(
                left,
                a['shap_by_class'],
                a['shap_feature_names'],
                title=name_a
            )
        else:
            ttk.Label(left, text=f'{name_a}: SHAP summary not available').pack(pady=12)

        if b.get('shap_by_class') and b.get('shap_feature_names') is not None:
            self._plot_class_comparison_radar(
                right,
                b['shap_by_class'],
                b['shap_feature_names'],
                title=name_b
            )
        else:
            ttk.Label(right, text=f'{name_b}: SHAP summary not available').pack(pady=12)

        self.status_var.set('Comparison refreshed')


    # ---------- Utilities: feature names / importances / preprocessing / alignment ----------
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
    
        # ---------- Compute SHAP Summary: mean|SHAP| by class ----------
    def _compute_shap_summary(self, model, X_test: pd.DataFrame, y_test: pd.Series):
        """
        Returns:
            {
                "feature_names": pd.Index([...]),
                "by_class": { class_id: np.ndarray[feat_dim], ... }   # mean(|SHAP|) for each class
            }
        Notes:
            - For binary classification, use SHAP values for the "positive class" (consistent with common explanations), with keys in format "pos=<label>".
            - For multi-class classification, keys are class identifiers (model.classes_ or unique values from y_test).
        """
        # —— Same numerical processing steps as in ResultPanel (simplified) ——
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

        # —— Compute SHAP values (prefer TreeExplainer, fallback to LinearExplainer) ——
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
                return [arr]  # single output
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
        if isinstance(model_classes, (list, np.ndarray)):
            model_classes = list(model_classes)
        uniq_y = list(np.unique(y_test))

        shap_by_class = {}

        # —— Binary classification: Use SHAP values for 'positive class' for model-wide radar —— 
        if (model_classes and len(model_classes) == 2) or (not model_classes and len(uniq_y) == 2):
            if model_classes and len(model_classes) == 2:
                neg_label, pos_label = model_classes[0], model_classes[1]
            else:
                neg_label, pos_label = sorted(uniq_y)[0], sorted(uniq_y)[1]

            if len(sv_list) == 1:
                sv_pos = np.asarray(sv_list[0])
            else:
                idx_pos = 1 if len(sv_list) > 1 else 0
                sv_pos = np.asarray(sv_list[idx_pos])

            if sv_pos.ndim != 2:
                sv_pos = sv_pos.reshape(sv_pos.shape[0], -1)

            mean_abs = np.mean(np.abs(sv_pos), axis=0)
            shap_by_class[f"pos={pos_label}"] = mean_abs
            return {"feature_names": X_num.columns, "by_class": shap_by_class}

        # —— Multi-class classification: One mean|SHAP| per class ----
        if model_classes and len(model_classes) == len(sv_list):
            class_ids = model_classes
        else:
            class_ids = uniq_y if len(uniq_y) == len(sv_list) else list(range(len(sv_list)))

        for ci, sv_c in enumerate(sv_list):
            mean_abs = np.mean(np.abs(sv_c), axis=0)
            shap_by_class[class_ids[ci]] = mean_abs

        return {"feature_names": X_num.columns, "by_class": shap_by_class}

    # ---------- Plot "class comparison" radar chart (multiple classes overlay for one model) ----------
    def _plot_class_comparison_radar(self, parent, shap_by_class: dict, feature_names, title: str, top_n: int = 12):
        """
        Plot in parent: Select Top-N features based on average across all classes, normalize by "global max mean|SHAP|",
        and overlay all classes on the same radar chart to visually demonstrate "class differences".
        """
        try:
            class_ids = list(shap_by_class.keys())
            if len(class_ids) == 0:
                ttk.Label(parent, text='No SHAP summary available').pack(pady=10); return

            all_importances = np.stack([shap_by_class[c] for c in class_ids])
            mean_importance = np.mean(all_importances, axis=0)
            top_indices = np.argsort(mean_importance)[-top_n:][::-1]
            top_features = [feature_names[i] for i in top_indices]

            global_max = float(np.max(all_importances[:, top_indices]))
            if global_max <= 0: global_max = 1.0

            N = len(top_features)
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
            angles += angles[:1]

            fig = Figure(figsize=(6, 6))
            ax = fig.add_subplot(111, polar=True)
            fig.subplots_adjust(top=0.86, bottom=0.20, left=0.06, right=0.95)

            colors = plt.cm.Set3(np.linspace(0, 1, len(class_ids)))
            for i, cid in enumerate(class_ids):
                importance = shap_by_class[cid][top_indices] / global_max
                values = np.concatenate([importance, [importance[0]]])
                ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Class {cid}', color=colors[i])
                ax.fill(angles, values, color=colors[i], alpha=0.10)

            import textwrap
            feature_labels = ['\n'.join(textwrap.wrap(str(f), width=16)) for f in top_features]
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(feature_labels, fontsize=9)

            ax.set_ylim(0, 1.05)
            yticks = np.linspace(0, 1.0, 5)
            ax.set_yticks(yticks); ax.set_yticklabels([f"{v:.2f}" for v in yticks], fontsize=8)

            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            ax.legend(loc='upper right', bbox_to_anchor=(1.28, 1.02))
            ax.grid(True)

            canvas = FigureCanvasTkAgg(fig, master=parent)
            canvas.draw(); canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            ttk.Label(parent, text=f"SHAP radar error: {str(e)}").pack(pady=12)



# ------------------------------
# Main entry point
# ------------------------------
def main():
    root = tk.Tk()
    app = ModelComparisonApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
