# -*- coding: utf-8 -*-
from typing import List, Optional, Dict, Any
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer
from sdv.sampling import Condition


def _build_conditions(
    df: pd.DataFrame,
    target_col: str,
    n_new_total: int,
    balance: Optional[str],
    target_proportions: Optional[Dict[Any, float]]
) -> Optional[List[Condition]]:
    """Use balance to generate Condition objects for sample_from_conditions."""
    if (balance is None) or (target_col not in df.columns) or (n_new_total <= 0):
        return None

    vc = df[target_col].value_counts(dropna=False)

    # Does not generate condition for na classes
    classes = [c for c in vc.index if pd.notna(c)]
    if not classes:
        return None

    # According to balance strategy, calculate how many samples are needed for each class
    req: Dict[Any, int]
    if balance == "match_max":
        max_count = int(vc.loc[classes].max())
        needs = {cls: max(max_count - int(vc.get(cls, 0)), 0) for cls in classes}
        total_need = sum(needs.values())
        if total_need == 0:
            return None
        scale = min(1.0, n_new_total / total_need) if n_new_total else 1.0
        req = {cls: int(round(needs[cls] * scale)) for cls in classes}

    elif balance == "by_proportion":
        if not target_proportions:
            raise ValueError("balance='by_proportion' requires target_proportions.")
        p_map = {cls: float(target_proportions.get(cls, 0.0)) for cls in classes}
        s = sum(p_map.values())
        if s <= 0:
            raise ValueError("sum of target_proportions must be > 0.")
        req = {cls: int(round(n_new_total * (p_map[cls] / s))) for cls in classes}

    else:
        raise ValueError("balance must be None / 'match_max' / 'by_proportion'.")

    # Convert to Condition objects for classes with positive requirements
    conditions = [
        Condition(column_values={target_col: cls}, num_rows=int(n))
        for cls, n in req.items() if int(n) > 0
    ]
    return conditions if conditions else None


def tvae_augment(
    df: pd.DataFrame,
    *,
    continuous_cols: List[str],
    ignore_cols: Optional[List[str]] = None,
    # Number of new rows to generate; use either n_new or ratio, not both
    n_new: Optional[int] = None,
    ratio: float = 0.5,
    # Target column for conditional sampling
    target_col: Optional[str] = None,
    balance: Optional[str] = None,  # None | "match_max" | "by_proportion"
    target_proportions: Optional[Dict[Any, float]] = None,
    # TVAE parameters
    tvae_params: Optional[Dict] = None,
    random_state: int = 42,
    # Whether to add a source column to distinguish real vs synthetic data
    add_source_col: bool = True,
    shuffle_output: bool = True
) -> pd.DataFrame:
    """
    Use TVAE to augment tabular data. All columns except continuous_cols and ignore_cols are treated as categorical.
    Use target_col + balance for conditional sampling (using sdv.sampling.Condition).

    parameters:
    - df: Input DataFrame to augment.
    - continuous_cols: List of columns to treat as continuous (numerical).
    - ignore_cols: List of columns to ignore (not used in augmentation).
    - n_new: Number of new rows to generate; if None, uses ratio.
    - ratio: Ratio of new rows to original rows; used if n_new is None.
    - target_col: Column to condition on for sampling; if None, unconditional sampling.
    - balance: How to balance classes in target_col; None / "match_max" / "by_proportion".
    - target_proportions: Proportions for "by_proportion" balance; dict mapping class to proportion.
    - tvae_params: Additional parameters for TVAE in {}.
    - random_state: Random seed for reproducibility.
    - add_source_col: Whether to add a "source" column to distinguish real vs synthetic data.
    - shuffle_output: Whether to shuffle the output DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be pandas.DataFrame.")

    # Skip missing columns
    ignore_cols = list(ignore_cols or [])
    continuous_cols = list(continuous_cols or [])

    missing_cont = [c for c in continuous_cols if c not in df.columns]
    missing_ignore = [c for c in ignore_cols if c not in df.columns]
    if missing_cont:
        print(f"[TVAE] Continuous columns missing, skipped: {missing_cont}")
    if missing_ignore:
        print(f"[TVAE] Ignored columns missing, skipped: {missing_ignore}")

    continuous_cols = [c for c in continuous_cols if c in df.columns]
    ignore_cols = [c for c in ignore_cols if c in df.columns]

    # Valid columns
    model_cols = [c for c in df.columns if c not in ignore_cols]
    if len(model_cols) == 0:
        raise ValueError("No columns available for modeling (all excluded by ignore_cols).")

    # target_col must be included in modeling; otherwise, conditional sampling is not possible
    if target_col is not None:
        if target_col in ignore_cols:
            print(f"[TVAE] target_col='{target_col}' is in ignore_cols, cannot be used for balancing, will ignore balance.")
            target_col = None
            balance = None
        elif target_col not in model_cols:
            print(f"[TVAE] target_col='{target_col}' is not in model_cols, cannot be used for balancing, will ignore balance.")
            target_col = None
            balance = None

    # Create metadata for SDV, treating all non-continuous columns as categorical
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df[model_cols])
    for c in model_cols:
        if c in continuous_cols:
            metadata.update_column(column_name=c, sdtype="numerical")
        else:
            metadata.update_column(column_name=c, sdtype="categorical")

    # Train the TVAE synthesizer
    tvae_params = tvae_params or {}
    synth = TVAESynthesizer(metadata, verbose=False, **tvae_params)
    synth.fit(df[model_cols])

    # Determine number of new rows to generate
    n_new_rows = int(round(len(df) * ratio)) if n_new is None else int(n_new)
    if n_new_rows <= 0:
        out = df.copy()
        if add_source_col:
            out["source"] = "real"
        return out

    # Build conditions for conditional sampling if target_col is specified
    conditions = (
        _build_conditions(
            df=df[model_cols],
            target_col=target_col,
            n_new_total=n_new_rows,
            balance=balance,
            target_proportions=target_proportions
        )
        if target_col is not None else None
    )

    if conditions is None:
        synth_df = synth.sample(num_rows=n_new_rows)
    else:
        synth_df = synth.sample_from_conditions(conditions)

    # Combine synthetic data with original data
    out_cols = list(df.columns)
    synth_full = pd.DataFrame(columns=out_cols)
    for c in model_cols:
        synth_full[c] = synth_df[c]
    for c in ignore_cols:
        synth_full[c] = pd.NA

    real_df = df.copy()
    if add_source_col:
        if "source" not in real_df.columns:
            real_df["source"] = "real"
            synth_full["source"] = "synthetic"
        else:
            synth_full["__source__"] = "synthetic"

    out = pd.concat([real_df, synth_full], axis=0, ignore_index=True)

    if shuffle_output:
        out = out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return out
