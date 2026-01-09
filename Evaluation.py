#!/usr/bin/env python3
# -*- coding: utf-8 -*-


r"""
Evaluates all columns AFTER a gold-standard column (0/1) against that gold standard.
Prediction columns contain y/Y (=1) or n/N (=0); empty cells are ignored by default.

Example:
  python Evaluation.py "/path/to/results.csv" --gold "§130 StGB - Goldstandard" -o "/path/to/evaluation_results.csv"

Useful options:
  --sep ";"                      If the CSV uses semicolons
  --encoding latin1              If umlauts/special characters look wrong
  --include-empty-as-negative    Treat empty predictions as 0/N instead of ignoring them
"""

import argparse, os, sys
import numpy as np
import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser(description="Calc Accuracy, F1, F2, Balanced Accuracy and Cohen's Kappa against Goldstandard.")
    ap.add_argument("input", help="Path to Input-CSV")
    ap.add_argument("--gold", required=True,
                    help="Exact Column Name of Goldstandard (0/1). All columns right of this will be evaluated.")
    ap.add_argument("-o", "--output", default=None, help="Path to Result-CSV (Default: next to input)")
    ap.add_argument("--sep", default=",", help="CSV-Delimiter (Default ',')")
    ap.add_argument("--encoding", default="utf-8", help="Datei-Encoding (z. B. 'utf-8', 'latin1', 'utf-8-sig')")
    ap.add_argument("--include-empty-as-negative", action="store_true",
                    help="Leere/fehlende Vorhersagen als 0/N behandeln (Default: Zeilen mit fehlender Vorhersage ignorieren)")
    return ap.parse_args()

def make_default_output(input_path):
    base, _ = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(os.path.abspath(input_path)),
                        f"{base}_metrics_by_model.csv")

def normalize_gold(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = s2.where(s2.isin([0,1]), pd.NA)
    return s2.astype("Float64")

def normalize_pred(s: pd.Series) -> pd.Series:
    # y/Y/1/true/ja/j -> 1, n/N/0/false/nein/f -> 0, else NaN
    def _map(v):
        if pd.isna(v):
            return pd.NA
        t = str(v).strip().lower()
        if t in ("y","yes","1","true","ja","j","t"):
            return 1
        if t in ("n","no","0","false","nein","f"):
            return 0
        if t == "":
            return pd.NA
        return pd.NA
    out = s.map(_map)
    return out.astype("Float64")

def confusion_counts(y_true: pd.Series, y_pred: pd.Series):
    # Expects 0/1 without NaN
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n  = tp + tn + fp + fn
    return tp, tn, fp, fn, n

# Cohen’s Kappa from counts (robust to edge cases)
def cohen_kappa_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    n = tp + tn + fp + fn
    if n == 0:
        return 0.0
    po = (tp + tn) / n  # observed agreement (accuracy)
    # Class marginals
    p_true_pos = (tp + fn) / n
    p_true_neg = (tn + fp) / n
    p_pred_pos = (tp + fp) / n
    p_pred_neg = (tn + fn) / n
    pe = p_true_pos * p_pred_pos + p_true_neg * p_pred_neg  # expected by chance
    denom = 1.0 - pe
    # Handle degenerate case where denom ~ 0 (identical marginals). If perfect agreement, set κ=1, else 0.
    if np.isclose(denom, 0.0):
        return 1.0 if np.isclose(po, 1.0) else 0.0
    return (po - pe) / denom

def precision_recall_f(tp, fp, fn, beta=1.0):
    # Precision/Recall/F-beta with robust 0-handling
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if prec == 0.0 and rec == 0.0:
        fbeta = 0.0
    else:
        b2 = beta * beta
        denom = b2 * prec + rec
        fbeta = (1 + b2) * prec * rec / denom if denom > 0 else 0.0
    return prec, rec, fbeta

def smart_read_csv(path, sep, enc_try_first="utf-8"):
    encodings = [enc_try_first, "utf-8-sig", "cp1252", "latin1", "iso-8859-1"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(path, sep=sep, encoding=enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err

def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"File not found: {args.input}")

    try:
        df = smart_read_csv(args.input, sep=args.sep, enc_try_first=args.encoding)
    except Exception as e:
        sys.exit(f"CSV couldn't be read: {e}")

    if args.gold not in df.columns:
        sys.exit(f"Goldstandard-Column '{args.gold}' not found.\nExisting columns:\n{list(df.columns)}")

    # Normalise Goldstandard
    y_true_all = normalize_gold(df[args.gold])

    # Result Columns: all columns to the right of Goldstandard
    gold_idx = list(df.columns).index(args.gold)
    pred_cols = list(df.columns[gold_idx+1:])
    if not pred_cols:
        sys.exit("No columns found right of goldstandard.")

    rows = []
    for col in pred_cols:
        y_pred_raw = df[col]
        y_pred = normalize_pred(y_pred_raw)

        if args.include_empty_as_negative:
            y_pred = y_pred.fillna(0)

        mask = y_true_all.notna()
        if not args.include_empty_as_negative:
            mask &= y_pred.notna()

        y_true = y_true_all[mask].astype(int)
        y_pred_used = y_pred[mask].astype(int)

        tp, tn, fp, fn, n = confusion_counts(y_true, y_pred_used)
        acc = (tp + tn) / n if n > 0 else 0.0

        # Precision, Recall (TPR) and F1/F2
        prec, rec, f1 = precision_recall_f(tp, fp, fn, beta=1.0)
        _,   _,   f2 = precision_recall_f(tp, fp, fn, beta=2.0)

        # Specificity (TNR) and Balanced Accuracy
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        balanced_acc = 0.5 * (rec + tnr)

        # Cohen’s Kappa
        kappa = cohen_kappa_from_counts(tp, tn, fp, fn)

        rows.append({
            "model_column": col,
            "f2": round(f2, 2),
            "cohen_kappa": round(float(kappa), 2),
            "balanced_accuracy": round(balanced_acc, 2),
            "f1": round(f1, 2),
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "accuracy": round(acc, 2),
            "evaluated_rows": n,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        })
        
    out_df = pd.DataFrame(rows).sort_values(
        ["f2","cohen_kappa","f1"], ascending=[False, False, False]
    )

    output_path = args.output or make_default_output(args.input)
    try:
        out_df.to_csv(output_path, index=False, encoding="utf-8")
    except Exception as e:
        sys.exit(f"Result-CSV couldn't be written: {e}")

    # Compact Output
    display_cols = ["model_column","evaluated_rows","f2","f1","cohen_kappa","balanced_accuracy","accuracy","precision","recall","tp","fp","fn","tn"]
    print(out_df[display_cols].to_string(index=False))
    print(f"\nResults written to: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()
