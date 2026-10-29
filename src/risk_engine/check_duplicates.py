"""
Exact-duplicate audit across train / test (CIC flow datasets contain many
identical flow records, which inflates test scores when they straddle the split).

Reports:
  * duplicate rows inside the isolated test set,
  * test rows whose feature vector also occurs in the training file
    (and how many of those carry a different label),
  * conflicting labels for identical feature vectors inside each file,
  * per-class breakdown of test rows that are seen in train.

Run with --write-dedup to also write X_test_isolated_dedup.csv /
y_test_isolated_dedup.csv. Every evaluation can instead pass dedup_test=True
to data.load_splits.

    python src/risk_engine/check_duplicates.py [--train raw|legacy_balanced] [--write-dedup]
"""

import argparse

import numpy as np
import pandas as pd

from data import (LEGACY_FILES, NUM_CLASSES, RAW_FILES, TEST_FILES, data_dir, raw_train_available,
                  read_xy, row_hashes)


def _clean(df):
    X = df.to_numpy(np.float64)
    X[~np.isfinite(X)] = np.nan
    return np.nan_to_num(X, nan=-1.2345e300)  # stable sentinel so NaN rows hash equal


def label_conflicts(h, y):
    df = pd.DataFrame({'h': h, 'y': y})
    n_labels = df.groupby('h')['y'].nunique()
    return int((n_labels > 1).sum()), int(df['h'].isin(n_labels[n_labels > 1].index).sum())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', choices=['raw', 'legacy_balanced'], default=None)
    ap.add_argument('--write-dedup', action='store_true')
    a = ap.parse_args(argv)
    source = a.train or ('raw' if raw_train_available() else 'legacy_balanced')

    X_tr, y_tr = read_xy(RAW_FILES if source == 'raw' else LEGACY_FILES)
    X_te, y_te = read_xy(TEST_FILES)
    X_te = X_te[list(X_tr.columns)]
    print(f"train ({source}): {len(y_tr)} rows | isolated test: {len(y_te)} rows")

    h_tr = row_hashes(_clean(X_tr))
    h_te = row_hashes(_clean(X_te))

    te_dup = pd.Series(h_te).duplicated(keep='first').to_numpy()
    print(f"\nduplicates inside test:  {te_dup.sum()} rows ({te_dup.mean():.2%})")
    tr_dup = pd.Series(h_tr).duplicated(keep='first').to_numpy()
    print(f"duplicates inside train: {tr_dup.sum()} rows ({tr_dup.mean():.2%})")

    g, r = label_conflicts(h_tr, y_tr)
    print(f"train: {g} feature vectors carry >1 label ({r} rows)")
    g, r = label_conflicts(h_te, y_te)
    print(f"test:  {g} feature vectors carry >1 label ({r} rows)")

    train_set = set(h_tr)
    seen = np.array([h in train_set for h in h_te])
    print(f"\ntest rows whose exact feature vector appears in train: {seen.sum()} ({seen.mean():.2%})")
    first_label = pd.Series(y_tr, index=h_tr).groupby(level=0).first()
    mism = np.array([seen[i] and first_label[h_te[i]] != y_te[i] for i in range(len(h_te))])
    print(f"  ... of which with a different label in train: {mism.sum()}")

    print("\nper class: test rows | seen in train | fraction")
    for c in range(NUM_CLASSES):
        m = y_te == c
        if m.any():
            print(f"  {c}: {m.sum():>7} | {seen[m].sum():>7} | {seen[m].mean():.2%}")

    if a.write_dedup:
        keep = ~seen
        out_x = data_dir() / 'X_test_isolated_dedup.csv'
        out_y = data_dir() / 'y_test_isolated_dedup.csv'
        X_te[keep].to_csv(out_x, index=False)
        pd.DataFrame({'Label': y_te[keep]}).to_csv(out_y, index=False)
        print(f"\nwrote {keep.sum()} rows -> {out_x}, {out_y}")


if __name__ == '__main__':
    main()
