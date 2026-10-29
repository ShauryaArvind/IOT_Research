"""
Shared data protocol for every experiment in this package.

Rules enforced here (see README, "Evaluation protocol"):
  * Reported test numbers come ONLY from the isolated test set
    (X_test_isolated.csv / y_test_isolated.csv). It is never resampled and
    never used for checkpoint selection.
  * The validation split used for model selection / early stopping is carved
    out of the training rows BEFORE any resampling, so SMOTE / Tomek / cGAN
    output never reaches validation or test.
  * Exactly one preprocessor (NaN/inf fill values + standardisation) is fitted
    per (data source, seed) on real training rows, saved under
    models/scalers/, and re-loaded by every downstream script. Nothing refits
    a scaler on its own.

Two training sources are supported:
  'raw'              X_train_raw.csv / y_train_raw.csv (output of data_split.py,
                     before balancing). Required for the balancing ablation.
  'legacy_balanced'  X_train_final_balanced.csv (the committed SMOTE+Tomek
                     output). Usable, but its validation split necessarily
                     contains SMOTE rows, so validation scores are optimistic.
                     Test scores are still clean.

Set IOT_DATA_DIR to point at a directory holding the CSVs if they are not in
the repository root, and IOT_OUTPUT_DIR to write runs / scalers / results
somewhere other than src/risk_engine/.
"""

import hashlib
import json
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parents[1]
MODELS_DIR = PKG_DIR / 'models'
# IOT_OUTPUT_DIR redirects trained runs, scalers and results (e.g. for smoke tests).
_OUT = Path(os.environ['IOT_OUTPUT_DIR']) if os.environ.get('IOT_OUTPUT_DIR') else None
RUNS_DIR = (_OUT or MODELS_DIR) / 'runs'
SCALERS_DIR = (_OUT or MODELS_DIR) / 'scalers'
RESULTS_DIR = (_OUT / 'results') if _OUT else PKG_DIR / 'results'

BENIGN = 0
NUM_CLASSES = 10

RAW_FILES = ('X_train_raw.csv', 'y_train_raw.csv')
LEGACY_FILES = ('X_train_final_balanced.csv', 'y_train_final_balanced.csv')
TEST_FILES = ('X_test_isolated.csv', 'y_test_isolated.csv')


def data_dir() -> Path:
    return Path(os.environ.get('IOT_DATA_DIR', REPO_ROOT))


def cache_dir() -> Path:
    return data_dir() / '.cache'


def _paths(files: Tuple[str, str]) -> Tuple[Path, Path]:
    return data_dir() / files[0], data_dir() / files[1]


def raw_train_available() -> bool:
    return all(p.exists() for p in _paths(RAW_FILES))


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def _check_not_lfs_pointer(path: Path) -> None:
    with open(path, 'rb') as f:
        head = f.read(64)
    if head.startswith(b'version https://git-lfs'):
        raise RuntimeError(
            f"{path} is a Git LFS pointer, not the data itself. "
            f"Run `git lfs install && git lfs pull` first."
        )


def read_csv_cached(path: Path) -> pd.DataFrame:
    """pd.read_csv with a pickle cache keyed on path, size and mtime (the
    feature CSVs are ~240 MB and every script reads them)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    _check_not_lfs_pointer(path)
    st = path.stat()
    key = hashlib.md5(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    cached = cache_dir() / f"{path.stem}_{key}.pkl"
    if cached.exists():
        return pd.read_pickle(cached)
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    try:
        cached.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cached)
    except OSError:
        pass
    return df


def read_xy(files: Tuple[str, str]) -> Tuple[pd.DataFrame, np.ndarray]:
    x_path, y_path = _paths(files)
    X = read_csv_cached(x_path)
    y = read_csv_cached(y_path).iloc[:, 0].to_numpy(np.int64)
    if len(X) != len(y):
        raise ValueError(f"{x_path.name} has {len(X)} rows but {y_path.name} has {len(y)}")
    return X, y


# ---------------------------------------------------------------------------
# Preprocessor: one fitted object, saved once, loaded everywhere
# ---------------------------------------------------------------------------

@dataclass
class Preprocessor:
    """NaN/inf imputation + standardisation, plus the raw-space statistics the
    constrained attacks need (per-feature bounds and integer-valued mask)."""
    feature_names: List[str]
    fill_values: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    raw_min: np.ndarray
    raw_max: np.ndarray
    integer_mask: np.ndarray
    source: str = ''
    seed: int = 0

    @classmethod
    def fit(cls, X_raw: np.ndarray, feature_names: List[str], source: str, seed: int) -> 'Preprocessor':
        X = np.array(X_raw, dtype=np.float64)
        X[~np.isfinite(X)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            fill = np.nanmedian(X, axis=0)
        fill = np.where(np.isfinite(fill), fill, 0.0)
        X = np.where(np.isnan(X), fill, X)
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        scale = np.where(std > 0, std, 1.0)
        is_int = np.all(np.mod(X, 1.0) == 0, axis=0) & (std > 0)
        return cls(list(feature_names), fill, mean, scale, X.min(axis=0), X.max(axis=0),
                   is_int, source, seed)

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def clean(self, X_raw) -> np.ndarray:
        X = np.array(X_raw, dtype=np.float64)
        bad = ~np.isfinite(X)
        if bad.any():
            X[bad] = np.broadcast_to(self.fill_values, X.shape)[bad]
        return X

    def transform(self, X_raw) -> np.ndarray:
        return ((self.clean(X_raw) - self.mean) / self.scale).astype(np.float32)

    def inverse(self, X_scaled) -> np.ndarray:
        return np.asarray(X_scaled, dtype=np.float64) * self.scale + self.mean

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        with open(path.with_suffix('.json'), 'w') as f:
            json.dump(self.to_json_dict(), f, indent=2)

    def to_json_dict(self) -> Dict:
        """Format consumed by the ONNX edge runtime (edge_inference_engine.py)."""
        return {
            'input_dim': self.n_features,
            'feature_names': self.feature_names,
            'fill_values': self.fill_values.tolist(),
            'mean': self.mean.tolist(),
            'scale': self.scale.tolist(),
            'var': (self.scale ** 2).tolist(),
            'source': self.source,
            'seed': self.seed,
        }

    @staticmethod
    def load(path: Path) -> 'Preprocessor':
        return joblib.load(path)


def scaler_path(source: str, seed: int) -> Path:
    return SCALERS_DIR / f'{source}__seed{seed}.joblib'


# ---------------------------------------------------------------------------
# Resampling (applied to the training portion only)
# ---------------------------------------------------------------------------

RESAMPLING_METHODS = ('none', 'smote', 'smote_tomek', 'cgan', 'legacy_balanced')


def resample(X: np.ndarray, y: np.ndarray, method: str, seed: int,
             target: Optional[int] = None, cgan_epochs: int = 60):
    """Resample the (already scaled) training portion. Returns X, y and a
    boolean mask marking synthetic rows.

    Minority attack classes are raised to `target` rows (default: the size of
    the largest attack class). Benign is never touched here; Tomek links
    (strategy='all') may remove rows from any class.
    """
    is_synth = np.zeros(len(y), dtype=bool)
    if method in ('none', 'legacy_balanced'):
        return X, y, is_synth

    counts = np.bincount(y, minlength=NUM_CLASSES)
    if target is None:
        target = int(np.delete(counts, BENIGN).max())
    strategy = {c: target for c in range(NUM_CLASSES)
                if c != BENIGN and 0 < counts[c] < target}
    if not strategy:
        return X, y, is_synth

    if method in ('smote', 'smote_tomek'):
        from imblearn.over_sampling import SMOTE
        k = max(1, min(5, min(counts[c] for c in strategy) - 1))
        Xr, yr = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed).fit_resample(X, y)
        # imblearn appends synthetic rows after the originals
        is_synth = np.arange(len(yr)) >= len(y)
        if method == 'smote_tomek':
            from imblearn.under_sampling import TomekLinks
            tomek = TomekLinks(sampling_strategy='all', n_jobs=-1)
            Xr, yr = tomek.fit_resample(Xr, yr)
            is_synth = is_synth[tomek.sample_indices_]
        return Xr.astype(np.float32), yr.astype(np.int64), is_synth

    if method == 'cgan':
        from gan_augmentation import augment_with_cgan
        need = {c: t - int(counts[c]) for c, t in strategy.items()}
        X_new, y_new = augment_with_cgan(X, y, need, seed=seed, epochs=cgan_epochs)
        Xr = np.vstack([X, X_new]).astype(np.float32)
        yr = np.concatenate([y, y_new]).astype(np.int64)
        is_synth = np.arange(len(yr)) >= len(y)
        return Xr, yr, is_synth

    raise ValueError(f"Unknown resampling method {method!r}; choose from {RESAMPLING_METHODS}")


def undersample_class(X, y, is_synth, cls: int, cap: int, seed: int):
    idx = np.where(y == cls)[0]
    if len(idx) <= cap:
        return X, y, is_synth
    rng = np.random.default_rng(seed)
    drop = rng.choice(idx, size=len(idx) - cap, replace=False)
    keep = np.setdiff1d(np.arange(len(y)), drop)
    return X[keep], y[keep], is_synth[keep]


def row_hashes(X: np.ndarray) -> np.ndarray:
    """Hash each row's exact float64 byte pattern (for duplicate detection)."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    return np.array([hashlib.blake2b(r.tobytes(), digest_size=12).hexdigest() for r in X])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    prep: Preprocessor
    train_is_synthetic: np.ndarray
    info: Dict = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return self.X_train.shape[1]


def load_splits(resampling: str = 'legacy_balanced', seed: int = 42, val_size: float = 0.1,
                benign_cap: Optional[int] = None, dedup_test: bool = False,
                smote_target: Optional[int] = None, verbose: bool = True,
                skip_resampling: bool = False) -> Splits:
    """Build train / val / test arrays under the leakage-free protocol.

    `skip_resampling=True` returns the un-resampled training portion (used by
    evaluation-only scripts that just need the val/test arrays and the
    preprocessor, so they don't pay for SMOTE or cGAN).
    """
    if resampling not in RESAMPLING_METHODS:
        raise ValueError(f"Unknown resampling {resampling!r}; choose from {RESAMPLING_METHODS}")
    source = 'legacy_balanced' if resampling == 'legacy_balanced' else 'raw'
    notes = []

    if source == 'raw' and not raw_train_available():
        raise FileNotFoundError(
            f"Resampling '{resampling}' needs the un-balanced training split "
            f"({RAW_FILES[0]}, {RAW_FILES[1]}). Regenerate it with data_split.py, or use "
            f"resampling='legacy_balanced'."
        )

    X_df, y = read_xy(RAW_FILES if source == 'raw' else LEGACY_FILES)
    X_test_df, y_test = read_xy(TEST_FILES)
    names = list(X_df.columns)
    missing = [c for c in names if c not in X_test_df.columns]
    if missing:
        raise ValueError(f"Test set is missing training columns: {missing[:5]}...")
    X_test_df = X_test_df[names]

    X_raw = X_df.to_numpy(np.float64)
    X_test_raw = X_test_df.to_numpy(np.float64)

    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=val_size, stratify=y, random_state=seed
    )

    sp = scaler_path(source, seed)
    if sp.exists():
        prep = Preprocessor.load(sp)
        if prep.feature_names != names:
            raise ValueError(f"Saved preprocessor {sp} was fitted on different columns; delete it to refit.")
    else:
        prep = Preprocessor.fit(X_raw[idx_tr], names, source, seed)
        prep.save(sp)
        notes.append(f"fitted and saved preprocessor -> {sp}")

    X_tr, y_tr = prep.transform(X_raw[idx_tr]), y[idx_tr]
    X_val, y_val = prep.transform(X_raw[idx_val]), y[idx_val]
    X_te = prep.transform(X_test_raw)

    n_test_dups = None
    if dedup_test:
        train_h = set(row_hashes(prep.clean(X_raw)))
        test_h = row_hashes(prep.clean(X_test_raw))
        keep = np.array([h not in train_h for h in test_h])
        n_test_dups = int((~keep).sum())
        X_te, y_test = X_te[keep], y_test[keep]
        notes.append(f"dropped {n_test_dups} test rows that exactly duplicate a training row")

    is_synth = np.zeros(len(y_tr), dtype=bool)
    if benign_cap is not None:
        X_tr, y_tr, is_synth = undersample_class(X_tr, y_tr, is_synth, BENIGN, benign_cap, seed)
        notes.append(f"benign capped at {benign_cap} in train")
    if not skip_resampling:
        X_tr, y_tr, is_synth = resample(X_tr, y_tr, resampling, seed, smote_target)

    if source == 'legacy_balanced':
        notes.append("validation split is drawn from SMOTE+Tomek output, so it contains synthetic "
                     "rows; val scores are optimistic. Test scores are unaffected.")

    test_counts = np.bincount(y_test, minlength=NUM_CLASSES)
    train_counts = np.bincount(y_tr, minlength=NUM_CLASSES)
    synth_counts = np.bincount(y_tr[is_synth], minlength=NUM_CLASSES)
    if source == 'legacy_balanced':
        # Stratified 80/20 split: real train rows per class ~= 4x test rows.
        est_real = np.minimum(train_counts, np.round(test_counts * 4 * (1 - val_size)))
        synth_frac = 1.0 - est_real / np.maximum(train_counts, 1)
        synth_note = 'estimated from test counts (legacy file does not mark synthetic rows)'
    else:
        synth_frac = synth_counts / np.maximum(train_counts, 1)
        synth_note = 'exact'

    info = {
        'source': source, 'resampling': resampling, 'seed': seed, 'val_size': val_size,
        'benign_cap': benign_cap, 'dedup_test': dedup_test, 'n_test_duplicates_dropped': n_test_dups,
        'n_features': len(names),
        'train_counts': train_counts.tolist(), 'val_counts': np.bincount(y_val, minlength=NUM_CLASSES).tolist(),
        'test_counts': test_counts.tolist(),
        'train_synthetic_fraction': np.round(synth_frac, 4).tolist(),
        'synthetic_fraction_note': synth_note, 'notes': notes,
    }
    if verbose:
        print(f"[data] source={source} resampling={resampling} seed={seed} features={len(names)}")
        print(f"[data] train={train_counts.tolist()}")
        print(f"[data] val  ={info['val_counts']}")
        print(f"[data] test ={test_counts.tolist()} (isolated)")
        for n in notes:
            print(f"[data] NOTE: {n}")
    return Splits(X_tr, y_tr, X_val, y_val, X_te, y_test, prep, is_synth, info)
