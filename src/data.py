import copy
import pandas as pd
from .features import StreamingFeaturizer, V_COLS, FEATURE_COLS


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def split_points(n: int, train=0.6, val=0.2):
    return int(n * train), int(n * (train + val))


def build_features(df: pd.DataFrame, snapshot_at: int):
    """Featurize the full chronological stream once. Returns (X, featurizer
    snapshot taken just before row `snapshot_at`, i.e. at the val/test boundary)."""
    f, rows, snap = StreamingFeaturizer(), [], None
    for i, (t, a) in enumerate(zip(df["Time"].values, df["Amount"].values)):
        if i == snapshot_at:
            snap = copy.deepcopy(f)
        rows.append(f.transform_and_update(float(t), float(a)))
    X = pd.concat([df[V_COLS], pd.DataFrame(rows, index=df.index)], axis=1)[FEATURE_COLS]
    return X, snap


def chronological_split(df, X):
    a, b = split_points(len(df))
    y = df["Class"].values
    return (X.iloc[:a], y[:a]), (X.iloc[a:b], y[a:b]), (X.iloc[b:], y[b:])
