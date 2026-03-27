from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.models.bitcoin.config import (
    BITCOIN_DATA_PATH,
    DATE_COL,
    FEATURE_COLS,
    LAG_STEPS,
    TARGET_COL,
    TEST_RATIO,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def load_bitcoin_data() -> pd.DataFrame:
    """Load processed Bitcoin feature data."""
    return pd.read_csv(BITCOIN_DATA_PATH)


def prepare_bitcoin_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare Bitcoin data for t-1 -> t prediction.

    Steps:
    1. Parse and sort dates
    2. Create lagged features X(t-1)
    3. Keep target as y(t)
    4. Drop rows with NaNs caused by lagging / rolling features
    """
    df = df.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    lagged_feature_cols = []
    for col in FEATURE_COLS:
        lagged_col = f"{col}_lag{LAG_STEPS}"
        df[lagged_col] = df[col].shift(LAG_STEPS)
        lagged_feature_cols.append(lagged_col)

    modelling_df = df[[DATE_COL, *lagged_feature_cols, TARGET_COL]].copy()
    modelling_df = modelling_df.dropna().reset_index(drop=True)

    return modelling_df


def split_dataset(df: pd.DataFrame) -> DatasetSplit:
    """Split prepared dataset chronologically into train / val / test."""
    n_rows = len(df)

    train_end = int(n_rows * TRAIN_RATIO)
    val_end = train_end + int(n_rows * VALIDATION_RATIO)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    lagged_feature_cols = [f"{col}_lag{LAG_STEPS}" for col in FEATURE_COLS]

    X_train = train_df[lagged_feature_cols]
    y_train = train_df[TARGET_COL]

    X_val = val_df[lagged_feature_cols]
    y_val = val_df[TARGET_COL]

    X_test = test_df[lagged_feature_cols]
    y_test = test_df[TARGET_COL]

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )


def load_and_split_data() -> DatasetSplit:
    """Full data preparation pipeline."""
    df = load_bitcoin_data()
    prepared_df = prepare_bitcoin_dataset(df)
    return split_dataset(prepared_df)