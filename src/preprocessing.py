from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from .config import PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_STATE
except ImportError:
    from config import PROCESSED_DATA_DIR, PROJECT_ROOT, RANDOM_STATE


@dataclass(frozen=True)
class DatasetPaths:
    twitter: Path
    reddit: Path


def _project_root() -> Path:
    return PROJECT_ROOT


def find_raw_files() -> DatasetPaths:
    root = _project_root()
    candidates = [
        root / "data" / "raw",
        root.parent / "dataset",
    ]

    for candidate in candidates:
        twitter = candidate / "Twitter_Data.csv"
        reddit = candidate / "Reddit_Data.csv"
        if twitter.exists() and reddit.exists():
            return DatasetPaths(twitter=twitter, reddit=reddit)

    raise FileNotFoundError(
        "Twitter_Data.csv and Reddit_Data.csv were not found in "
        f"{candidates[0]} or {candidates[1]}."
    )


def load_raw_data() -> pd.DataFrame:
    paths = find_raw_files()

    twitter = pd.read_csv(paths.twitter).rename(columns={"clean_text": "text"})
    twitter["platform"] = "twitter"

    reddit = pd.read_csv(paths.reddit).rename(columns={"clean_comment": "text"})
    reddit["platform"] = "reddit"

    combined = pd.concat(
        [
            twitter[["text", "category", "platform"]],
            reddit[["text", "category", "platform"]],
        ],
        ignore_index=True,
    )
    return combined


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    def count_items(items: list[str] | object) -> int | pd._libs.missing.NAType:
        return len(items) if isinstance(items, list) else pd.NA

    def count_unique_items(items: list[str] | object) -> int | pd._libs.missing.NAType:
        return len(set(items)) if isinstance(items, list) else pd.NA

    result = df.copy()
    text = result["text"].astype("string")
    tokens = text.str.split()

    result["message_length"] = text.str.len().astype("Int64")
    result["word_count"] = tokens.map(count_items).astype("Int64")
    result["unique_word_count"] = tokens.map(count_unique_items).astype("Int64")
    result["avg_word_length"] = result["message_length"] / result["word_count"]
    result["unique_word_ratio"] = result["unique_word_count"] / result["word_count"]
    result["digit_count"] = text.str.count(r"\d").astype("Int64")
    result["digit_ratio"] = result["digit_count"] / result["message_length"]
    result["has_digits"] = (result["digit_count"].fillna(0) > 0).astype(int)
    result["sentiment_missing"] = result["category"].isna().astype(int)

    return result


def clean_dataset(df: pd.DataFrame, outlier_quantile: float = 0.995) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["text"] = cleaned["text"].astype("string")
    cleaned["text_stripped"] = cleaned["text"].str.strip()

    cleaned = cleaned.dropna(subset=["text"]).copy()
    cleaned = cleaned[cleaned["text_stripped"] != ""].copy()
    cleaned = cleaned.drop_duplicates(subset=["text", "platform"]).copy()

    cleaned = add_features(cleaned)

    # Extremely long Reddit comments are valid, but a tiny tail destabilizes metrics.
    threshold = cleaned["message_length"].quantile(outlier_quantile)
    cleaned = cleaned[cleaned["message_length"] <= threshold].copy()

    cleaned = cleaned.drop(columns=["text_stripped"])
    cleaned["category"] = cleaned["category"].astype("float64")
    return cleaned.reset_index(drop=True)


def make_stratify_labels(df: pd.DataFrame, bins: int = 10) -> pd.Series:
    quantile_bins = pd.qcut(df["message_length"], q=bins, duplicates="drop")
    return df["platform"].astype(str) + "__" + quantile_bins.astype(str)


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        random_state=RANDOM_STATE,
        stratify=make_stratify_labels(df),
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=make_stratify_labels(temp_df),
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def save_processed_splits(output_dir: Path | None = None) -> dict[str, Path]:
    output = output_dir or PROCESSED_DATA_DIR
    output.mkdir(parents=True, exist_ok=True)

    cleaned = clean_dataset(load_raw_data())
    train_df, val_df, test_df = split_dataset(cleaned)

    paths = {
        "full": output / "messages_full.csv",
        "train": output / "messages_train.csv",
        "val": output / "messages_val.csv",
        "test": output / "messages_test.csv",
    }

    cleaned.to_csv(paths["full"], index=False)
    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths


if __name__ == "__main__":
    written = save_processed_splits()
    for name, path in written.items():
        print(f"{name}: {path}")
