from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from .config import PROCESSED_DATA_DIR, PROJECT_ROOT, REPORT_IMAGES_DIR
    from .preprocessing import clean_dataset, load_raw_data, split_dataset
except ImportError:
    from config import PROCESSED_DATA_DIR, PROJECT_ROOT, REPORT_IMAGES_DIR
    from preprocessing import clean_dataset, load_raw_data, split_dataset


def _project_root() -> Path:
    return PROJECT_ROOT


def _prepare_dirs() -> tuple[Path, Path]:
    images_dir = REPORT_IMAGES_DIR
    processed_dir = PROCESSED_DATA_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, processed_dir


def build_summary(raw_df: pd.DataFrame, clean_df: pd.DataFrame) -> dict:
    train_df, val_df, test_df = split_dataset(clean_df)
    raw_platform_counts = raw_df["platform"].value_counts().to_dict()
    clean_platform_counts = clean_df["platform"].value_counts().to_dict()
    blank_mask = (raw_df["text"].astype("string").str.strip() == "").fillna(False)
    full_platform_ratio = clean_df["platform"].value_counts(normalize=True).round(4).to_dict()
    train_platform_ratio = train_df["platform"].value_counts(normalize=True).round(4).to_dict()
    val_platform_ratio = val_df["platform"].value_counts(normalize=True).round(4).to_dict()
    test_platform_ratio = test_df["platform"].value_counts(normalize=True).round(4).to_dict()
    correlations = {}
    for column in ["word_count", "unique_word_ratio", "digit_count", "category"]:
        feature_values = pd.to_numeric(clean_df[column], errors="coerce")
        correlations[column] = float(clean_df["message_length"].corr(feature_values))

    summary = {
        "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
        "clean_shape": [int(clean_df.shape[0]), int(clean_df.shape[1])],
        "raw_platform_counts": {key: int(value) for key, value in raw_platform_counts.items()},
        "clean_platform_counts": {key: int(value) for key, value in clean_platform_counts.items()},
        "missing_text_rows": int(raw_df["text"].isna().sum()),
        "blank_text_rows": int(blank_mask.sum()),
        "duplicate_text_platform_rows": int(raw_df.duplicated(subset=["text", "platform"]).sum()),
        "feature_count_before_engineering": 3,
        "feature_count_after_engineering": int(clean_df.shape[1] - 1),
        "target_summary": {
            key: float(value)
            for key, value in clean_df["message_length"]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
            .to_dict()
            .items()
        },
        "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
        "val_shape": [int(val_df.shape[0]), int(val_df.shape[1])],
        "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
        "platform_ratio_full": {key: float(value) for key, value in full_platform_ratio.items()},
        "platform_ratio_train": {key: float(value) for key, value in train_platform_ratio.items()},
        "platform_ratio_val": {key: float(value) for key, value in val_platform_ratio.items()},
        "platform_ratio_test": {key: float(value) for key, value in test_platform_ratio.items()},
        "target_mean_full": float(clean_df["message_length"].mean()),
        "target_mean_train": float(train_df["message_length"].mean()),
        "target_mean_val": float(val_df["message_length"].mean()),
        "target_mean_test": float(test_df["message_length"].mean()),
        "correlations": correlations,
    }
    return summary


def save_summary(summary: dict, processed_dir: Path) -> Path:
    path = processed_dir / "eda_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def plot_length_distribution(clean_df: pd.DataFrame, images_dir: Path) -> Path:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=clean_df, x="message_length", hue="platform", bins=50, kde=True, ax=ax)
    ax.set_title("Message length distribution by platform")
    ax.set_xlabel("Length in characters")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path = images_dir / "length_distribution.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_platform_boxplot(clean_df: pd.DataFrame, images_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=clean_df, x="platform", y="message_length", ax=ax)
    ax.set_title("Message length by platform")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Length in characters")
    ax.set_ylim(0, clean_df["message_length"].quantile(0.99))
    fig.tight_layout()
    path = images_dir / "platform_boxplot.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_word_count_relation(clean_df: pd.DataFrame, images_dir: Path) -> Path:
    sample = clean_df.sample(min(5000, len(clean_df)), random_state=42)
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=sample,
        x="word_count",
        y="message_length",
        hue="platform",
        alpha=0.5,
        ax=ax,
    )
    ax.set_title("Word count vs message length")
    ax.set_xlabel("Word count")
    ax.set_ylabel("Length in characters")
    fig.tight_layout()
    path = images_dir / "word_count_relation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_correlation_heatmap(clean_df: pd.DataFrame, images_dir: Path) -> Path:
    numeric = clean_df[
        [
            "message_length",
            "word_count",
            "unique_word_count",
            "avg_word_length",
            "unique_word_ratio",
            "digit_count",
            "category",
        ]
    ].copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric.corr(numeric_only=True), annot=True, fmt=".2f", cmap="Blues", ax=ax)
    ax.set_title("Correlation heatmap")
    fig.tight_layout()
    path = images_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def main() -> None:
    images_dir, processed_dir = _prepare_dirs()
    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df)

    summary = build_summary(raw_df=raw_df, clean_df=clean_df)
    summary_path = save_summary(summary=summary, processed_dir=processed_dir)

    written_paths = [
        plot_length_distribution(clean_df, images_dir),
        plot_platform_boxplot(clean_df, images_dir),
        plot_word_count_relation(clean_df, images_dir),
        plot_correlation_heatmap(clean_df, images_dir),
        summary_path,
    ]

    for path in written_paths:
        print(path)


if __name__ == "__main__":
    main()
