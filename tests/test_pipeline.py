from src.preprocessing import clean_dataset, load_raw_data, split_dataset


def test_clean_dataset_creates_target_and_features() -> None:
    cleaned = clean_dataset(load_raw_data())

    expected_columns = {
        "text",
        "category",
        "platform",
        "message_length",
        "word_count",
        "unique_word_count",
        "avg_word_length",
        "unique_word_ratio",
        "digit_count",
        "digit_ratio",
        "has_digits",
        "sentiment_missing",
    }

    assert expected_columns.issubset(cleaned.columns)
    assert cleaned["message_length"].isna().sum() == 0
    assert (cleaned["message_length"] > 0).all()


def test_split_dataset_preserves_row_count() -> None:
    cleaned = clean_dataset(load_raw_data())
    train_df, val_df, test_df = split_dataset(cleaned)

    assert len(train_df) + len(val_df) + len(test_df) == len(cleaned)
    assert len(train_df) > len(val_df) > 0
    assert len(test_df) > 0
