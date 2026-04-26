from __future__ import annotations

from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_IMAGES_DIR = REPORT_DIR / "images"
