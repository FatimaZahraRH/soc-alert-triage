#!/usr/bin/env python3
"""Génère le dataset synthétique de flux réseau et le sauvegarde en CSV.

Usage :
    python scripts/generate_dataset.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RAW_DATA_PATH
from src.data.flow_generator import generate_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = generate_dataset()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Dataset généré : %d flux, sauvegardé dans %s", len(df), RAW_DATA_PATH)
    logger.info("Répartition des classes :\n%s", df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
