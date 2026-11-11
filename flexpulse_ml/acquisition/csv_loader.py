"""
CSV Ingestion and Parsing Harness for FlexPulse-ML.

Reads raw CSV files containing electrical measurements, converts them into typed
CableMeasurementRecord objects, and validates them with DataIntegrityValidator.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd

from flexpulse_ml.acquisition.schema import CableMeasurementRecord
from flexpulse_ml.acquisition.validator import DataIntegrityValidator, ValidationReport
from flexpulse_ml.utils.logging import get_logger

logger = get_logger("acquisition.csv_loader")


def load_dataset_from_csv(
    csv_path: Path,
    disallow_synthetic: bool = False,
    validator: Optional[DataIntegrityValidator] = None,
) -> Tuple[List[CableMeasurementRecord], ValidationReport]:
    """Load, parse, and validate experimental records from a CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file to ingest.
    disallow_synthetic : bool
        If True, rejects any records labeled SYNTHETIC_DEMO to protect real benchmark runs.
    validator : DataIntegrityValidator, optional
        Custom validator instance. If None, instantiates a default validator.

    Returns
    -------
    Tuple[List[CableMeasurementRecord], ValidationReport]
        A tuple of valid typed records and the detailed validation report.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    logger.info(f"Reading CSV dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    # Convert DataFrame rows to dictionary records
    raw_dicts = df.to_dict(orient="records")
    parsed_records: List[CableMeasurementRecord] = []

    for d in raw_dicts:
        try:
            rec = CableMeasurementRecord.from_flat_dict(d)
            parsed_records.append(rec)
        except Exception as e:
            logger.warning(f"Failed to parse row into typed schema: {e}. Row: {d}")

    val = validator or DataIntegrityValidator(disallow_synthetic=disallow_synthetic)
    val.reset_duplicate_cache()

    valid_records, report = val.validate_batch(parsed_records, raw_dicts=raw_dicts)
    logger.info(
        f"Ingestion complete: {report.valid_records}/{report.total_records} valid records. "
        f"({report.synthetic_demo_records} synthetic, {report.real_measurement_records} real, "
        f"{report.invalid_records} quarantined)"
    )

    return valid_records, report

