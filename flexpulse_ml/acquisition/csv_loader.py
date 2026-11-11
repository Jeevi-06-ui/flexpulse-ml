"""
CSV Ingestion and Parsing Harness for FlexPulse-ML.

Reads raw CSV files containing electrical measurements, validates them against the
22-column canonical schema, catches malformed rows, and preserves valid observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from flexpulse_ml.acquisition.raw_storage import RawDataStorage
from flexpulse_ml.acquisition.schema import (
    CANONICAL_COLUMNS,
    CableMeasurementRecord,
    CanonicalMeasurementRecord,
    DatasetCategory,
    FaultClass,
)
from flexpulse_ml.acquisition.validator import DataIntegrityValidator, ValidationReport
from flexpulse_ml.utils.logging import get_logger

logger = get_logger("acquisition.csv_loader")


@dataclass
class CanonicalValidationReport:
    """Detailed audit report for canonical CSV dataset ingestion."""
    total_rows: int = 0
    valid_rows: int = 0
    invalid_rows: int = 0
    real_measurement_rows: int = 0
    synthetic_demo_rows: int = 0
    duplicate_rows: int = 0
    missing_columns: List[str] = field(default_factory=list)
    unique_cables: Set[str] = field(default_factory=set)
    unique_sessions: Set[str] = field(default_factory=set)
    unique_trials: Set[str] = field(default_factory=set)
    error_summary: Dict[str, int] = field(default_factory=dict)
    quarantine_path: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.total_rows > 0 and self.invalid_rows == 0 and len(self.missing_columns) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "invalid_rows": self.invalid_rows,
            "real_measurement_rows": self.real_measurement_rows,
            "synthetic_demo_rows": self.synthetic_demo_rows,
            "duplicate_rows": self.duplicate_rows,
            "missing_columns": self.missing_columns,
            "unique_cables": sorted(list(self.unique_cables)),
            "unique_sessions": sorted(list(self.unique_sessions)),
            "unique_trials": sorted(list(self.unique_trials)),
            "error_summary": self.error_summary,
            "quarantine_path": self.quarantine_path,
        }


def load_canonical_csv(
    csv_path: Path,
    disallow_synthetic: bool = False,
    quarantine_invalid: bool = True,
) -> Tuple[List[CanonicalMeasurementRecord], CanonicalValidationReport]:
    """Load, validate, and parse a CSV file conforming to the 22-column canonical schema.

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file to inspect and ingest.
    disallow_synthetic : bool
        If True, rejects any row where source_type == 'SYNTHETIC_DEMO'.
    quarantine_invalid : bool
        If True, writes invalid rows to the quarantine directory and audit log.

    Returns
    -------
    Tuple[List[CanonicalMeasurementRecord], CanonicalValidationReport]
        The validated records and the comprehensive audit report.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    report = CanonicalValidationReport()
    logger.info(f"Loading canonical dataset from: {csv_path}")

    df = pd.read_csv(csv_path)
    report.total_rows = len(df)

    # 1. Column header check
    existing_cols = set(df.columns)
    missing_cols = [c for c in CANONICAL_COLUMNS if c not in existing_cols]
    if missing_cols:
        report.missing_columns = missing_cols
        for col in missing_cols:
            report.error_summary[f"MISSING_COLUMN_{col}"] = report.error_summary.get(f"MISSING_COLUMN_{col}", 0) + 1
        logger.warning(f"CSV is missing {len(missing_cols)} canonical columns: {missing_cols}")

    valid_records: List[CanonicalMeasurementRecord] = []
    rejected_rows: List[Dict[str, Any]] = []
    rejection_reasons: List[str] = []
    seen_identifiers: Set[Tuple[str, str, str, int]] = set()

    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        errors: List[str] = []

        # Cable ID check
        cable_id = str(row_dict.get("cable_id", "")).strip()
        if not cable_id or cable_id.lower() in ("nan", "none"):
            errors.append("MISSING_CABLE_ID")

        # Session & Trial
        session_id = str(row_dict.get("session_id", "")).strip()
        trial_id = str(row_dict.get("trial_id", "")).strip()
        if not session_id or session_id.lower() in ("nan", "none"):
            errors.append("MISSING_SESSION_ID")
        if not trial_id or trial_id.lower() in ("nan", "none"):
            errors.append("MISSING_TRIAL_ID")

        # Numeric field validation
        try:
            vin = float(row_dict.get("vin_v", 0.0))
            vout = float(row_dict.get("vout_v", 0.0))
            current = float(row_dict.get("current_a", 0.0))
            sample_idx = int(row_dict.get("sample_index", 0))
        except (ValueError, TypeError) as e:
            errors.append(f"NON_NUMERIC_FIELD: {e}")
            vin, vout, current, sample_idx = -1.0, -1.0, -1.0, -1

        if vin < 0.0 or vin > 28.0:
            errors.append(f"OUT_OF_BOUNDS_VIN: {vin}V")
        if vout < 0.0 or vout > 28.0:
            errors.append(f"OUT_OF_BOUNDS_VOUT: {vout}V")
        if vout > (vin + 0.15):
            errors.append(f"PASSIVE_POWER_VIOLATION: vout({vout}V) > vin({vin}V)")
        if current < 0.0 or current > 6.0:
            errors.append(f"OUT_OF_BOUNDS_CURRENT: {current}A")

        # Source type & synthetic isolation
        source_type = str(row_dict.get("source_type", DatasetCategory.REAL_MEASUREMENT.value)).strip()
        if disallow_synthetic and source_type == DatasetCategory.SYNTHETIC_DEMO.value:
            errors.append("SYNTHETIC_CONTAMINATION")

        # Duplicate check: (cable_id, session_id, trial_id, sample_index)
        sig = (cable_id, session_id, trial_id, sample_idx)
        if sig in seen_identifiers:
            errors.append("DUPLICATE_SAMPLE_INDEX")
            report.duplicate_rows += 1
        else:
            seen_identifiers.add(sig)

        if errors:
            report.invalid_rows += 1
            rejected_rows.append(row_dict)
            rejection_reasons.extend(errors)
            for err in errors:
                err_key = err.split(":")[0]
                report.error_summary[err_key] = report.error_summary.get(err_key, 0) + 1
        else:
            report.valid_rows += 1
            report.unique_cables.add(cable_id)
            report.unique_sessions.add(session_id)
            report.unique_trials.add(trial_id)
            if source_type == DatasetCategory.SYNTHETIC_DEMO.value:
                report.synthetic_demo_rows += 1
            else:
                report.real_measurement_rows += 1

            record = CanonicalMeasurementRecord.from_dict(row_dict)
            valid_records.append(record)

    # Quarantine invalid records if requested
    if rejected_rows and quarantine_invalid:
        storage = RawDataStorage()
        sid = valid_records[0].session_id if valid_records else "CORRUPT_BATCH"
        tid = valid_records[0].trial_id if valid_records else "CORRUPT_BATCH"
        q_path = storage.quarantine_records(
            raw_rows=rejected_rows,
            errors=list(set(rejection_reasons)),
            session_id=sid,
            trial_id=tid,
        )
        report.quarantine_path = str(q_path)

    logger.info(
        f"Canonical ingestion finished: {report.valid_rows}/{report.total_rows} valid records "
        f"({report.synthetic_demo_rows} synthetic, {report.real_measurement_rows} real, "
        f"{report.invalid_rows} invalid/quarantined)"
    )

    return valid_records, report


def load_dataset_from_csv(
    csv_path: Path,
    disallow_synthetic: bool = False,
    validator: Optional[DataIntegrityValidator] = None,
) -> Tuple[List[CableMeasurementRecord], ValidationReport]:
    """Phase 1 backwards-compatible loader.

    Reads raw CSV files containing electrical measurements, converts them into typed
    CableMeasurementRecord objects, and validates them with DataIntegrityValidator.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

    logger.info(f"Reading CSV dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

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
