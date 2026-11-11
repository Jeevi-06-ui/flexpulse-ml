"""
Raw Data Storage and Provenance Subsystem for FlexPulse-ML.

Saves original, untouched experimental observations into physically segregated directories:
- data/raw/real/            : Verified experimental measurements from physical fixtures.
- data/raw/synthetic_demo/ : Strictly isolated synthetic data for testing.
- data/metadata/quarantine/: Rejected or corrupt records with audit logs.
- data/metadata/manifests/ : Cryptographic SHA-256 manifests for trial verification.

SCIENTIFIC INTEGRITY RULE:
Original observations are saved WITHOUT modifying raw floating-point values.
Every record retains complete traceability to cable_id, session_id, and trial_id.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from flexpulse_ml.acquisition.schema import (
    CANONICAL_COLUMNS,
    CanonicalMeasurementRecord,
    DatasetCategory,
)
from flexpulse_ml.utils.logging import get_logger
from flexpulse_ml.utils.paths import (
    AUDIT_LOG_FILE,
    MANIFESTS_METADATA_DIR,
    QUARANTINE_DIR,
    RAW_REAL_DIR,
    RAW_SYNTHETIC_DIR,
    ensure_directories_exist,
)

logger = get_logger("acquisition.raw_storage")


class RawDataStorage:
    """Manages immutable raw experimental persistence with provenance checksums."""

    def __init__(
        self,
        real_dir: Optional[Path] = None,
        synthetic_dir: Optional[Path] = None,
        quarantine_dir: Optional[Path] = None,
        manifests_dir: Optional[Path] = None,
    ):
        ensure_directories_exist()
        self.real_dir = real_dir or RAW_REAL_DIR
        self.synthetic_dir = synthetic_dir or RAW_SYNTHETIC_DIR
        self.quarantine_dir = quarantine_dir or QUARANTINE_DIR
        self.manifests_dir = manifests_dir or MANIFESTS_METADATA_DIR

    def compute_file_sha256(self, file_path: Path) -> str:
        """Compute cryptographic hash of a stored file for audit verification."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_canonical_records(
        self,
        records: List[CanonicalMeasurementRecord],
        is_synthetic: bool = False,
        file_prefix: Optional[str] = None,
    ) -> Path:
        """Persist a list of canonical observations to their segregated directory.

        Parameters
        ----------
        records : List[CanonicalMeasurementRecord]
            The list of typed records to store.
        is_synthetic : bool
            If True, saves into data/raw/synthetic_demo/ and enforces SYNTHETIC_DEMO tagging.
            If False, saves into data/raw/real/.

        Returns
        -------
        Path
            Path to the newly saved CSV file.
        """
        if not records:
            raise ValueError("Cannot save empty records list.")

        first = records[0]
        cable_id = first.cable_id
        session_id = first.session_id
        trial_id = first.trial_id

        # Determine target directory
        target_dir = self.synthetic_dir if is_synthetic else self.real_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Enforce category tag
        expected_cat = DatasetCategory.SYNTHETIC_DEMO.value if is_synthetic else DatasetCategory.REAL_MEASUREMENT.value
        for rec in records:
            if is_synthetic:
                rec.source_type = expected_cat
            elif rec.source_type == DatasetCategory.SYNTHETIC_DEMO.value:
                raise ValueError(
                    f"Integrity Violation: Attempted to store SYNTHETIC_DEMO row in real dataset: {rec}"
                )

        prefix = file_prefix or f"{cable_id}_{session_id}_{trial_id}"
        csv_path = target_dir / f"{prefix}_raw.csv"

        # Build DataFrame with exact 22 columns preserving raw precision
        rows = [rec.to_dict() for rec in records]
        df = pd.DataFrame(rows, columns=list(CANONICAL_COLUMNS))
        df.to_csv(csv_path, index=False)

        # Generate and save cryptographic manifest
        checksum = self.compute_file_sha256(csv_path)
        manifest = {
            "file_name": csv_path.name,
            "relative_path": str(csv_path.relative_to(target_dir.parent.parent)),
            "sha256_checksum": checksum,
            "record_count": len(records),
            "cable_id": cable_id,
            "session_id": session_id,
            "trial_id": trial_id,
            "source_type": expected_cat,
            "stored_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = self.manifests_dir / f"{prefix}_manifest.json"
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Saved raw experimental observations: {csv_path} ({len(records)} rows, SHA256: {checksum[:8]}...)"
        )
        return csv_path

    def quarantine_records(
        self,
        raw_rows: List[Dict[str, Any]],
        errors: List[str],
        session_id: str = "UNKNOWN_SESSION",
        trial_id: str = "UNKNOWN_TRIAL",
    ) -> Path:
        """Store rejected or corrupt records into quarantine directory and append to audit log."""
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        quarantine_csv = self.quarantine_dir / f"quarantine_{session_id}_{trial_id}_{timestamp_str}.csv"

        df = pd.DataFrame(raw_rows)
        df.to_csv(quarantine_csv, index=False)

        # Append to audit JSONL
        AUDIT_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        audit_entry = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "quarantined_file": str(quarantine_csv),
            "record_count": len(raw_rows),
            "errors": errors,
        }
        with open(AUDIT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

        logger.warning(
            f"Quarantined {len(raw_rows)} invalid records to {quarantine_csv}. Reason: {errors}"
        )
        return quarantine_csv
