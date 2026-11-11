"""
Dataset Validation & Integrity Audit Subsystem for FlexPulse-ML.

Enforces strict physical constraints, provenance guarantees, duplicate checks,
and synthetic-demo isolation. Quarantines invalid records into an audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from flexpulse_ml.acquisition.schema import (
    CableMeasurementRecord,
    DatasetCategory,
    FaultClass,
    LabelSource,
)
from flexpulse_ml.utils.paths import AUDIT_LOG_FILE


@dataclass
class ValidationReport:
    """Summary of validation outcomes for a dataset or stream."""
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    synthetic_demo_records: int = 0
    real_measurement_records: int = 0
    duplicate_records: int = 0
    unique_cables: Set[str] = field(default_factory=set)
    unique_sessions: Set[str] = field(default_factory=set)
    validation_errors_summary: Dict[str, int] = field(default_factory=dict)
    audit_log_path: Optional[str] = None

    @property
    def is_acceptable_for_research(self) -> bool:
        """True if there are records, zero critical schema corruptions, and at least some valid rows."""
        return self.total_records > 0 and self.invalid_records == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_records": self.total_records,
            "valid_records": self.valid_records,
            "invalid_records": self.invalid_records,
            "synthetic_demo_records": self.synthetic_demo_records,
            "real_measurement_records": self.real_measurement_records,
            "duplicate_records": self.duplicate_records,
            "unique_cables_count": len(self.unique_cables),
            "unique_cables_list": sorted(list(self.unique_cables)),
            "unique_sessions_count": len(self.unique_sessions),
            "validation_errors_summary": self.validation_errors_summary,
            "audit_log_path": self.audit_log_path,
        }


class DataIntegrityValidator:
    """Validates experimental USB-C records against physical and scientific constraints."""

    # Physical limits for safe 5V USB-C / PD power-path experiments
    MIN_VOLTAGE_V = 0.0
    MAX_VOLTAGE_V = 28.0       # Standard USB PD Extended Power Range upper bound
    MIN_CURRENT_A = 0.0
    MAX_CURRENT_A = 6.0        # 5A standard with 20% spike margin
    MAX_VOLTAGE_DROP_TOLERANCE_V = 0.15 # Allow up to 150mV reverse spike due to measurement noise

    def __init__(self, audit_file_path: Optional[Path] = None, disallow_synthetic: bool = False):
        self.audit_file_path = audit_file_path or AUDIT_LOG_FILE
        self.disallow_synthetic = disallow_synthetic
        self.seen_signatures: Set[Tuple[str, str, str, int]] = set()

    def reset_duplicate_cache(self) -> None:
        """Clear memory cache of seen record signatures."""
        self.seen_signatures.clear()

    def validate_record(self, record: CableMeasurementRecord) -> List[str]:
        """Validate a single CableMeasurementRecord.

        Returns an empty list if valid, or a list of specific error descriptions.
        """
        errors: List[str] = []

        # 1. Physical Cable ID Check
        if not record.cable.cable_id or not record.cable.cable_id.strip():
            errors.append("MISSING_CABLE_ID: Cable identifier cannot be empty or whitespace.")

        # 2. Session and Trial Identity
        if not record.session.session_id or not record.session.session_id.strip():
            errors.append("MISSING_SESSION_ID: Session identifier cannot be empty.")
        if not record.session.trial_id or not record.session.trial_id.strip():
            errors.append("MISSING_TRIAL_ID: Trial identifier cannot be empty.")

        # 3. Numeric Physical Voltage Checks
        v_src = record.observation.v_source_v
        v_snk = record.observation.v_sink_v
        i_ld = record.observation.current_a

        if v_src < self.MIN_VOLTAGE_V or v_src > self.MAX_VOLTAGE_V:
            errors.append(f"INVALID_V_SOURCE: Voltage {v_src:.3f}V is outside physical bound [0.0, 28.0]V.")

        if v_snk < self.MIN_VOLTAGE_V or v_snk > self.MAX_VOLTAGE_V:
            errors.append(f"INVALID_V_SINK: Voltage {v_snk:.3f}V is outside physical bound [0.0, 28.0]V.")

        # In a passive cable, sink voltage cannot exceed source voltage significantly
        if v_snk > (v_src + self.MAX_VOLTAGE_DROP_TOLERANCE_V):
            errors.append(
                f"PASSIVE_POWER_VIOLATION: Sink voltage ({v_snk:.3f}V) exceeds source ({v_src:.3f}V) "
                f"beyond noise margin {self.MAX_VOLTAGE_DROP_TOLERANCE_V}V."
            )

        # 4. Numeric Current Checks
        if i_ld < self.MIN_CURRENT_A or i_ld > self.MAX_CURRENT_A:
            errors.append(f"INVALID_CURRENT: Current {i_ld:.3f}A is outside physical bound [0.0, 6.0]A.")

        # 5. Timestamp Checks
        if record.observation.timestamp_rel_ms < 0.0:
            errors.append(f"NEGATIVE_TIMESTAMP: Relative timestamp {record.observation.timestamp_rel_ms}ms cannot be negative.")

        # 6. Unverified Label Integrity
        if record.label_source == LabelSource.UNVERIFIED and record.ground_truth != FaultClass.UNKNOWN:
            # Enforce that unverified measurements are marked as UNKNOWN ground truth
            record.ground_truth = FaultClass.UNKNOWN

        # 7. Synthetic Contamination Check
        if self.disallow_synthetic and record.dataset_category == DatasetCategory.SYNTHETIC_DEMO:
            errors.append("SYNTHETIC_CONTAMINATION: SYNTHETIC_DEMO records are strictly forbidden in real experimental benchmarks.")

        # 8. Duplicate Check
        sig = (
            record.cable.cable_id,
            record.session.session_id,
            record.session.trial_id,
            record.session.step_index,
        )
        if sig in self.seen_signatures:
            errors.append(
                f"DUPLICATE_RECORD: Record with signature {sig} was already observed in this ingestion stream."
            )
        else:
            self.seen_signatures.add(sig)

        return errors

    def quarantine_invalid_record(self, raw_record: Dict[str, Any], errors: List[str]) -> None:
        """Append an invalid record and its rejection reasons to the audit log."""
        self.audit_file_path.parent.mkdir(parents=True, exist_ok=True)
        audit_entry = {
            "timestamp_audit_utc": datetime.now(timezone.utc).isoformat(),
            "errors": errors,
            "raw_record": raw_record,
        }
        with open(self.audit_file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(audit_entry) + "\n")

    def validate_batch(
        self, records: List[CableMeasurementRecord], raw_dicts: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[List[CableMeasurementRecord], ValidationReport]:
        """Validate a list of records, returning valid records and a summary report."""
        valid_records: List[CableMeasurementRecord] = []
        report = ValidationReport()
        report.audit_log_path = str(self.audit_file_path)

        for i, rec in enumerate(records):
            report.total_records += 1
            raw_dict = raw_dicts[i] if (raw_dicts and i < len(raw_dicts)) else rec.to_flat_dict()

            errors = self.validate_record(rec)
            if errors:
                report.invalid_records += 1
                for err in errors:
                    err_code = err.split(":")[0]
                    report.validation_errors_summary[err_code] = report.validation_errors_summary.get(err_code, 0) + 1
                    if err_code == "DUPLICATE_RECORD":
                        report.duplicate_records += 1
                self.quarantine_invalid_record(raw_dict, errors)
            else:
                report.valid_records += 1
                report.unique_cables.add(rec.cable.cable_id)
                report.unique_sessions.add(rec.session.session_id)
                if rec.dataset_category == DatasetCategory.SYNTHETIC_DEMO:
                    report.synthetic_demo_records += 1
                else:
                    report.real_measurement_records += 1
                valid_records.append(rec)

        return valid_records, report
