"""
Experiment Management Subsystem for FlexPulse-ML.

Provides scientific workflows for:
- Registering physical cables with hardware specifications
- Managing session lifecycle (start, register trials, record conditions, complete)
- Independent provenance tracking of ground-truth condition labels
- Storing immutable trial manifests and audit metadata

SCIENTIFIC INTEGRITY RULE:
Ground-truth condition labels MUST be explicitly declared in trial experimental
manifests prior to model evaluation, and NEVER inferred from ML model outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from flexpulse_ml.acquisition.schema import (
    DatasetCategory,
    FaultClass,
    FaultOrigin,
    LabelSource,
    PhysicalCableIdentity,
)
from flexpulse_ml.utils.logging import get_logger
from flexpulse_ml.utils.paths import (
    CABLE_REGISTRY_FILE,
    SESSIONS_METADATA_DIR,
    ensure_directories_exist,
)

logger = get_logger("acquisition.experiment_manager")


@dataclass
class RegisteredCable:
    """Registered physical cable in the lab inventory."""
    cable_id: str
    manufacturer: str
    cable_type: str
    rated_current_a: float
    rated_length_m: float
    has_emask_chip: bool
    registered_at_utc: str
    notes: str = ""

    def to_identity(self) -> PhysicalCableIdentity:
        return PhysicalCableIdentity(
            cable_id=self.cable_id,
            manufacturer=self.manufacturer,
            cable_type=self.cable_type,
            rated_current_a=self.rated_current_a,
            rated_length_m=self.rated_length_m,
            has_emask_chip=self.has_emask_chip,
        )


@dataclass
class TrialManifest:
    """Independent record of a specific experimental trial and declared ground truth."""
    trial_id: str
    cable_id: str
    fixture_id: str
    action_id: str
    motion_region: str
    nominal_voltage_v: float
    target_current_a: float
    ambient_temp_c: float
    ground_truth_label: str
    ground_truth_source: str
    fault_origin: str
    fault_condition: str
    dataset_category: str
    started_at_utc: str
    completed_at_utc: Optional[str] = None
    observation_count: int = 0
    notes: str = ""


@dataclass
class SessionManifest:
    """Master manifest capturing an entire experimental session."""
    session_id: str
    operator_id: str
    fixture_id: str
    description: str
    started_at_utc: str
    completed_at_utc: Optional[str] = None
    status: str = "ACTIVE" # ACTIVE, COMPLETED, ABORTED
    trials: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    total_samples: int = 0


class ExperimentManager:
    """Manages physical cable registries, trial manifests, and session provenance."""

    def __init__(
        self,
        registry_file: Optional[Path] = None,
        sessions_dir: Optional[Path] = None,
    ):
        ensure_directories_exist()
        self.registry_file = registry_file or CABLE_REGISTRY_FILE
        self.sessions_dir = sessions_dir or SESSIONS_METADATA_DIR
        self.cables: Dict[str, RegisteredCable] = self._load_cable_registry()
        self.active_sessions: Dict[str, SessionManifest] = {}

    def _load_cable_registry(self) -> Dict[str, RegisteredCable]:
        """Load physical cable registry from disk."""
        if not self.registry_file.exists():
            return {}
        try:
            with open(self.registry_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {cid: RegisteredCable(**info) for cid, info in data.items()}
        except Exception as e:
            logger.warning(f"Could not load cable registry: {e}")
            return {}

    def _save_cable_registry(self) -> None:
        """Persist physical cable registry to disk."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {cid: asdict(cable) for cid, cable in self.cables.items()}
        with open(self.registry_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def register_cable(
        self,
        cable_id: str,
        manufacturer: str = "GENERIC",
        cable_type: str = "USB_C_TO_C",
        rated_current_a: float = 3.0,
        rated_length_m: float = 1.0,
        has_emask_chip: bool = False,
        notes: str = "",
    ) -> RegisteredCable:
        """Register a new physical cable in the laboratory catalog."""
        cid = cable_id.strip()
        if not cid:
            raise ValueError("cable_id cannot be empty.")

        reg = RegisteredCable(
            cable_id=cid,
            manufacturer=manufacturer,
            cable_type=cable_type,
            rated_current_a=rated_current_a,
            rated_length_m=rated_length_m,
            has_emask_chip=has_emask_chip,
            registered_at_utc=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )
        self.cables[cid] = reg
        self._save_cable_registry()
        logger.info(f"Registered physical cable: {cid} ({manufacturer}, {rated_length_m}m)")
        return reg

    def get_cable(self, cable_id: str) -> Optional[RegisteredCable]:
        """Retrieve registered cable by ID."""
        return self.cables.get(cable_id.strip())

    def list_cables(self) -> List[RegisteredCable]:
        """Return all registered physical cables."""
        return list(self.cables.values())

    def start_session(
        self,
        session_id: str,
        operator_id: str = "LAB_OPERATOR",
        fixture_id: str = "FIXTURE_ESP32_01",
        description: str = "",
    ) -> SessionManifest:
        """Initialize a new experimental recording session."""
        sid = session_id.strip()
        if not sid:
            raise ValueError("session_id cannot be empty.")

        session_manifest_path = self.sessions_dir / f"{sid}.json"
        if session_manifest_path.exists():
            logger.warning(f"Session manifest {sid} already exists on disk. Resuming session.")
            with open(session_manifest_path, "r", encoding="utf-8") as f:
                d = json.load(f)
            manifest = SessionManifest(**d)
            self.active_sessions[sid] = manifest
            return manifest

        manifest = SessionManifest(
            session_id=sid,
            operator_id=operator_id,
            fixture_id=fixture_id,
            description=description,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            status="ACTIVE",
        )
        self.active_sessions[sid] = manifest
        self._save_session_manifest(manifest)
        logger.info(f"Started experimental session: {sid} (Operator: {operator_id})")
        return manifest

    def register_trial(
        self,
        session_id: str,
        trial_id: str,
        cable_id: str,
        action_id: str = "STATIC_HOLD",
        motion_region: str = "STATIC",
        nominal_voltage_v: float = 5.0,
        target_current_a: float = 1.0,
        ambient_temp_c: float = 25.0,
        ground_truth_label: str = FaultClass.INTACT.value,
        ground_truth_source: str = LabelSource.HUMAN_VERIFIED.value,
        fault_origin: str = FaultOrigin.GOLDEN_REFERENCE.value,
        fault_condition: str = "NONE",
        dataset_category: str = DatasetCategory.REAL_MEASUREMENT.value,
        notes: str = "",
    ) -> TrialManifest:
        """Register an experimental trial and record its independent ground truth.

        Scientific Rule:
        Labels are declared directly in this manifest by human lab inspection or
        controlled fault injection, completely uncoupled from model outputs.
        """
        sid = session_id.strip()
        tid = trial_id.strip()
        cid = cable_id.strip()

        if sid not in self.active_sessions:
            self.start_session(session_id=sid)

        session = self.active_sessions[sid]
        trial = TrialManifest(
            trial_id=tid,
            cable_id=cid,
            fixture_id=session.fixture_id,
            action_id=action_id,
            motion_region=motion_region,
            nominal_voltage_v=nominal_voltage_v,
            target_current_a=target_current_a,
            ambient_temp_c=ambient_temp_c,
            ground_truth_label=ground_truth_label,
            ground_truth_source=ground_truth_source,
            fault_origin=fault_origin,
            fault_condition=fault_condition,
            dataset_category=dataset_category,
            started_at_utc=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

        session.trials[tid] = asdict(trial)
        self._save_session_manifest(session)
        logger.info(f"Registered trial: {tid} in session {sid} for cable {cid} (Label: {ground_truth_label})")
        return trial

    def update_trial_observation_count(self, session_id: str, trial_id: str, count: int) -> None:
        """Update sample counts in active trial."""
        if session_id in self.active_sessions and trial_id in self.active_sessions[session_id].trials:
            trial_dict = self.active_sessions[session_id].trials[trial_id]
            trial_dict["observation_count"] = count
            trial_dict["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            self._save_session_manifest(self.active_sessions[session_id])

    def complete_session(self, session_id: str) -> SessionManifest:
        """Finalize an experimental session and write the finalized manifest to disk."""
        sid = session_id.strip()
        if sid not in self.active_sessions:
            session_manifest_path = self.sessions_dir / f"{sid}.json"
            if session_manifest_path.exists():
                with open(session_manifest_path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                manifest = SessionManifest(**d)
            else:
                raise ValueError(f"No active session found with id: {sid}")
        else:
            manifest = self.active_sessions[sid]

        manifest.status = "COMPLETED"
        manifest.completed_at_utc = datetime.now(timezone.utc).isoformat()
        manifest.total_samples = sum(t.get("observation_count", 0) for t in manifest.trials.values())
        self._save_session_manifest(manifest)
        logger.info(f"Completed session: {sid} with {len(manifest.trials)} trials and {manifest.total_samples} samples.")
        return manifest

    def _save_session_manifest(self, manifest: SessionManifest) -> None:
        """Write session manifest JSON file to data/metadata/sessions/."""
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        path = self.sessions_dir / f"{manifest.session_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(manifest), f, indent=2)

