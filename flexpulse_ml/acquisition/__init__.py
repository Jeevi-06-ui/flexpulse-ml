"""
Acquisition module for FlexPulse-ML.
Provides schemas, validators, CSV loaders, synthetic generators, and hardware interfaces.
"""

from flexpulse_ml.acquisition.schema import (
    AcquisitionQuality,
    CANONICAL_COLUMNS,
    CableMeasurementRecord,
    CanonicalMeasurementRecord,
    DataProvenance,
    DatasetCategory,
    ElectricalObservation,
    FaultClass,
    FaultOrigin,
    InstrumentCalibration,
    LabelSource,
    MeasurementQuality,
    PhysicalCableIdentity,
    TestConditions,
    TrialSessionIdentity,
)
from flexpulse_ml.acquisition.validator import (
    DataIntegrityValidator,
    ValidationReport,
)
from flexpulse_ml.acquisition.synthetic_generator import (
    export_canonical_synthetic_csv,
    export_synthetic_dataset_to_csv,
    generate_canonical_synthetic_dataset,
    generate_synthetic_cable_dataset,
)
from flexpulse_ml.acquisition.csv_loader import (
    CanonicalValidationReport,
    load_canonical_csv,
    load_dataset_from_csv,
)
from flexpulse_ml.acquisition.hardware_interface import (
    BaseAcquisitionSource,
    ESP32SerialAdapter,
    HardwareNotConnectedError,
    HardwareSafetyViolationError,
)
from flexpulse_ml.acquisition.serial_adapter import (
    ESP32JSONSerialAdapter,
    PacketValidationResult,
    SerialStreamStatistics,
    SimulatedESP32Stream,
)
from flexpulse_ml.acquisition.experiment_manager import (
    ExperimentManager,
    RegisteredCable,
    SessionManifest,
    TrialManifest,
)
from flexpulse_ml.acquisition.raw_storage import (
    RawDataStorage,
)

__all__ = [
    "AcquisitionQuality",
    "BaseAcquisitionSource",
    "CANONICAL_COLUMNS",
    "CableMeasurementRecord",
    "CanonicalMeasurementRecord",
    "CanonicalValidationReport",
    "DataIntegrityValidator",
    "DataProvenance",
    "DatasetCategory",
    "ESP32JSONSerialAdapter",
    "ESP32SerialAdapter",
    "ElectricalObservation",
    "ExperimentManager",
    "FaultClass",
    "FaultOrigin",
    "HardwareNotConnectedError",
    "HardwareSafetyViolationError",
    "InstrumentCalibration",
    "LabelSource",
    "MeasurementQuality",
    "PacketValidationResult",
    "PhysicalCableIdentity",
    "RawDataStorage",
    "RegisteredCable",
    "SerialStreamStatistics",
    "SessionManifest",
    "SimulatedESP32Stream",
    "TestConditions",
    "TrialManifest",
    "TrialSessionIdentity",
    "ValidationReport",
    "export_canonical_synthetic_csv",
    "export_synthetic_dataset_to_csv",
    "generate_canonical_synthetic_dataset",
    "generate_synthetic_cable_dataset",
    "load_canonical_csv",
    "load_dataset_from_csv",
]
