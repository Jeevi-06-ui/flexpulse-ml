"""
Acquisition module for FlexPulse-ML.
Provides schemas, validators, CSV loaders, synthetic generators, and hardware interfaces.
"""

from flexpulse_ml.acquisition.schema import (
    AcquisitionQuality,
    CableMeasurementRecord,
    DataProvenance,
    DatasetCategory,
    ElectricalObservation,
    FaultClass,
    FaultOrigin,
    InstrumentCalibration,
    LabelSource,
    PhysicalCableIdentity,
    TestConditions,
    TrialSessionIdentity,
)
from flexpulse_ml.acquisition.validator import (
    DataIntegrityValidator,
    ValidationReport,
)
from flexpulse_ml.acquisition.synthetic_generator import (
    export_synthetic_dataset_to_csv,
    generate_synthetic_cable_dataset,
)
from flexpulse_ml.acquisition.csv_loader import (
    load_dataset_from_csv,
)
from flexpulse_ml.acquisition.hardware_interface import (
    BaseAcquisitionSource,
    ESP32SerialAdapter,
    HardwareNotConnectedError,
    HardwareSafetyViolationError,
)

__all__ = [
    "AcquisitionQuality",
    "BaseAcquisitionSource",
    "CableMeasurementRecord",
    "DataIntegrityValidator",
    "DataProvenance",
    "DatasetCategory",
    "ElectricalObservation",
    "ESP32SerialAdapter",
    "FaultClass",
    "FaultOrigin",
    "HardwareNotConnectedError",
    "HardwareSafetyViolationError",
    "InstrumentCalibration",
    "LabelSource",
    "PhysicalCableIdentity",
    "TestConditions",
    "TrialSessionIdentity",
    "ValidationReport",
    "export_synthetic_dataset_to_csv",
    "generate_synthetic_cable_dataset",
    "load_dataset_from_csv",
]

