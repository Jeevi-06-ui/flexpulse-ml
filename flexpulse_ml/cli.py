"""
Command Line Interface (CLI) for FlexPulse-ML Research Application.

Supports:
- env-check: Inspects Python, libraries, platform, and git integrity.
- validate-data: Validates dataset CSV files against physical and schema constraints.
- import-csv: Ingests 22-column canonical CSV dataset into managed storage.
- register-cable: Enrolls a physical cable in the research registry.
- list-cables: Displays all registered physical cables.
- status: Displays research progress across all 7 phases and hardware status.
- test: Executes automated pytest test suite.
- generate-demo: Generates a strictly-labeled SYNTHETIC_DEMO dataset for testing.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import List, Optional

from flexpulse_ml.acquisition.csv_loader import (
    load_canonical_csv,
    load_dataset_from_csv,
)
from flexpulse_ml.acquisition.experiment_manager import ExperimentManager
from flexpulse_ml.acquisition.raw_storage import RawDataStorage
from flexpulse_ml.acquisition.synthetic_generator import (
    export_canonical_synthetic_csv,
    export_synthetic_dataset_to_csv,
)
from flexpulse_ml.acquisition.validator import DataIntegrityValidator
from flexpulse_ml.utils.paths import (
    DEMO_DATA_DIR,
    PROJECT_ROOT,
    RAW_REAL_DIR,
    RESEARCH_STATUS_FILE,
    ensure_directories_exist,
)


def cmd_env_check() -> int:
    """Validate runtime environment, dependencies, and platform."""
    print("=" * 68)
    print(" FlexPulse-ML :: Environment & Platform Validation")
    print("=" * 68)

    py_ver = sys.version.split()[0]
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Python Version: {py_ver}")
    if (py_major, py_minor) < (3, 11):
        print(f" [!] ERROR: Python 3.11 or later is required. Found: {py_ver}")
        return 1
    print(" [✓] Python version requirement (>= 3.11) satisfied.")

    os_name = platform.system()
    machine = platform.machine()
    print(f"[*] Host OS: {os_name} ({platform.platform()})")
    print(f"[*] Architecture: {machine} (Apple Silicon: {'Yes' if machine == 'arm64' else 'No'})")

    packages = [
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "matplotlib",
        "plotly",
        "streamlit",
        "serial",
        "joblib",
        "pytest",
    ]
    print("\n[*] Scientific Library Availability:")
    all_packages_ok = True
    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "installed")
            print(f"  - {pkg:<12} : {ver}")
        except ImportError:
            print(f"  - {pkg:<12} : MISSING [!]")
            all_packages_ok = False

    print("\n[*] Git Repository:")
    git_dir = PROJECT_ROOT / ".git"
    if git_dir.exists():
        try:
            branch = subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            branch_label = f" (Branch: {branch})" if branch else " (Initial empty repo)"
            print(f"  - Git initialized: Yes{branch_label}")
        except Exception:
            print("  - Git initialized: Yes")
    else:
        print("  - Git initialized: NO [!]")

    ensure_directories_exist()
    print("  - Standard directories: Verified")

    print("=" * 68)
    if all_packages_ok:
        print("[SUCCESS] Environment is fully configured for FlexPulse-ML research.")
        return 0
    else:
        print("[FAILURE] Some dependencies are missing. Run: pip install -r requirements.txt")
        return 1


def cmd_status() -> int:
    """Display research phase progression and real hardware status."""
    print("=" * 72)
    print(" FlexPulse-ML :: Research Status & Scientific Integrity Audit")
    print("=" * 72)

    if not RESEARCH_STATUS_FILE.exists():
        print(f"[ERROR] research_status.json not found at {RESEARCH_STATUS_FILE}")
        return 1

    with open(RESEARCH_STATUS_FILE, "r", encoding="utf-8") as f:
        status_data = json.load(f)

    print(f"Project Name   : {status_data.get('project_name')}")
    print(f"Institution    : {status_data.get('academic_context', {}).get('institution')}")
    print(f"Target Venue   : {status_data.get('academic_context', {}).get('target_venue')}")
    print(f"Submission Due : {status_data.get('academic_context', {}).get('submission_deadline')}")
    print("-" * 72)

    print("\nDevelopment Phase Progression:")
    phases = status_data.get("phases", {})
    for phase_key, phase_info in phases.items():
        name = phase_info.get("name")
        status = phase_info.get("status")
        verif = phase_info.get("verification")
        icon = "[✓]" if status == "COMPLETED" else "[ ]"
        print(f"  {icon} {phase_key.upper():<9} : {status:<22} | {name} ({verif})")

    print("\nHardware Telemetry & Scientific Rigor:")
    hw = status_data.get("hardware_status", {})
    print(f"  - Hardware Connected : {'YES' if hw.get('real_hardware_connected') else 'NO (Offline Mode)'}")
    print(f"  - Real Measurements  : {hw.get('real_data_record_count')} records")
    print(f"  - Real Experiment    : {hw.get('real_experiments_status')}")
    print(f"  - Synthetic Demo Data: {hw.get('synthetic_demo_records_count')} records (Strictly Isolated)")

    exp_mgr = ExperimentManager()
    cables = exp_mgr.list_cables()
    print(f"\nRegistered Physical Cables: {len(cables)}")
    for c in cables:
        print(f"    * {c.cable_id:<18} ({c.manufacturer}, {c.cable_type}, {c.rated_length_m}m)")

    print("=" * 72)
    return 0


def cmd_validate_data(csv_path_str: str, disallow_synthetic: bool = False) -> int:
    """Validate a CSV dataset file against physical and schema constraints."""
    csv_path = Path(csv_path_str).resolve()
    print("=" * 68)
    print(" FlexPulse-ML :: Dataset Validation Audit")
    print(f" Target File: {csv_path}")
    print(f" Disallow Synthetic Data: {disallow_synthetic}")
    print("=" * 68)

    if not csv_path.exists():
        print(f"[ERROR] File does not exist: {csv_path}")
        return 1

    try:
        # First attempt canonical 22-column validation
        valid_records, report = load_canonical_csv(
            csv_path, disallow_synthetic=disallow_synthetic, quarantine_invalid=True
        )
        print("\nCanonical Validation Summary:")
        print(f"  - Total Ingested Records : {report.total_rows}")
        print(f"  - Valid Usable Records   : {report.valid_rows}")
        print(f"  - Invalid (Quarantined)  : {report.invalid_rows}")
        print(f"  - Real Measurements      : {report.real_measurement_rows}")
        print(f"  - Synthetic Demo Records : {report.synthetic_demo_rows}")
        print(f"  - Duplicate Records      : {report.duplicate_rows}")
        print(f"  - Unique Physical Cables : {len(report.unique_cables)} ({', '.join(report.unique_cables)})")
        print(f"  - Unique Sessions        : {len(report.unique_sessions)}")
        if report.missing_columns:
            print(f"  - Missing Canonical Cols : {report.missing_columns}")
        if report.error_summary:
            print("\nRejection Breakdown (Saved to audit log):")
            for err_code, count in report.error_summary.items():
                print(f"    * {err_code:<28} : {count} occurrences")
            if report.quarantine_path:
                print(f"  Quarantine file: {report.quarantine_path}")

        print("=" * 68)
        if report.is_valid:
            print("[RESULT: PASSED] Dataset conforms to 22-column canonical schema.")
            return 0
        else:
            print(f"[RESULT: WARNING/FAILED] Found {report.invalid_rows} invalid or missing records.")
            return 1 if report.invalid_rows > 0 or report.missing_columns else 0

    except Exception as e:
        # Fallback to Phase 1 validator
        print(f"Notice: Falling back to legacy validator ({e})...")
        validator = DataIntegrityValidator(disallow_synthetic=disallow_synthetic)
        valid_records, legacy_report = load_dataset_from_csv(
            csv_path, disallow_synthetic=disallow_synthetic, validator=validator
        )
        if legacy_report.is_acceptable_for_research:
            print("[RESULT: PASSED] Dataset conforms to legacy schema.")
            return 0
        return 1


def cmd_import_csv(csv_path_str: str, disallow_synthetic: bool = False) -> int:
    """Import a validated canonical CSV file into managed research storage."""
    csv_path = Path(csv_path_str).resolve()
    print("=" * 68)
    print(" FlexPulse-ML :: Raw Dataset Ingestion Engine")
    print(f" Ingesting from: {csv_path}")
    print("=" * 68)

    if not csv_path.exists():
        print(f"[ERROR] CSV path does not exist: {csv_path}")
        return 1

    valid_recs, report = load_canonical_csv(csv_path, disallow_synthetic=disallow_synthetic)
    if not report.is_valid or not valid_recs:
        print("[ERROR] Dataset contains validation errors. Aborting ingestion into managed storage.")
        return 1

    storage = RawDataStorage()
    is_synth = report.synthetic_demo_rows > 0
    saved_path = storage.save_canonical_records(valid_recs, is_synthetic=is_synth)
    print(f"[✓] Successfully stored {len(valid_recs)} records in: {saved_path}")
    return 0


def cmd_register_cable(
    cable_id: str,
    manufacturer: str,
    cable_type: str,
    rated_current_a: float,
    rated_length_m: float,
    notes: str,
) -> int:
    """Enroll a physical cable in the laboratory catalog."""
    exp_mgr = ExperimentManager()
    cable = exp_mgr.register_cable(
        cable_id=cable_id,
        manufacturer=manufacturer,
        cable_type=cable_type,
        rated_current_a=rated_current_a,
        rated_length_m=rated_length_m,
        notes=notes,
    )
    print(f"[✓] Physical cable registered: {cable.cable_id} ({cable.manufacturer})")
    return 0


def cmd_list_cables() -> int:
    """Print all enrolled physical cables."""
    exp_mgr = ExperimentManager()
    cables = exp_mgr.list_cables()
    print("=" * 68)
    print(" FlexPulse-ML :: Physical Cable Laboratory Inventory")
    print("=" * 68)
    if not cables:
        print("No cables registered yet. Use 'flexpulse register-cable' to register hardware.")
    else:
        for c in cables:
            print(f"Cable ID      : {c.cable_id}")
            print(f"  Manufacturer : {c.manufacturer}")
            print(f"  Type         : {c.cable_type}")
            print(f"  Rating       : {c.rated_current_a}A, {c.rated_length_m}m")
            print(f"  Enrolled At  : {c.registered_at_utc}")
            if c.notes:
                print(f"  Notes        : {c.notes}")
            print("-" * 68)
    return 0


def cmd_generate_demo(output_str: str, samples: int) -> int:
    """Generate a sample synthetic demo dataset in 22-column canonical format."""
    out_path = Path(output_str).resolve()
    print(f"[*] Generating {samples} synthetic demo samples per trial across 4 test cables...")
    export_canonical_synthetic_csv(out_path, samples_per_trial=samples)
    print(f"[✓] Canonical synthetic demo dataset generated at: {out_path}")
    print("    NOTE: All rows are strictly tagged as 'SYNTHETIC_DEMO' to preserve scientific integrity.")
    return 0


def cmd_test() -> int:
    """Run automated pytest test suite."""
    print("=" * 68)
    print(" FlexPulse-ML :: Running Pytest Test Suite")
    print("=" * 68)
    cmd = [sys.executable, "-m", "pytest", "-v"]
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    return result.returncode


def main(args: Optional[List[str]] = None) -> None:
    """Main CLI entrypoint."""
    ensure_directories_exist()
    parser = argparse.ArgumentParser(
        prog="flexpulse",
        description="FlexPulse-ML Research Application CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # env-check
    subparsers.add_parser("env-check", help="Check system, Python, and scientific dependencies")

    # status
    subparsers.add_parser("status", help="Print research progression and hardware readiness")

    # test
    subparsers.add_parser("test", help="Run automated unit and integration tests")

    # validate-data
    val_parser = subparsers.add_parser("validate-data", help="Validate a dataset CSV file")
    val_parser.add_argument("path", help="Path to the CSV dataset file")
    val_parser.add_argument(
        "--disallow-synthetic",
        action="store_true",
        help="Reject any records labeled SYNTHETIC_DEMO (use for real benchmark data)",
    )

    # import-csv
    import_parser = subparsers.add_parser("import-csv", help="Import a canonical CSV into managed storage")
    import_parser.add_argument("path", help="Path to the CSV file to import")
    import_parser.add_argument("--disallow-synthetic", action="store_true", help="Reject synthetic records")

    # register-cable
    reg_parser = subparsers.add_parser("register-cable", help="Register a physical cable in lab inventory")
    reg_parser.add_argument("--id", required=True, help="Unique cable ID (e.g. CABLE-ANK-001)")
    reg_parser.add_argument("--manufacturer", default="GENERIC", help="Manufacturer brand")
    reg_parser.add_argument("--type", default="USB_C_TO_C", help="Cable type")
    reg_parser.add_argument("--current", type=float, default=3.0, help="Rated current (A)")
    reg_parser.add_argument("--length", type=float, default=1.0, help="Rated length (m)")
    reg_parser.add_argument("--notes", default="", help="Experimental notes")

    # list-cables
    subparsers.add_parser("list-cables", help="List all registered physical cables")

    # generate-demo
    demo_parser = subparsers.add_parser("generate-demo", help="Generate a synthetic demo dataset")
    demo_parser.add_argument(
        "--output",
        default=str(DEMO_DATA_DIR / "canonical_synthetic_demo.csv"),
        help="Target path for generated CSV",
    )
    demo_parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples per trial",
    )

    parsed = parser.parse_args(args)

    if parsed.command == "env-check":
        sys.exit(cmd_env_check())
    elif parsed.command == "status":
        sys.exit(cmd_status())
    elif parsed.command == "test":
        sys.exit(cmd_test())
    elif parsed.command == "validate-data":
        sys.exit(cmd_validate_data(parsed.path, disallow_synthetic=parsed.disallow_synthetic))
    elif parsed.command == "import-csv":
        sys.exit(cmd_import_csv(parsed.path, disallow_synthetic=parsed.disallow_synthetic))
    elif parsed.command == "register-cable":
        sys.exit(
            cmd_register_cable(
                cable_id=parsed.id,
                manufacturer=parsed.manufacturer,
                cable_type=parsed.type,
                rated_current_a=parsed.current,
                rated_length_m=parsed.length,
                notes=parsed.notes,
            )
        )
    elif parsed.command == "list-cables":
        sys.exit(cmd_list_cables())
    elif parsed.command == "generate-demo":
        sys.exit(cmd_generate_demo(parsed.output, parsed.samples))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
