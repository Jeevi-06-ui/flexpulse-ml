"""
Command Line Interface (CLI) for FlexPulse-ML Research Application.

Supports:
- env-check: Inspects Python, libraries, platform, and git integrity.
- validate-data: Validates dataset CSV files against physical and schema constraints.
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
from typing import List

from flexpulse_ml.acquisition.csv_loader import load_dataset_from_csv
from flexpulse_ml.acquisition.synthetic_generator import export_synthetic_dataset_to_csv
from flexpulse_ml.acquisition.validator import DataIntegrityValidator
from flexpulse_ml.utils.paths import (
    DEMO_DATA_DIR,
    PROJECT_ROOT,
    RESEARCH_STATUS_FILE,
    ensure_directories_exist,
)


def cmd_env_check() -> int:
    """Validate runtime environment, dependencies, and platform."""
    print("=" * 68)
    print(" FlexPulse-ML :: Environment & Platform Validation")
    print("=" * 68)

    # 1. Python version check
    py_ver = sys.version.split()[0]
    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    print(f"[*] Python Version: {py_ver}")
    if (py_major, py_minor) < (3, 11):
        print(f" [!] ERROR: Python 3.11 or later is required. Found: {py_ver}")
        return 1
    print(" [✓] Python version requirement (>= 3.11) satisfied.")

    # 2. Operating System & Hardware Platform
    os_name = platform.system()
    machine = platform.machine()
    print(f"[*] Host OS: {os_name} ({platform.platform()})")
    print(f"[*] Architecture: {machine} (Apple Silicon: {'Yes' if machine == 'arm64' else 'No'})")

    # 3. Required Scientific Packages
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

    # 4. Git Repository Integrity
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

    print("=" * 72)
    return 0


def cmd_validate_data(csv_path_str: str, disallow_synthetic: bool = False) -> int:
    """Validate a CSV dataset file against physical and schema constraints."""
    csv_path = Path(csv_path_str).resolve()
    print("=" * 68)
    print(f" FlexPulse-ML :: Dataset Validation Audit")
    print(f" Target File: {csv_path}")
    print(f" Disallow Synthetic Data: {disallow_synthetic}")
    print("=" * 68)

    if not csv_path.exists():
        print(f"[ERROR] File does not exist: {csv_path}")
        return 1

    validator = DataIntegrityValidator(disallow_synthetic=disallow_synthetic)
    try:
        valid_records, report = load_dataset_from_csv(
            csv_path, disallow_synthetic=disallow_synthetic, validator=validator
        )
    except Exception as e:
        print(f"[FATAL ERROR] Ingestion failed: {e}")
        return 1

    print("\nValidation Summary:")
    print(f"  - Total Ingested Records : {report.total_records}")
    print(f"  - Valid Usable Records   : {report.valid_records}")
    print(f"  - Invalid (Quarantined)  : {report.invalid_records}")
    print(f"  - Real Measurements      : {report.real_measurement_records}")
    print(f"  - Synthetic Demo Records : {report.synthetic_demo_records}")
    print(f"  - Duplicate Records      : {report.duplicate_records}")
    print(f"  - Unique Physical Cables : {len(report.unique_cables)} ({', '.join(sorted(report.unique_cables))})")
    print(f"  - Unique Sessions        : {len(report.unique_sessions)}")

    if report.validation_errors_summary:
        print("\nRejection Breakdown (Saved to audit log):")
        for err_code, count in report.validation_errors_summary.items():
            print(f"    * {err_code:<28} : {count} occurrences")
        print(f"  Audit log: {report.audit_log_path}")

    print("=" * 68)
    if report.is_acceptable_for_research:
        print("[RESULT: PASSED] Dataset conforms to physical and scientific schema.")
        return 0
    else:
        print(f"[RESULT: WARNING/FAILED] Found {report.invalid_records} invalid records.")
        return 1 if report.invalid_records > 0 else 0


def cmd_generate_demo(output_str: str, samples: int) -> int:
    """Generate a sample synthetic demo dataset for software testing."""
    out_path = Path(output_str).resolve()
    print(f"[*] Generating {samples} synthetic demo samples per trial across 4 test cables...")
    export_synthetic_dataset_to_csv(out_path, samples_per_trial=samples)
    print(f"[✓] Synthetic demo dataset generated at: {out_path}")
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

    # generate-demo
    demo_parser = subparsers.add_parser("generate-demo", help="Generate a synthetic demo dataset")
    demo_parser.add_argument(
        "--output",
        default=str(DEMO_DATA_DIR / "synthetic_demo_cables.csv"),
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
    elif parsed.command == "generate-demo":
        sys.exit(cmd_generate_demo(parsed.output, parsed.samples))
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
