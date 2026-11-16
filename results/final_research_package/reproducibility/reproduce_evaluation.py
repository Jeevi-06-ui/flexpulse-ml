"""
One-click reproduction script for FlexPulse-ML Phase 7 Evidence Package.
"""

from flexpulse_ml.final_evidence.package_builder import build_research_package

if __name__ == "__main__":
    res = build_research_package()
    print("FlexPulse-ML Evidence Package reproduced successfully.")
