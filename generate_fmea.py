"""
generate_fmea.py
----------------------------------
Creates a synthetic FMEA (Failure Mode and Effects Analysis) risk register
per product -- this represents the "known risks" a real company would
already have documented in their risk management file (per ISO 14971).

Each risk item has a short text description, which is what we will match
new incoming complaints against in complaint_matching.py.

Output: data/fmea_risk_items.csv
"""

import pandas as pd

# Reuse the same 8 categories from the complaint generator, and give each
# product 4-6 known risk items covering some (not all) of these categories.
# This mirrors reality: a risk file never covers 100% of what eventually
# goes wrong -- that gap is exactly what the "unknown risk" module targets.

FAILURE_MODES = {
    "battery_failure": "Battery depletes faster than rated shelf life, causing unexpected device shutdown during use.",
    "software_error": "Software freezes or crashes during active operation, requiring manual restart.",
    "mechanical_defect": "Housing or mechanical linkage cracks or loosens under normal handling.",
    "labeling_issue": "Instructions for use are ambiguous, leading to incorrect setup by the user.",
    "connector_failure": "Cable connector loses contact intermittently, causing signal dropout.",
    "sensor_inaccuracy": "Sensor readings drift out of calibration range over repeated use cycles.",
    "user_error": "Device is operated outside intended use conditions due to unclear workflow.",
    "packaging_defect": "Sterile packaging seal is compromised before first use.",
}

PRODUCTS_RISK_COVERAGE = {
    "DEV-001": ["battery_failure", "mechanical_defect", "labeling_issue", "user_error"],
    "DEV-002": ["software_error", "connector_failure", "sensor_inaccuracy"],  # deliberately missing battery/mechanical
    "DEV-003": ["mechanical_defect", "packaging_defect", "user_error", "labeling_issue"],
    "DEV-004": ["battery_failure", "software_error", "connector_failure"],
    "DEV-005": ["sensor_inaccuracy", "user_error", "mechanical_defect"],
    "DEV-006": ["battery_failure", "mechanical_defect", "labeling_issue"],  # missing sensor/connector on purpose
    "DEV-007": ["software_error", "packaging_defect", "connector_failure"],
    "DEV-008": ["user_error", "sensor_inaccuracy", "labeling_issue"],
    "DEV-009": ["battery_failure", "software_error", "mechanical_defect"],  # missing connector on purpose
    "DEV-010": ["connector_failure", "packaging_defect", "user_error"],
    "DEV-011": ["sensor_inaccuracy", "labeling_issue", "battery_failure"],
    "DEV-012": ["mechanical_defect", "software_error", "user_error"],
}

rows = []
risk_id = 1
for product, categories in PRODUCTS_RISK_COVERAGE.items():
    for cat in categories:
        rows.append({
            "risk_id": f"FM-{risk_id:03d}",
            "product_id": product,
            "category": cat,
            "description": FAILURE_MODES[cat],
        })
        risk_id += 1

df = pd.DataFrame(rows)
df.to_csv("data/fmea_risk_items.csv", index=False)
print(f"Generated {len(df)} FMEA risk items across {len(PRODUCTS_RISK_COVERAGE)} products.")
print("Note: each product is MISSING coverage for 1-2 real complaint categories on purpose,")
print("so the 'unknown risk' detector has something genuine to find.")
