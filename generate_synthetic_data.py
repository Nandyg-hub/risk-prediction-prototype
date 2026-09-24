"""
generate_synthetic_data.py
----------------------------------
Creates a synthetic medical device complaint dataset that mimics real-world
post-market surveillance data. A handful of products have an embedded
"risk escalation" pattern (rising complaint frequency + rising severity)
that culminates in a simulated field-safety event. This lets us test
whether the risk-prediction model can catch the warning signs BEFORE
the event, which is the core value proposition of the platform idea.

Output: data/complaints.csv
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---- Configuration ----
N_PRODUCTS = 12
START_DATE = datetime(2024, 1, 1)
N_DAYS = 730  # 2 years
CATEGORIES = [
    "battery_failure", "software_error", "mechanical_defect",
    "labeling_issue", "connector_failure", "sensor_inaccuracy",
    "user_error", "packaging_defect"
]

# Products that will have an embedded escalating-risk pattern
# (simulating a real emerging safety issue building up over time)
RISKY_PRODUCTS = {2: 420, 6: 550, 9: 630}  # product_id -> day of "event"

COMPLAINT_TEMPLATES = {
    "battery_failure": [
        "Battery drained much faster than expected and the device shut off unexpectedly during use.",
        "Unexpected shutdown occurred because battery life was shorter than the rated shelf life.",
        "Device powered down on its own; battery depleted well before expected shelf life.",
    ],
    "software_error": [
        "Software froze during active operation and required a manual restart to recover.",
        "Application crashed mid-session; device needed a manual restart afterward.",
        "System locked up during normal operation, forcing a manual restart.",
    ],
    "mechanical_defect": [
        "Device housing cracked under normal handling during routine use.",
        "Mechanical linkage loosened after normal handling, affecting device function.",
        "Housing showed a crack after being handled in a normal, expected way.",
    ],
    "labeling_issue": [
        "Instructions for use were ambiguous, leading to incorrect setup by the user.",
        "User set up the device incorrectly because the instructions for use were unclear.",
        "Ambiguous instructions for use caused confusion during initial device setup.",
    ],
    "connector_failure": [
        "Cable connector lost contact intermittently, causing the signal to drop out.",
        "Signal dropout occurred repeatedly due to an intermittent connector contact issue.",
        "Connector contact was unreliable, leading to intermittent signal dropout during use.",
    ],
    "sensor_inaccuracy": [
        "Sensor readings drifted out of the calibration range after repeated use cycles.",
        "Readings became inaccurate over time as the sensor drifted out of calibration.",
        "After many use cycles, sensor calibration drifted, producing inaccurate readings.",
    ],
    "user_error": [
        "Device was operated outside its intended use conditions due to an unclear workflow.",
        "Unclear workflow led the user to operate the device outside intended conditions.",
        "Operator used the device outside intended conditions because the workflow was unclear.",
    ],
    "packaging_defect": [
        "Sterile packaging seal was compromised before first use of the device.",
        "Packaging seal was broken prior to first use, compromising sterility.",
        "Device packaging arrived with a compromised seal before it was first used.",
    ],
}

product_ids = [f"DEV-{i:03d}" for i in range(1, N_PRODUCTS + 1)]


def _severity_dist(ramp):
    # shift probability mass toward higher severities as ramp increases
    low = max(0.05, 0.45 - 0.35 * ramp)
    mid1 = max(0.05, 0.30 - 0.10 * ramp)
    mid2 = 0.15
    high1 = 0.07 + 0.15 * ramp
    high2 = 0.03 + 0.20 * ramp
    total = low + mid1 + mid2 + high1 + high2
    return [low/total, mid1/total, mid2/total, high1/total, high2/total]


rows = []
complaint_id = 1

for day_offset in range(N_DAYS):
    current_date = START_DATE + timedelta(days=day_offset)

    for idx, pid in enumerate(product_ids, start=1):
        # baseline complaint probability per product per day
        base_rate = 0.12

        # inject escalating risk pattern for selected products
        if idx in RISKY_PRODUCTS:
            event_day = RISKY_PRODUCTS[idx]
            days_to_event = event_day - day_offset
            if 0 <= days_to_event <= 120:
                # ramp up complaint rate and severity as event approaches
                ramp = (120 - days_to_event) / 120  # 0 -> 1
                base_rate = base_rate + ramp * 0.65
            elif days_to_event < 0 and days_to_event >= -30:
                # short tail after event (recall/CAPA response period)
                base_rate = base_rate + 0.25

        # decide if a complaint occurs today for this product
        n_complaints_today = np.random.poisson(base_rate)

        for _ in range(n_complaints_today):
            category = np.random.choice(CATEGORIES)

            # severity: 1 (minor) - 5 (serious/death-adjacent)
            if idx in RISKY_PRODUCTS and 0 <= (RISKY_PRODUCTS[idx] - day_offset) <= 120:
                ramp = (120 - (RISKY_PRODUCTS[idx] - day_offset)) / 120
                severity = min(5, np.random.choice([1, 2, 3, 4, 5],
                                p=_severity_dist(ramp)))
            else:
                severity = np.random.choice([1, 2, 3, 4, 5], p=[0.45, 0.30, 0.15, 0.07, 0.03])

            rows.append({
                "complaint_id": f"C-{complaint_id:06d}",
                "date": current_date.strftime("%Y-%m-%d"),
                "product_id": pid,
                "category": category,
                "severity": int(severity),
                "description": np.random.choice(COMPLAINT_TEMPLATES[category])
            })
            complaint_id += 1

df = pd.DataFrame(rows)
df.to_csv("data/complaints.csv", index=False)

print(f"Generated {len(df)} complaints across {N_PRODUCTS} products over {N_DAYS} days.")
print(f"Embedded risk-escalation events on products: "
      f"{[product_ids[i-1] for i in RISKY_PRODUCTS]} "
      f"at day offsets {list(RISKY_PRODUCTS.values())}")
