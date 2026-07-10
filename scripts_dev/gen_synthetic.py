"""
Generates a small synthetic dataset (catalogue.json + date-folder waveform
JSON files) matching the real data shape, purely so the pipeline
(build_dataset -> labeling -> train -> evaluate_normal) can be smoke-tested
without real AAMS data. NOT used in production - delete data/raw contents
before pointing this project at your real data.
"""
import json
import random
from pathlib import Path

import numpy as np

random.seed(0)
np.random.seed(0)

ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"
ROOT.mkdir(parents=True, exist_ok=True)
DATE = "2026-06-01"
(ROOT / DATE).mkdir(parents=True, exist_ok=True)

SR = 3000
N = 4096
STATUSES = ["Normal"] * 12 + ["Marginal"] * 10 + ["Unacceptable"] * 10

catalogue = []
files_written = 0

for i, status in enumerate(STATUSES):
    bearing_id = f"synthetic_bearing_{i:03d}"
    rpm = random.choice([1000, 1475, 1800, 3000])
    shaft_hz = rpm / 60.0

    cat_entry = {
        "bearingLocationId": bearing_id,
        "name": "MOTOR DE",
        "bearingMeasuringType": "sensor",
        "source": "online",
        "customerId": "cust_001",
        "customerName": "Synthetic Co",
        "machineId": f"machine_{i:03d}",
        "machineName": f"Test Machine {i}",
        "machineRpm": rpm,
        "statusName": status,
        "statusKey": status.upper(),
        "axes": ["X", "Y", "Z"],
        "fMax": 800,
        "nol": 1600,
        "createdAt": "2026-01-01T00:00:00",
        "bearingId": f"bearing_{i:03d}",
        "id": f"bearing_{i:03d}",
        "manufacturerName": "SKF",
        "bearingNumber": "6205",
        "type": "F",
        "innerRacePass": 4.9205,
        "outerRacePass": 3.0795,
        "rollElementPass": 8.0512,
        "cageRotation": 0.3849,
        "updatedAt": "2026-01-01T00:00:00",
    }
    catalogue.append(cat_entry)

    t = np.arange(N) / SR
    packets = []
    for axis in ["X", "Y", "Z"]:
        noise = np.random.normal(0, 0.05, N)

        if status in ("Marginal", "Unacceptable"):
            fault_type = random.choice(["innerRacePass", "outerRacePass"])
            fault_hz = cat_entry[fault_type] * shaft_hz
            amp = 0.3 if status == "Unacceptable" else 0.15
            signal = amp * np.sin(2 * np.pi * fault_hz * t)
            signal += 0.5 * amp * np.sin(2 * np.pi * 2 * fault_hz * t)  # 2nd harmonic
            # sidebands
            signal += 0.2 * amp * np.sin(2 * np.pi * (fault_hz + shaft_hz) * t)
            signal += 0.2 * amp * np.sin(2 * np.pi * (fault_hz - shaft_hz) * t)
            samples = signal + noise
        else:
            samples = noise + 0.05 * np.sin(2 * np.pi * shaft_hz * t)

        packets.append({
            "axis": axis,
            "epoch_time": 1780000000,
            "timestamp": f"2026-06-01T00:00:00+00:00",
            "sampling_rate": SR,
            "num_samples": N,
            "analytics_type": None,
            "f_max": None,
            "nol": None,
            "rpm": rpm,
            "samples": [round(float(s), 5) for s in samples],
        })

    doc = {
        "bearingLocationId": bearing_id,
        "source": "online",
        "date": DATE,
        "count": len(packets),
        "status": status,
        "customerName": "Synthetic Co",
        "machineName": f"Test Machine {i}",
        "rpm": rpm,
        "measuringType": "sensor",
        "packets": packets,
    }
    with open(ROOT / DATE / f"{bearing_id}.json", "w") as f:
        json.dump(doc, f)
    files_written += 1

with open(ROOT / "catalogue.json", "w") as f:
    json.dump(catalogue, f, indent=2)

print(f"Wrote {files_written} waveform files + catalogue.json with {len(catalogue)} bearings -> {ROOT}")
