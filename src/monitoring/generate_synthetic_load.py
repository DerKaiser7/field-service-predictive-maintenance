"""
Synthetic load generator for local demo purposes.

Fires /predict requests against a running API instance so:
- Prometheus/Grafana have real request-rate/latency data to show.
- prediction_logs.features gets populated with enough rows for
  src/monitoring/drift_report.py to have something to compare.

The back half of the run deliberately drifts vibration/error/maintenance
features upward (simulating a fleet degrading over time) so the resulting
drift report has something real to detect, rather than a "no drift" no-op.

Usage:
    python -m src.monitoring.generate_synthetic_load --requests 200 --api-url http://localhost:8000
"""

import argparse
import random
import time
from datetime import datetime, timedelta

import requests

BASE_FEATURES = {
    "model": "model3", "age_category": "aged", "age": 12,
    "voltage_mean_3h": 169.5, "voltage_std_3h": 2.2, "voltage_min_3h": 165.0, "voltage_max_3h": 173.0,
    "rotation_mean_3h": 445.0, "rotation_std_3h": 5.0, "rotation_min_3h": 434.0, "rotation_max_3h": 455.0,
    "pressure_mean_3h": 99.0, "pressure_std_3h": 1.4, "pressure_min_3h": 96.0, "pressure_max_3h": 102.0,
    "vibration_mean_3h": 42.0, "vibration_std_3h": 2.5, "vibration_min_3h": 38.0, "vibration_max_3h": 46.0,
    "voltage_mean_12h": 169.8, "voltage_std_12h": 2.1,
    "rotation_mean_12h": 445.5, "rotation_std_12h": 4.8,
    "pressure_mean_12h": 99.1, "pressure_std_12h": 1.4,
    "vibration_mean_12h": 41.8, "vibration_std_12h": 2.6,
    "voltage_mean_24h": 170.0, "voltage_std_24h": 2.0,
    "rotation_mean_24h": 446.0, "rotation_std_24h": 4.9,
    "pressure_mean_24h": 99.3, "pressure_std_24h": 1.4,
    "vibration_mean_24h": 41.5, "vibration_std_24h": 2.4,
    "error_count_24h": 1,
    "hours_since_last_error": 80.0,
    "distinct_error_types": 1,
    "days_since_last_maintenance": 45.0,
    "component_diversity": 1,
    "total_prior_failures": 1,
    "days_since_last_failure": 120.0,
    "distinct_failure_types": 1,
}


def drifted_payload(severity: float, rng: random.Random) -> dict:
    """severity in [0, 1] — 0 looks like BASE_FEATURES, 1 simulates an aged,
    error-prone, overdue-for-maintenance fleet."""
    f = dict(BASE_FEATURES)
    f["vibration_mean_24h"] += severity * 15 + rng.uniform(-1, 1)
    f["vibration_std_24h"] += severity * 4
    f["vibration_mean_3h"] += severity * 15
    f["error_count_24h"] = int(f["error_count_24h"] + severity * 5 + rng.uniform(0, 1))
    f["days_since_last_maintenance"] += severity * 150
    f["hours_since_last_error"] = max(1.0, f["hours_since_last_error"] - severity * 70)
    f["total_prior_failures"] = int(f["total_prior_failures"] + severity * 3)
    return f


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--delay", type=float, default=0.05, help="seconds between requests")
    args = parser.parse_args()

    rng = random.Random(42)
    print(f"Firing {args.requests} requests at {args.api_url}/predict...")

    ok, failed = 0, 0
    for i in range(args.requests):
        # Ramp severity from 0 to 1 over the back 60% of the run to simulate
        # gradual fleet drift rather than an abrupt step change.
        severity = max(0.0, (i - args.requests * 0.4) / (args.requests * 0.6))
        payload = {
            "machine_id": str(rng.randint(1, 20)),
            "observation_time": (datetime.utcnow() - timedelta(minutes=args.requests - i)).isoformat(),
            "features": drifted_payload(severity, rng),
        }
        try:
            resp = requests.post(f"{args.api_url}/predict", json=payload, timeout=5)
            resp.raise_for_status()
            ok += 1
        except requests.RequestException as exc:
            failed += 1
            print(f"  request {i} failed: {exc}")

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{args.requests} sent (severity={severity:.2f})")
        time.sleep(args.delay)

    print(f"\nDone. {ok} succeeded, {failed} failed.")
    print("Prometheus should now have request data — check Grafana after a few scrape intervals (~5s).")
    print("Run 'python -m src.monitoring.drift_report' now that prediction_logs has feature data.")


if __name__ == "__main__":
    main()
