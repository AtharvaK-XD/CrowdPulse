"""Fill actual closes into data/outputs/predictions_log.csv and print a scorecard.

Run after the predicted day's market has closed (HKEX 08:00 UTC, KRX 06:30 UTC):
    python scripts/check_predictions.py

Rows whose target day hasn't closed yet (or had no bar, e.g. a holiday) stay
pending and are picked up on a later run. The updated log is also copied to
data/outputs/latest/ so the report shows the latest scorecard.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowdpulse.config import load_config  # noqa: E402
from crowdpulse.model.check import fill_actuals, scorecard  # noqa: E402
from crowdpulse.schema import PREDICTIONS_LOG_COLUMNS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check past predictions against actual closes.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    settings = load_config(args.config)
    log_path = settings.data_dir / "outputs" / "predictions_log.csv"
    if not log_path.is_file():
        print(f"No predictions log yet at {log_path}; run scripts/run_pipeline.py first.")
        return 1
    log = pd.read_csv(log_path, dtype={"run_id": str, "ticker": str, "target_date": str})
    log, filled = fill_actuals(log, settings)
    log = log[list(PREDICTIONS_LOG_COLUMNS)]
    log["direction_hit"] = log["direction_hit"].astype("Int64")
    log.to_csv(log_path, index=False)
    latest = settings.data_dir / "outputs" / "latest"
    if latest.is_dir():
        shutil.copy2(log_path, latest / "predictions_log.csv")

    pending = int(log["actual_close"].isna().sum())
    print(f"Filled {filled} prediction(s); {pending} still pending.\n")
    card = scorecard(log)
    if card.empty:
        print("No checked predictions yet.")
    else:
        print(card.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
