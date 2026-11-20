"""CrowdPulse main entrypoint: runs pipeline and compiles visual HTML report.

Usage:
    python run.py                  # Live pipeline run (or fallback to latest) -> builds report.html
    python run.py --sample         # Runs from offline fixtures -> builds report.html
    python run.py --skip-scrape    # Reuses existing scraped data in latest/ -> builds report.html
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="CrowdPulse end-to-end execution")
    parser.add_argument("--sample", action="store_true", help="Build report using sample fixture data")
    parser.add_argument("--skip-scrape", action="store_true", help="Run pipeline using existing scraped data in latest/")
    parser.add_argument("--output", default="report.html", help="Path to output HTML report")
    parser.add_argument("--img", default="leaderboard.png", help="Path to output leaderboard image")
    args = parser.parse_args()

    run_dir = ROOT / "data" / "outputs" / "sample"

    if args.sample:
        if not (run_dir / "summary.json").is_file():
            print("Generating sample fixtures...")
            subprocess.run([sys.executable, str(ROOT / "fixtures" / "make_fixtures.py"), "--output-dir", str(run_dir)], check=True)
    else:
        # Check if API keys exist
        env_file = ROOT / ".env"
        has_keys = env_file.is_file() and ("APIFY_TOKEN" in env_file.read_text(encoding="utf-8", errors="ignore"))
        
        if not has_keys and not args.skip_scrape:
            print("Notice: No live API keys in .env. Using data/outputs/latest or sample fixtures.")
            latest_dir = ROOT / "data" / "outputs" / "latest"
            if (latest_dir / "summary.json").is_file():
                run_dir = latest_dir
            else:
                run_dir = ROOT / "data" / "outputs" / "sample"
        else:
            pipeline_cmd = [sys.executable, str(ROOT / "scripts" / "run_pipeline.py")]
            if args.skip_scrape:
                pipeline_cmd.append("--skip-scrape")
            print("Executing CrowdPulse pipeline...")
            res = subprocess.run(pipeline_cmd)
            latest_dir = ROOT / "data" / "outputs" / "latest"
            if (latest_dir / "summary.json").is_file():
                run_dir = latest_dir

    print(f"Building standalone HTML report and leaderboard from {run_dir}...")
    report_cmd = [
        sys.executable,
        str(ROOT / "report" / "build_report.py"),
        "--run", str(run_dir),
        "--output", str(args.output),
        "--img", str(args.img)
    ]
    subprocess.run(report_cmd, check=True)
    print(f"\n[DONE] Successfully generated {args.output} and {args.img}!")


if __name__ == "__main__":
    main()
