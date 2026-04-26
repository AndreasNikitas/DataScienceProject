"""
Football Match Predictor — Full Pipeline Runner

Run this single script to go from raw data to predictions in one command:

    python src/run_pipeline.py

Steps executed in order:
    1. generate_features  — build the feature table from historical matches
    2. train_models       — tune, train, cross-validate, and save both models + plots
    3. predict_upcoming   — predict all upcoming matches using the saved models

Each step is optional via command-line flags (see --help).
"""

import argparse
import sys
import time

from generate_features import generate_all_features
from train_models import train_and_evaluate_models
from predict_upcoming import predict_upcoming_matches


def run_step(name: str, fn, *args, **kwargs):
    """Run a pipeline step, print a header, and time it."""
    border = "=" * 70
    print(f"\n{border}")
    print(f"  STEP: {name}")
    print(f"{border}\n")
    start = time.time()
    fn(*args, **kwargs)
    elapsed = time.time() - start
    print(f"\n✅ {name} complete ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(
        description="Run the full football prediction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/run_pipeline.py                        # run all steps
  python src/run_pipeline.py --skip-features        # skip feature generation
  python src/run_pipeline.py --skip-training        # skip model training
  python src/run_pipeline.py --skip-predict         # only generate features + train
  python src/run_pipeline.py --league esp.1         # predict Spanish league
        """,
    )
    parser.add_argument("--skip-features", action="store_true",
                        help="Skip feature generation (use existing features in DB)")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip model training (use existing saved models)")
    parser.add_argument("--skip-predict", action="store_true",
                        help="Skip upcoming match predictions")
    parser.add_argument("--form-window", type=int, default=20,
                        help="Number of recent matches used for form features (default: 20)")
    parser.add_argument("--league", default="eng.1",
                        help="ESPN league slug for player availability (default: eng.1)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  FOOTBALL MATCH PREDICTOR — PIPELINE START")
    print("=" * 70)
    pipeline_start = time.time()

    if not args.skip_features:
        run_step(
            "Generate Features",
            generate_all_features,
            form_window=args.form_window,
        )
    else:
        print("\n⏭  Skipping feature generation (--skip-features)")

    if not args.skip_training:
        run_step("Train Models", train_and_evaluate_models)
    else:
        print("⏭  Skipping model training (--skip-training)")

    if not args.skip_predict:
        run_step(
            "Predict Upcoming Matches",
            predict_upcoming_matches,
            league_slug=args.league,
            form_window=args.form_window,
        )
    else:
        print("⏭  Skipping predictions (--skip-predict)")

    total = time.time() - pipeline_start
    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE — total time: {total:.1f}s")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
