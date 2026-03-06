"""CLI entrypoint for running a single simulation experiment.

Usage:
    python main.py path/to/experiment.json --out out_dir

The experiment setup file is parsed by :func:`tms_sim.load_experiment_config` and the
simulation itself is executed by :func:`tms_sim.run_experiment`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tms_sim import load_experiment_config, run_experiment



def main() -> int:
    """Parse CLI args, run the experiment, and optionally write outputs.

    Inputs (CLI):
        setup: Path to experiment setup JSON.
        --out: Optional output directory.

    Returns:
        Process exit code (0 on success).
    """

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run TMS trust simulation experiment")
    parser.add_argument("setup", type=str, help="Path to experiment setup JSON")
    parser.add_argument("--out", type=str, default=None, help="Optional output directory")
    args = parser.parse_args()

    # Load experiment config and run the simulation.
    cfg = load_experiment_config(args.setup)
    result = run_experiment(cfg)

    # Print summary stats to console.
    print("Experiment finished")
    print(json.dumps(result.stats, indent=2))

    # Optionally write outputs to files.
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
        # Keep transactions optional; can be large.
        tx_path = out_dir / "transactions.jsonl"
        with tx_path.open("w", encoding="utf-8") as f:
            for tx in result.transactions:
                f.write(json.dumps(tx.__dict__) + "\n")
        print(f"Wrote outputs to {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
