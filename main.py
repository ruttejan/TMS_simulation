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

from tms_sim import load_experiment_configs, run_experiment



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

    # Load experiment config(s) and run the simulation(s).
    cfgs = load_experiment_configs(args.setup)

    out_root: Path | None
    if args.out:
        out_root = Path(args.out)
        out_root.mkdir(parents=True, exist_ok=True)
    else:
        out_root = None

    multi = len(cfgs) > 1

    for i, cfg in enumerate(cfgs, start=1):
        run_out: Path | None
        if out_root is None:
            run_out = None
        else:
            mode_out = out_root / str(cfg.global_trust.mode)
            mode_out.mkdir(parents=True, exist_ok=True)

            if multi:
                run_out = mode_out / f"seed_{cfg.seed}"
                run_out.mkdir(parents=True, exist_ok=True)
            else:
                run_out = mode_out

        plot_path = None
        if run_out is not None:
            plot_path = run_out / f"{cfg.global_trust.mode}_min_{cfg.n_steps}.png"

        result = run_experiment(cfg, plot_path=plot_path)

        # Print summary stats to console.
        label = f"Run {i}/{len(cfgs)} (seed={cfg.seed})" if multi else "Experiment finished"
        print(label)
        print(json.dumps(result.stats, indent=2))

        # Optionally write outputs to files.
        if run_out is not None:
            (run_out / "summary.json").write_text(json.dumps(result.stats, indent=2), encoding="utf-8")
            # Keep transactions optional; can be large.
            tx_path = run_out / "transactions.jsonl"
            with tx_path.open("w", encoding="utf-8") as f:
                for tx in result.transactions:
                    f.write(json.dumps(tx.__dict__) + "\n")
            print(f"Wrote outputs to {run_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
