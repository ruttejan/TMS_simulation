# TMS_simulation

Python implementation of the simulation described in `ideas/simulation_overview.md` (non-Appendix part).

## Run

From this folder:

```
python main.py experiments/example.json5 --out out/example_run
```

If the experiment config uses multiple seeds (``seed`` as an array), the experiment
is run once per seed, and outputs are written into per-seed subfolders:

```
python main.py experiments/example.json5 --out out/example_run
# writes out/example_run/seed_123/, out/example_run/seed_456/, ...
```

Outputs:

- `summary.json` — aggregated statistics
- `transactions.jsonl` — one transaction per line (can be large)

## Experiment setup format

Experiments are defined as a JSON/JSON5 file. See `experiments/example.json5`.

JSON5 allows comments, trailing commas, and unquoted keys.

Key concepts:

- Peers are created from typed `peers` entries (`kind`, `count`, optional `params`).
- `receivers_per_step` can be a fixed integer or an interval object (`min_count`, `max_count`) sampled each step.
- `seed` can be either a single integer (`seed: 123`) or an array of integers (`seed: [123, 456, 789]`).
- Peers are always online (no churn model).
- Local trust is a decayed, price-weighted average of normalized star ratings.
- Seller selection uses a convex combination of local trust and global seller reputation.

## Code layout

- `tms_sim/config.py` — config dataclasses + JSON loader
- `tms_sim/transaction.py` — outcome + rating generation
- `tms_sim/trust.py` — local trust and seller reputation stores
- `tms_sim/selection.py` — argmax / softmax seller selection
- `tms_sim/stats.py` — aggregated measurements
- `tms_sim/simulation.py` — main loop
