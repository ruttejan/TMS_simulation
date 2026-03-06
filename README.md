# TMS_simulation

Python implementation of the simulation described in `ideas/simulation_overview.md` (non-Appendix part).

## Run

From this folder:

```
python main.py experiments/example.json --out out/example_run
```

Outputs:

- `summary.json` — aggregated statistics
- `transactions.jsonl` — one transaction per line (can be large)

## Experiment setup format

Experiments are defined as a JSON file. See `experiments/example.json`.

Key concepts:

- Peers are created from one or more `peer_groups`.
- Each peer has `(q, h)`:
	- `q` = service quality, used for ground-truth outcome
	- `h` = reporting honesty, used for rating inversion
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
