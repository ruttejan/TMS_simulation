# TMS_simulation

## Run

Always use the `--out` flag. Without the flag the program only prints the the statistics on the terminal.

```
python main.py experiments\example.json5 --out out\example
```

or

```
python main.py experiments\THREAT_A\A1.json5 -- out out\A\A1
```

The outputs are written into per-algorithm subfolders and further in per-seed subfolders.

Outputs:

- `global_trust.csv` - final global trust values
- `summary.json` — statistics
- `transactions.jsonl` — one transaction per line (can be large)
- plots of final global trust values and distributions of sellers/buyers

## Experiment setup format

Experiments are defined as a JSON/JSON5 file. 
See `experiments/example.json5`. You will see every possible setup with comments.


## Code layout

- `main.py` - reding input and running main loop
- `tms_sim/simulation.py` - main loop
- `tms_sim/transaction.py` - rating generation
- `tms_sim/price.py` - price handler implementation
- `tms_sim/selection.py` - argmax / softmax seller selection
- `tms_sim/local_trust.py` - local trust computation and storing
- `tms_sim/global_trust.py` - global trust computation and storing
- `tms_sim/eigentrust.py` - EigenTrust implementation
- `tms_sim/shapetrust.py` - SHAPE-Trust implementation
- `tms_sim/peers.py` - peer definitions
- `tms_sim/stats.py` - statistical metrics and plots
- `tms_sim/config.py` - config dataclasses + JSON loader
- `tms_sim/distributions.py` - helper functions

## Extras

### Aggregation of statistics

For averaging all seeds for all algorithms in a scenario we have `aggregate_algorithm_summaries.py`.

```
python aggregate_algorithm_summaries.py --input .\out\A\A1 --output .\aggregated_summaries\A\A1
```

### Trator analysis notebook

In the `seller_cheat_analysis.ipynb` we did the posthoc analysis of Traitor peers for Experiment D.

# Requirements

Python 3.9+ and the following libraries:

```
pip install numpy pandas matplotlib numba json5
```

- `numpy` - numerical computation
- `pandas` - transaction data handling
- `matplotlib` - plots
- `numba` - SHAPE-Trust acceleration
- `json5` - reading JSON5 experiment files