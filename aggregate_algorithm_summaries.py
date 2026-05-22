from __future__ import annotations

import argparse
import json
from collections import defaultdict
from statistics import mean, pstdev
from numbers import Number
from pathlib import Path
from typing import Any


def _is_number(value: Any) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _find_summary_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.rglob("summary.json") if path.is_file())


def _find_algorithm_dirs(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.iterdir() if path.is_dir())


def _round_numeric_values(value: Any, digits: int = 4) -> Any:
    if _is_number(value):
        return round(float(value), digits)

    if isinstance(value, dict):
        return {key: _round_numeric_values(nested_value, digits) for key, nested_value in value.items()}

    if isinstance(value, list):
        return [_round_numeric_values(item, digits) for item in value]

    return value


def _average_summaries(summary_files: list[Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    top_level: dict[str, list[float]] = defaultdict(list)
    nested: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for file_path in summary_files:
        with file_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        for key, value in data.items():
            if _is_number(value):
                top_level[key].append(float(value))
                continue

            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if _is_number(nested_value):
                        nested[key][nested_key].append(float(nested_value))

    averaged: dict[str, Any] = {}
    stds: dict[str, Any] = {}

    for key, values in top_level.items():
        averaged[key] = round(mean(values), 4)
        stds[key] = round(pstdev(values), 4)

    for key, nested_values in nested.items():
        averaged[key] = {
            nested_key: round(mean(nested_values_list), 4)
            for nested_key, nested_values_list in nested_values.items()
        }
        stds[key] = {
            nested_key: round(pstdev(nested_values_list), 4)
            for nested_key, nested_values_list in nested_values.items()
        }

    return averaged, stds


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Given a run root containing algorithm subdirectories, "
            "average all summary.json files for each algorithm and save one JSON per algorithm."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a run root containing algorithm subdirectories, e.g. out/A/A1",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Folder where the averaged summary JSON files will be written",
    )

    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist or is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    algorithm_dirs = _find_algorithm_dirs(input_dir)
    if not algorithm_dirs:
        raise SystemExit(f"No algorithm subdirectories found under: {input_dir}")

    written_files: list[Path] = []

    for algorithm_dir in algorithm_dirs:
        summary_files = _find_summary_files(algorithm_dir)
        if not summary_files:
            print(f"Skipping '{algorithm_dir.name}': no summary.json files found under {algorithm_dir}")
            continue

        averaged, stds = _average_summaries(summary_files)

        result = {
            "algorithm": algorithm_dir.name,
            "source_dir": str(algorithm_dir),
            "num_summaries": len(summary_files),
            "averages": averaged,
            "stds": stds,
            "summary_files": [str(path) for path in summary_files],
        }
        result = _round_numeric_values(result)

        output_file = output_dir / f"{algorithm_dir.name}_summary_average.json"
        with output_file.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")

        written_files.append(output_file)
        print(f"Averaged {len(summary_files)} summaries for '{algorithm_dir.name}'.")
        print(f"Saved: {output_file}")

    if not written_files:
        raise SystemExit(f"No summary.json files found under any algorithm directory in: {input_dir}")


if __name__ == "__main__":
    main()
