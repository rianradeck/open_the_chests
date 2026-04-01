from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping


def write_results_json(path: Path, results: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(dict(results), f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def read_results_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_results_csv(csv_path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return

    fieldnames: list[str] = sorted({k for r in rows_list for k in r.keys()})
    file_exists = csv_path.exists()

    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows_list:
            writer.writerow({k: row.get(k) for k in fieldnames})
