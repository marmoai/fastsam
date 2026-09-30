#!/usr/bin/env python3
"""Run isolated end-to-end ablations for each hard-product cleanup operation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from sam_ab_regression import BACKEND
from sam_cleanup_ab import run_variant


VARIANTS = {
    "A_cleanup_on": None,
    "skip_components": "components",
    "skip_close": "close",
    "skip_open": "open",
    "skip_holes": "holes",
}
CASE_REPEATS = {"person": 3, "armchair": 1, "tiered_table": 1, "food": 3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port-base", type=int, default=18005)
    args = parser.parse_args()
    output = BACKEND / "runs" / "cleanup-steps" / datetime.now().strftime("%Y%m%d-%H%M%S")
    output.mkdir(parents=True)
    report = {"variants": VARIANTS, "repeatsByCase": CASE_REPEATS, "cases": {}}
    print(f"OUTPUT {output}", flush=True)
    for index, (name, step) in enumerate(VARIANTS.items()):
        run_variant(
            name, "0", args.port_base + index, output, report,
            skip_step=step, case_repeats=CASE_REPEATS,
        )
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"COMPLETE {output}", flush=True)


if __name__ == "__main__":
    main()
