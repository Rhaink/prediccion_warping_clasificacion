#!/usr/bin/env python3
import argparse
import itertools
import re
import subprocess
import sys
from pathlib import Path


def parse_error(output: str) -> float:
    patterns = [
        r"Error promedio:\s*([0-9]+(?:\.[0-9]+)?)\s*px",
        r"Mean Error:\s*([0-9]+(?:\.[0-9]+)?)\s*px",
    ]
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            return float(match.group(1))
    raise ValueError("No error metric found in evaluation output.")


def evaluate_combo(models, tta: bool, clahe: bool) -> float:
    cmd = [sys.executable, "-m", "src_v2", "evaluate-ensemble", *models]
    if tta:
        cmd.append("--tta")
    if clahe:
        cmd.append("--clahe")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate-ensemble failed:\n{proc.stdout}")
    return parse_error(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep ensemble combinations.")
    parser.add_argument("--k", type=int, default=4, help="Ensemble size.")
    parser.add_argument("--tta", action="store_true", help="Enable TTA.")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE.")
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output file to save the sweep summary.",
    )
    parser.add_argument("models", nargs="+", help="Checkpoint paths.")
    args = parser.parse_args()

    models = [str(Path(p)) for p in args.models]
    for path in models:
        if not Path(path).is_file():
            raise SystemExit(f"Missing checkpoint: {path}")

    combos = list(itertools.combinations(models, args.k))
    results = []

    for combo in combos:
        error = evaluate_combo(combo, args.tta, args.clahe)
        results.append((error, combo))
        print(f"{error:.2f} px | {' '.join(combo)}")

    results.sort(key=lambda item: item[0])
    best_error, best_combo = results[0]
    best_line = f"BEST: {best_error:.2f} px | {' '.join(best_combo)}"
    print(best_line)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            for error, combo in results:
                handle.write(f"{error:.2f} px | {' '.join(combo)}\n")
            handle.write(best_line + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
