import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


SUITES = {
    "quick": ["tiny_exact_debug", "small_demo"],
    "main": ["main_dense_small_cell", "high_conflict_tradeoff_case"],
    "overnight": ["main_dense_small_cell", "hard_dense_small_cell", "channel_rich_case", "high_conflict_tradeoff_case", "multi_seed_main"],
}


def run_case(case):
    cmd = [sys.executable, "run_experiment.py", "--case", case]
    result = subprocess.run(cmd, text=True, capture_output=True)
    status = "pass" if result.returncode == 0 else "fail"
    row = {"case_name": case, "status": status}
    if Path("results/method_summary.csv").exists():
        df = pd.read_csv("results/method_summary.csv")
        best_final = df.sort_values("final_hv", ascending=False).iloc[0]
        best_auc = df.sort_values("auc_hv", ascending=False).iloc[0]
        row.update({
            "best_final_method": best_final["method"],
            "best_final_hv": best_final["final_hv"],
            "best_auc_method": best_auc["method"],
            "best_auc_hv": best_auc["auc_hv"],
        })
    if status == "fail":
        row["stderr"] = result.stderr[-1000:]
    return row


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true")
    group.add_argument("--main", action="store_true")
    group.add_argument("--overnight", action="store_true")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        cases = sorted(set(SUITES["quick"] + SUITES["main"] + SUITES["overnight"]))
    elif args.main:
        cases = SUITES["main"]
    elif args.overnight:
        cases = SUITES["overnight"]
    else:
        cases = SUITES["quick"]
    rows = [run_case(c) for c in cases]
    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/benchmark_suite_summary.csv", index=False)
    Path("results/benchmark_suite_summary.md").write_text(df.to_markdown(index=False), encoding="utf-8")
    print(df.to_string(index=False))
    return 0 if (df["status"] == "pass").all() else 1


if __name__ == "__main__":
    sys.exit(main())

