import argparse
import subprocess
import sys
from pathlib import Path
import pandas as pd


VARIANTS = [
    ("onehot_penalty + expected_cost", ["--qaoa-encoding", "onehot_penalty", "--qaoa-objective", "expected_cost"], "tiny_exact_debug", "onehot proxy because onehot state grows as 2^(U*C)"),
    ("onehot_penalty + cvar", ["--qaoa-encoding", "onehot_penalty", "--qaoa-objective", "cvar"], "tiny_exact_debug", "onehot proxy because onehot state grows as 2^(U*C)"),
    ("feasible_mixer + expected_cost", ["--qaoa-encoding", "feasible_mixer", "--qaoa-objective", "expected_cost"], None, "main case"),
    ("feasible_mixer + cvar", ["--qaoa-encoding", "feasible_mixer", "--qaoa-objective", "cvar"], None, "main case"),
    ("feasible_mixer + cvar + top", ["--qaoa-encoding", "feasible_mixer", "--qaoa-objective", "cvar", "--candidate-selection", "top"], None, "main case"),
    ("feasible_mixer + cvar + diverse", ["--qaoa-encoding", "feasible_mixer", "--qaoa-objective", "cvar", "--candidate-selection", "diverse"], None, "main case"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="main_dense_small_cell")
    args = parser.parse_args()
    rows = []
    for name, extra, case_override, note in VARIANTS:
        case = case_override or args.case
        cmd = [sys.executable, "run_experiment.py", "--case", case, "--outer-iters", "2", "--maxiter", "4", "--budget", "120", "--shots", "96"] + extra
        result = subprocess.run(cmd, text=True, capture_output=True)
        if result.returncode == 0 and Path("results/method_summary.csv").exists():
            df = pd.read_csv("results/method_summary.csv")
            q = df[df["method"].str.contains("QAOA|Quantum-Seeded", regex=True)].sort_values("auc_hv", ascending=False).iloc[0]
            rows.append({"case_name": case, "requested_case": args.case, "method_variant": name, **q.to_dict(), "notes": note})
        else:
            rows.append({"case_name": case, "requested_case": args.case, "method_variant": name, "notes": "fail"})
    out = pd.DataFrame(rows)
    out.to_csv("results/full_ablation_suite.csv", index=False)
    Path("results/full_ablation_suite.md").write_text(out.to_markdown(index=False), encoding="utf-8")
    print(out[["case_name", "method_variant", "notes"]].to_string(index=False))


if __name__ == "__main__":
    main()
