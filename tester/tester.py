#!/usr/bin/env python3
"""
DSLR Comprehensive Automated Tester & Peer-Defense Verification Suite
=====================================================================
Tests:
  1. [Part 1 - Mandatory] describe.py vs. pandas.DataFrame.describe() numerical precision
  2. [Bonus 1] Extended stats.py metrics (var, range, iqr, skew, kurt, missing%, unique, pearson_corr) & edge cases
  3. [Rule Compliance] AST static check for forbidden built-in calls in describe.py & utils/stats.py
  4. [Bonus 2 & 3] Sigmoid overflow stability, L2 (Ridge) weight decay & bias exclusion verification
  5. [Part 3 - End-to-End] All 5 optimizers (bgd, sgd, mini, momentum, adam), houses.csv format check & sklearn accuracy_score (>= 98%)
"""

import argparse
import ast
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
DATA_TRAIN = ROOT_DIR / "data" / "dataset_train.csv"
DATA_TEST = ROOT_DIR / "data" / "dataset_test.csv"

sys.path.insert(0, str(SRC_DIR))
import describe as dc  # noqa: E402
import utils.optimizers as opt_utils  # noqa: E402
import utils.stats as st  # noqa: E402

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

HOUSES = ("Gryffindor", "Hufflepuff", "Ravenclaw", "Slytherin")


def banner(title: str) -> None:
    print(f"\n{CYAN}{BOLD}{'=' * 78}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{BOLD}{'=' * 78}{RESET}")


def record_result(results: list[tuple[str, bool, str]], name: str, passed: bool, detail: str = "") -> None:
    status = f"{GREEN}[PASS]{RESET}" if passed else f"{RED}[FAIL]{RESET}"
    print(f"  {status} {BOLD}{name}{RESET}" + (f" — {detail}" if detail else ""))
    results.append((name, passed, detail))


# =====================================================================
# Test 1: Mandatory describe.py vs. pandas.DataFrame.describe()
# =====================================================================
def test_describe_vs_pandas(results: list[tuple[str, bool, str]], tol: float = 1e-9) -> None:
    banner("TEST 1: Mandatory describe.py vs. Official pandas.DataFrame.describe()")
    df = pd.read_csv(DATA_TRAIN)
    ours = dc.describe(df)
    ref = df.describe()

    mandatory_metrics = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    print(f"  {'Metric':<10} | {'Max Abs Error':<18} | {'Max Rel Error':<18} | {'Status'}")
    print(f"  {'-' * 64}")

    all_ok = True
    for metric in mandatory_metrics:
        if metric not in ours.index:
            print(f"  {metric:<10} | {'MISSING ROW':<18} | {'N/A':<18} | {RED}FAIL{RESET}")
            all_ok = False
            continue
        diff = (ours.loc[metric] - ref.loc[metric]).abs()
        max_abs_err = float(diff.max())
        max_rel_err = float((diff / ref.loc[metric].abs().replace(0, np.nan)).max())
        ok = bool(np.allclose(ours.loc[metric], ref.loc[metric], rtol=tol, atol=tol))
        all_ok = all_ok and ok
        badge = f"{GREEN}OK{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {metric:<10} | {max_abs_err:<18.6e} | {max_rel_err:<18.6e} | {badge}")

    record_result(
        results,
        "Part 1: describe.py 8 Mandatory metrics match pandas.describe()",
        all_ok,
        f"rtol/atol < {tol:.0e}",
    )


# =====================================================================
# Test 2: Bonus 1 Extended Statistics, Pearson Correlation & Edge Cases
# =====================================================================
def test_bonus_stats(results: list[tuple[str, bool, str]], tol: float = 1e-9) -> None:
    banner("TEST 2: Bonus 1 Extended Statistical Metrics & Edge Cases")
    df = pd.read_csv(DATA_TRAIN)
    num_df = df.select_dtypes(include="number")
    ours = dc.describe(df)

    # Reference calculations
    ref_var = num_df.var(ddof=1)
    ref_range = num_df.max() - num_df.min()
    ref_iqr = num_df.quantile(0.75) - num_df.quantile(0.25)
    ref_missing = num_df.isna().mean() * 100.0
    ref_unique = num_df.nunique(dropna=True).astype(float)

    # Fisher-Pearson skewness (g1) & excess kurtosis (g2) from central moments
    ref_skew = pd.Series(dtype=float)
    ref_kurt = pd.Series(dtype=float)
    for col in num_df.columns:
        s = num_df[col].dropna()
        m2 = ((s - s.mean()) ** 2).mean()
        m3 = ((s - s.mean()) ** 3).mean()
        m4 = ((s - s.mean()) ** 4).mean()
        ref_skew[col] = m3 / (m2 ** 1.5)
        ref_kurt[col] = (m4 / (m2 ** 2)) - 3.0

    bonus_checks = [
        ("var (s^2, ddof=1)", ours.loc["var"], ref_var),
        ("range (max - min)", ours.loc["range"], ref_range),
        ("iqr (Q75 - Q25)", ours.loc["iqr"], ref_iqr),
        ("skew (Fisher-Pearson g1)", ours.loc["skew"], ref_skew),
        ("kurt (Excess kurtosis g2)", ours.loc["kurt"], ref_kurt),
        ("missing% (NaN ratio)", ours.loc["missing%"], ref_missing),
        ("unique (cardinality)", ours.loc["unique"], ref_unique),
    ]

    for label, s_ours, s_ref in bonus_checks:
        diff = (s_ours - s_ref).abs()
        abs_err = float(diff.max())
        rel_err = float((diff / s_ref.abs().replace(0, np.nan)).fillna(0.0).max())
        ok = bool(np.allclose(s_ours, s_ref, rtol=tol, atol=tol))
        record_result(
            results,
            f"Bonus 1 Metric: {label}",
            ok,
            f"max_abs_err = {abs_err:.3e}, max_rel_err = {rel_err:.3e}",
        )

    # Test Pearson correlation on Astronomy vs Defense Against the Dark Arts
    s_astro = num_df["Astronomy"]
    s_dada = num_df["Defense Against the Dark Arts"]
    r_ours = st.pearson_corr(s_astro, s_dada)
    r_ref = float(s_astro.corr(s_dada, method="pearson"))
    r_err = abs(r_ours - r_ref)
    record_result(
        results,
        "Bonus 1 Pearson Correlation (Astronomy vs. DADA)",
        r_err < tol and r_ours < -0.9999,
        f"r_ours = {r_ours:.6f}, r_pandas = {r_ref:.6f}, err = {r_err:.3e}",
    )

    # Edge cases: empty/all-NaN column, single element (N=1), constant column (N=4 since kurt requires N>=4)
    edge_df = pd.DataFrame({
        "all_nan": [np.nan, np.nan, np.nan, np.nan],
        "single_val": [42.0, np.nan, np.nan, np.nan],
        "constant": [7.0, 7.0, 7.0, 7.0],
    })
    edge_desc = dc.describe(edge_df)
    edge_ok = (
        edge_desc.loc["count", "all_nan"] == 0.0
        and np.isnan(edge_desc.loc["mean", "all_nan"])
        and edge_desc.loc["count", "single_val"] == 1.0
        and edge_desc.loc["min", "single_val"] == 42.0
        and edge_desc.loc["50%", "single_val"] == 42.0
        and np.isnan(edge_desc.loc["kurt", "single_val"])
        and edge_desc.loc["std", "constant"] == 0.0
        and edge_desc.loc["skew", "constant"] == 0.0
        and edge_desc.loc["kurt", "constant"] == 0.0
    )
    record_result(results, "Edge Cases (All-NaN, N=1, Constant Column)", edge_ok, "No crashes or IndexErrors")


# =====================================================================
# Test 3: AST Static Audit ("No Heavy Lifting" Rule in describe/stats)
# =====================================================================
def test_no_forbidden_builtins(results: list[tuple[str, bool, str]]) -> None:
    banner("TEST 3: 'No Heavy Lifting' Static AST Audit (describe.py & stats.py)")
    forbidden_methods = {
        "describe", "mean", "std", "var", "min", "max",
        "percentile", "quantile", "median", "skew", "kurt", "kurtosis", "corr", "nunique"
    }
    files_to_check = [SRC_DIR / "describe.py", SRC_DIR / "utils" / "stats.py"]

    for path in files_to_check:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        violations = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    # Ignore calls on our own `stats` module (e.g. stats.dfmean)
                    if node.func.attr in forbidden_methods:
                        violations.append(f"line {node.lineno}: .{node.func.attr}()")
                elif isinstance(node.func, ast.Name):
                    # Flag built-in min()/max() if used inside stats.py or describe.py
                    if node.func.id in forbidden_methods and not (
                        path.name == "describe.py" and node.func.id == "describe"
                    ):
                        violations.append(f"line {node.lineno}: {node.func.id}()")

        record_result(
            results,
            f"AST Compliance: {path.relative_to(ROOT_DIR)}",
            len(violations) == 0,
            "0 forbidden calls" if not violations else f"Violations: {', '.join(violations)}",
        )


# =====================================================================
# Test 4: Sigmoid Stability & Bonus 3 L2 Regularization Verification
# =====================================================================
def test_optimizers_and_l2(results: list[tuple[str, bool, str]]) -> None:
    banner("TEST 4: Sigmoid Stability & Bonus 3 L2 Regularization (Ridge) Mechanics")

    # 1. Sigmoid overflow protection on extreme values
    extreme_z = np.array([-1000.0, 0.0, 1000.0])
    sig_out = opt_utils.sigmoid(extreme_z)
    sig_ok = (
        not np.isnan(sig_out).any()
        and np.isclose(sig_out[0], 0.0)
        and np.isclose(sig_out[1], 0.5)
        and np.isclose(sig_out[2], 1.0)
    )
    record_result(results, "Sigmoid Overflow Protection (z = +/-1000)", sig_ok, f"output = {sig_out.tolist()}")

    # 2. L2 penalty function check: (lambda / (2 * m)) * sum(w^2)
    w_test = np.array([3.0, 4.0])  # sum(w^2) = 25
    pen = opt_utils.l2_penalty(w_test, l2_lambda=2.0, m_samples=10)
    expected_pen = (2.0 / (2.0 * 10)) * 25.0  # 2.5
    record_result(
        results,
        "Bonus 3 l2_penalty() Formula Check",
        np.isclose(pen, expected_pen),
        f"computed = {pen:.4f}, expected = {expected_pen:.4f}",
    )

    # 3. Verify L2 Weight Decay shrinks feature weights w while leaving Bias b unpenalized at step 1
    rng = np.random.default_rng(42)
    synth_X = rng.normal(0, 1, size=(100, 3))
    synth_y = (synth_X[:, 0] - 0.5 * synth_X[:, 1] > 0).astype(int)
    synth_df = pd.DataFrame(synth_X, columns=["f1", "f2", "f3"])
    for h in HOUSES:
        synth_df[h] = synth_y

    init_w = pd.DataFrame(2.0, index=["f1", "f2", "f3"], columns=HOUSES)
    init_b = pd.Series(5.0, index=HOUSES)

    # 1 epoch step comparison: with 1 step from identical non-zero initialization,
    # b update must be 100% identical regardless of l2_lambda, while w must shrink more with l2_lambda > 0
    w_no_reg, b_no_reg = opt_utils.batch_gd(synth_df, init_w, init_b, learning_rate=0.1, epochs=1, l2_lambda=0.0)
    w_l2_reg, b_l2_reg = opt_utils.batch_gd(synth_df, init_w, init_b, learning_rate=0.1, epochs=1, l2_lambda=50.0)

    bias_unpenalized = np.allclose(b_no_reg, b_l2_reg, atol=1e-12)
    norm_no_reg = float(np.linalg.norm(w_no_reg.to_numpy()))
    norm_l2_reg = float(np.linalg.norm(w_l2_reg.to_numpy()))
    weight_decayed = norm_l2_reg < norm_no_reg

    record_result(
        results,
        "Bonus 3 Bias Exclusion (theta_0 unpenalized by L2)",
        bias_unpenalized,
        f"max |b_l2 - b_0| = {float((b_l2_reg - b_no_reg).abs().max()):.2e}",
    )
    record_result(
        results,
        "Bonus 3 Weight Decay (||w||_2 shrinks when lambda > 0)",
        weight_decayed,
        f"||w||_2 (lambda=0) = {norm_no_reg:.4f} > ||w||_2 (lambda=50) = {norm_l2_reg:.4f}",
    )


# =====================================================================
# Test 5: End-to-End Optimizer Benchmark, houses.csv & Accuracy (>= 98%)
# =====================================================================
def test_end_to_end_and_accuracy(results: list[tuple[str, bool, str]]) -> None:
    banner("TEST 5: End-to-End CLI Pipeline, Optimizer Benchmark & Accuracy (>= 98%)")

    # Backup existing workspace output files so testing is non-destructive
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        backups = {}
        for fname in ("modele.csv", "model_weights.csv", "houses.csv"):
            fpath = ROOT_DIR / fname
            if fpath.exists():
                bpath = tmp_path / fname
                shutil.copy2(fpath, bpath)
                backups[fpath] = bpath

        try:
            train_df = pd.read_csv(DATA_TRAIN)
            y_true_train = train_df["Hogwarts House"]

            optimizers = ["bgd", "sgd", "mini", "momentum", "adam"]
            print(f"\n  {'Optimizer':<12} | {'L2 Lambda':<10} | {'Train Acc (%)':<15} | {'Status (>= 98%)'}")
            print(f"  {'-' * 60}")

            for opt_name in optimizers:
                cmd_train = [
                    sys.executable,
                    str(SRC_DIR / "logreg_train.py"),
                    str(DATA_TRAIN),
                    "--opt",
                    opt_name,
                ]
                res_train = subprocess.run(cmd_train, cwd=str(ROOT_DIR), capture_output=True, text=True)
                if res_train.returncode != 0:
                    record_result(results, f"CLI logreg_train.py --opt {opt_name}", False, res_train.stderr.strip())
                    continue

                cmd_pred = [
                    sys.executable,
                    str(SRC_DIR / "logreg_predict.py"),
                    str(DATA_TRAIN),
                    str(ROOT_DIR / "modele.csv"),
                ]
                res_pred = subprocess.run(cmd_pred, cwd=str(ROOT_DIR), capture_output=True, text=True)
                if res_pred.returncode != 0:
                    record_result(results, f"CLI logreg_predict.py ({opt_name})", False, res_pred.stderr.strip())
                    continue

                preds_df = pd.read_csv(ROOT_DIR / "houses.csv")
                acc = accuracy_score(y_true_train, preds_df["Hogwarts House"]) * 100.0
                passed_98 = acc >= 98.0
                badge = f"{GREEN}PASS{RESET}" if passed_98 else f"{YELLOW}WARN (< 98%){RESET}"
                print(f"  {opt_name:<12} | {0.0:<10.2f} | {acc:<15.2f} | {badge}")
                record_result(
                    results,
                    f"Optimizer '{opt_name}' Training Accuracy >= 98.0%",
                    passed_98,
                    f"Accuracy = {acc:.2f}%",
                )

            # Test CLI with --lambda 0.01 on default BGD and print detailed confusion matrix
            subprocess.run(
                [sys.executable, str(SRC_DIR / "logreg_train.py"), str(DATA_TRAIN), "--opt", "bgd", "--lambda", "0.01"],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [sys.executable, str(SRC_DIR / "logreg_predict.py"), str(DATA_TRAIN), str(ROOT_DIR / "modele.csv")],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            preds_l2 = pd.read_csv(ROOT_DIR / "houses.csv")["Hogwarts House"]
            acc_l2 = accuracy_score(y_true_train, preds_l2) * 100.0
            record_result(
                results,
                "CLI logreg_train.py --opt bgd --lambda 0.01",
                acc_l2 >= 98.0,
                f"Accuracy = {acc_l2:.2f}%",
            )

            print(f"\n  {BOLD}Classification Report (BGD on dataset_train.csv):{RESET}")
            report = classification_report(y_true_train, preds_l2, labels=list(HOUSES), digits=4)
            for line in report.splitlines():
                print(f"    {line}")

            print(f"\n  {BOLD}Confusion Matrix (Rows: True House, Cols: Predicted House):{RESET}")
            cm = confusion_matrix(y_true_train, preds_l2, labels=list(HOUSES))
            cm_df = pd.DataFrame(cm, index=[f"True_{h[:4]}" for h in HOUSES], columns=[f"Pred_{h[:4]}" for h in HOUSES])
            for line in str(cm_df).splitlines():
                print(f"    {line}")

            # Finally, test prediction on official dataset_test.csv and validate houses.csv structure
            subprocess.run(
                [sys.executable, str(SRC_DIR / "logreg_train.py"), str(DATA_TRAIN)],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            subprocess.run(
                [sys.executable, str(SRC_DIR / "logreg_predict.py"), str(DATA_TEST), str(ROOT_DIR / "modele.csv")],
                cwd=str(ROOT_DIR),
                capture_output=True,
                text=True,
                check=True,
            )
            test_houses = pd.read_csv(ROOT_DIR / "houses.csv")
            test_df = pd.read_csv(DATA_TEST)
            format_ok = (
                list(test_houses.columns) == ["Index", "Hogwarts House"]
                and len(test_houses) == len(test_df)
                and not test_houses["Hogwarts House"].isna().any()
                and set(test_houses["Hogwarts House"].unique()).issubset(set(HOUSES))
                and (test_houses["Index"] == test_df["Index"]).all()
            )
            record_result(
                results,
                "houses.csv Format & Integrity on dataset_test.csv",
                format_ok,
                f"rows = {len(test_houses)}, columns = {list(test_houses.columns)}, NaN = 0",
            )

        finally:
            # Restore original workspace files
            for orig_path, bpath in backups.items():
                shutil.copy2(bpath, orig_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="DSLR Automated Verification & Defense Test Suite")
    parser.add_argument(
        "--only",
        choices=["describe", "stats", "ast", "l2", "e2e"],
        default=None,
        help="Run only a specific test group",
    )
    args = parser.parse_args()

    results: list[tuple[str, bool, str]] = []

    if args.only in (None, "describe"):
        test_describe_vs_pandas(results)
    if args.only in (None, "stats"):
        test_bonus_stats(results)
    if args.only in (None, "ast"):
        test_no_forbidden_builtins(results)
    if args.only in (None, "l2"):
        test_optimizers_and_l2(results)
    if args.only in (None, "e2e"):
        test_end_to_end_and_accuracy(results)

    banner("FINAL TEST SUMMARY")
    passed_count = sum(1 for _, ok, _ in results if ok)
    total_count = len(results)
    for name, ok, detail in results:
        mark = f"{GREEN}✔{RESET}" if ok else f"{RED}✘{RESET}"
        print(f"  {mark} {name}" + (f" ({detail})" if detail else ""))

    print(f"\n  {BOLD}Total Passed: {passed_count} / {total_count}{RESET}")
    if passed_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
