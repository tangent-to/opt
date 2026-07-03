#!/usr/bin/env python3
"""
Compare @tangent.to/opt minimizers (via Node) against scipy.optimize on
standard test functions.

Run from the package root:

    uv run --with scipy python3 tests_compare-to-scipy/compare_with_scipy.py

Both sides solve the same problems from the same starting points; we compare
the minima found (x, fx), not trajectories -- implementations differ in
simplex/step details, so agreement at the optimum is the meaningful check.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import approx_fprime, minimize as sp_minimize

ROOT = Path(__file__).resolve().parents[1]
NODE_SCRIPT = ROOT / "tests_compare-to-scipy" / "compare_opt.mjs"


def rosenbrock(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
        200 * (x[1] - x[0] ** 2),
    ])


def sphere(x):
    return float(np.sum(np.asarray(x) ** 2))


def sphere_grad(x):
    return 2 * np.asarray(x)


def booth(x):
    return (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2


def himmelblau(x):
    return (x[0] ** 2 + x[1] - 11) ** 2 + (x[0] + x[1] ** 2 - 7) ** 2


def beale(x):
    return (
        (1.5 - x[0] + x[0] * x[1]) ** 2
        + (2.25 - x[0] + x[0] * x[1] ** 2) ** 2
        + (2.625 - x[0] + x[0] * x[1] ** 3) ** 2
    )


PY_FUNCS = {
    "sphere": sphere,
    "rosenbrock": rosenbrock,
    "booth": booth,
    "himmelblau": himmelblau,
    "beale": beale,
}


def run_node(spec: dict) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(spec, fh)
        temp_path = fh.name
    result = subprocess.run(
        ["node", str(NODE_SCRIPT), temp_path],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return json.loads(result.stdout)


FAILURES = []


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(label)


def compare_neldermead():
    """opt Nelder-Mead vs scipy Nelder-Mead: same minima on standard problems."""
    print("\n== Nelder-Mead vs scipy.optimize.minimize(method='Nelder-Mead') ==")
    cases = [
        ("rosenbrock", [-1.2, 1.0], {"maxIter": 5000}),
        ("booth", [0.0, 0.0], {}),
        ("himmelblau", [4.0, 3.0], {}),
        ("beale", [1.0, 1.0], {"maxIter": 5000}),
        ("sphere", [3.0, -2.0, 1.5], {}),
    ]
    for name, x0, options in cases:
        js = run_node({"function": name, "x0": x0, "method": "neldermead", "options": options})
        sp = sp_minimize(PY_FUNCS[name], x0, method="Nelder-Mead",
                         options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 5000})
        dx = float(np.max(np.abs(np.asarray(js["x"]) - sp.x)))
        dfx = abs(js["fx"] - sp.fun)
        check(
            f"{name} from {x0}",
            dx < 1e-3 and dfx < 1e-6,
            f"|x_js - x_scipy|_max={dx:.2e}, |fx_js - fx_scipy|={dfx:.2e}",
        )


def compare_gradient_methods():
    """opt gradient methods vs scipy reference optimum (BFGS, tight tolerance)."""
    print("\n== Gradient methods vs scipy reference optimum (BFGS) ==")
    cases = [
        ("adam", "sphere", [3.0, -2.0], {"learningRate": 0.1, "maxIter": 10000}, 1e-4),
        ("gd", "sphere", [3.0, -2.0], {"learningRate": 0.1, "maxIter": 10000}, 1e-4),
        ("momentum", "sphere", [3.0, -2.0], {"learningRate": 0.05, "maxIter": 10000}, 1e-4),
        ("rmsprop", "sphere", [3.0, -2.0], {"learningRate": 0.05, "maxIter": 10000}, 1e-3),
        ("adam", "rosenbrock", [-1.2, 1.0], {"learningRate": 0.01, "maxIter": 50000}, 1e-2),
    ]
    grads = {"sphere": sphere_grad, "rosenbrock": rosenbrock_grad}
    for method, name, x0, options, tol in cases:
        js = run_node({
            "function": name, "x0": x0, "method": method,
            "grad": True, "options": options,
        })
        sp = sp_minimize(PY_FUNCS[name], x0, jac=grads[name], method="BFGS",
                         options={"gtol": 1e-10})
        dx = float(np.max(np.abs(np.asarray(js["x"]) - sp.x)))
        check(
            f"{method} on {name}",
            dx < tol,
            f"|x_js - x_scipy|_max={dx:.2e} (tol {tol:.0e})",
        )


def compare_numerical_gradient():
    """opt numericalGradient vs scipy.optimize.approx_fprime and analytic."""
    print("\n== numericalGradient vs scipy.optimize.approx_fprime ==")
    points = {
        "rosenbrock": [-1.2, 1.0],
        "sphere": [1.0, -2.0, 3.0],
        "beale": [1.0, 1.0],
    }
    for name, x in points.items():
        js = run_node({"function": name, "x0": x, "mode": "gradient"})
        sp_grad = approx_fprime(np.asarray(x), PY_FUNCS[name], 1.5e-8)
        diff = float(np.max(np.abs(np.asarray(js["gradient"]) - sp_grad)))
        # both are finite-difference approximations; agree to ~1e-4 on these scales
        check(f"{name} at {x}", diff < 1e-4, f"max diff vs approx_fprime={diff:.2e}")


def main():
    print(f"scipy comparison for @tangent.to/opt  (node {ROOT})")
    compare_neldermead()
    compare_gradient_methods()
    compare_numerical_gradient()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
