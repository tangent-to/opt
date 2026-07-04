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
from scipy.optimize import (
    approx_fprime,
    brentq,
    curve_fit as sp_curve_fit,
    minimize as sp_minimize,
    minimize_scalar as sp_minimize_scalar,
)

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


def compare_lbfgs():
    """opt L-BFGS vs scipy L-BFGS-B: same minima, comparable cost."""
    print("\n== L-BFGS vs scipy.optimize.minimize(method='L-BFGS-B') ==")
    cases = [
        ("rosenbrock", [-1.2, 1.0], True),
        ("rosenbrock", [-1.2, 1.0], False),  # finite-difference gradients on both sides
        ("sphere", list(np.tile([3.0, -2.0], 50)), True),  # 100-dimensional
        ("beale", [1.0, 1.0], False),
        ("booth", [0.0, 0.0], False),
        ("himmelblau", [4.0, 3.0], False),
    ]
    grads = {"sphere": sphere_grad, "rosenbrock": rosenbrock_grad}
    for name, x0, use_grad in cases:
        js = run_node({
            "function": name, "x0": x0, "method": "lbfgs",
            "grad": use_grad, "options": {},
        })
        sp = sp_minimize(
            PY_FUNCS[name], x0,
            jac=grads[name] if use_grad else None,
            method="L-BFGS-B", options={"gtol": 1e-10, "ftol": 1e-15},
        )
        dx = float(np.max(np.abs(np.asarray(js["x"]) - sp.x)))
        tol = 1e-4 if use_grad else 1e-3
        check(
            f"{name} n={len(x0)} grad={'analytic' if use_grad else 'numeric'}",
            dx < tol and js["converged"],
            f"|x_js - x_scipy|_max={dx:.2e}, iters js={js['iterations']} scipy={sp.nit}",
        )


SCALAR_PY = {
    "cos": np.cos,
    "quartic": lambda x: (x - 2) ** 4,
    "cubic": lambda x: x ** 3 - 2 * x - 5,
    "logroot": lambda x: np.log(x) - 1,
    "shifted_parabola": lambda x: (x - 7.3) ** 2 + 1,
}


def compare_scalar():
    """minimizeScalar vs scipy minimize_scalar; rootScalar vs scipy brentq."""
    print("\n== minimizeScalar vs scipy.optimize.minimize_scalar(method='brent') ==")
    min_cases = [
        ("cos", {"bracket": [2.0, 4.0]}, np.pi),
        ("quartic", {"bracket": [0.0, 1.0]}, 2.0),
        ("shifted_parabola", {}, 7.3),  # auto-bracketing from the default [0, 1]
    ]
    for name, options, x_true in min_cases:
        js = run_node({"mode": "minimize_scalar", "function": name, "options": options})
        bracket = tuple(options.get("bracket", (0, 1)))
        sp = sp_minimize_scalar(SCALAR_PY[name], bracket=bracket, method="brent")
        dx = abs(js["x"] - sp.x)
        # quartic has a flat minimum: agreement is limited by conditioning
        tol = 1e-2 if name == "quartic" else 1e-6
        check(f"{name} (min at {x_true:.4g})", dx < tol and abs(js["x"] - x_true) < tol,
              f"|x_js - x_scipy|={dx:.2e}, fevals js={js['fevals']}")

    print("\n== rootScalar vs scipy.optimize.brentq ==")
    root_cases = [
        ("cos", [1.0, 2.0], np.pi / 2),
        ("cubic", [2.0, 3.0], 2.0945514815423265),
        ("logroot", [0.1, 10.0], np.e),
    ]
    for name, bracket, x_true in root_cases:
        js = run_node({"mode": "root_scalar", "function": name, "options": {"bracket": bracket}})
        sp_x = brentq(SCALAR_PY[name], *bracket, xtol=1e-12)
        dx = abs(js["x"] - sp_x)
        check(f"{name} root in {bracket}", dx < 1e-10 and abs(js["x"] - x_true) < 1e-9,
              f"|x_js - x_scipy|={dx:.2e}")


def compare_curve_fit():
    """curveFit vs scipy.optimize.curve_fit: parameters AND standard errors."""
    print("\n== curveFit vs scipy.optimize.curve_fit ==")
    rng = np.random.default_rng(42)

    # Exponential decay with noise
    x = np.linspace(0, 4, 81)
    y_true = 2.5 * np.exp(-1.3 * x) + 0.5
    y = y_true + 0.02 * rng.standard_normal(x.size)
    js = run_node({
        "mode": "curve_fit", "model": "exp_decay",
        "x": x.tolist(), "y": y.tolist(), "p0": [1.0, 1.0, 0.0],
    })
    popt, pcov = sp_curve_fit(
        lambda t, a, b, c: a * np.exp(-b * t) + c, x, y, p0=[1.0, 1.0, 0.0],
    )
    dp = float(np.max(np.abs(np.asarray(js["params"]) - popt)))
    dse = float(np.max(np.abs(np.asarray(js["stdErr"]) - np.sqrt(np.diag(pcov)))))
    check("exp decay params", dp < 1e-5, f"|p_js - p_scipy|_max={dp:.2e}")
    check("exp decay std errors", dse < 1e-4, f"|se_js - se_scipy|_max={dse:.2e}")

    # Sigmoid (harder: correlated parameters)
    x2 = np.linspace(-2, 8, 101)
    y2_true = 3.0 / (1 + np.exp(-1.5 * (x2 - 2.0)))
    y2 = y2_true + 0.03 * rng.standard_normal(x2.size)
    js2 = run_node({
        "mode": "curve_fit", "model": "sigmoid",
        "x": x2.tolist(), "y": y2.tolist(), "p0": [1.0, 1.0, 1.0],
    })
    popt2, _ = sp_curve_fit(
        lambda t, a, b, c: a / (1 + np.exp(-b * (t - c))), x2, y2, p0=[1.0, 1.0, 1.0],
    )
    dp2 = float(np.max(np.abs(np.asarray(js2["params"]) - popt2)))
    check("sigmoid params", dp2 < 1e-4, f"|p_js - p_scipy|_max={dp2:.2e}")


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
    compare_lbfgs()
    compare_scalar()
    compare_curve_fit()
    compare_gradient_methods()
    compare_numerical_gradient()
    print(f"\n{len(FAILURES)} failure(s)" if FAILURES else "\nAll comparisons passed.")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
