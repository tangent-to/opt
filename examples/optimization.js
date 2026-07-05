// ---
// title: Numerical optimization
// id: opt-optimization
// ---

// %% [markdown]
/*
# Numerical optimization

`@tangent.to/opt` is a small, dependency-free optimization toolkit for
JavaScript: function minimization in one or many variables, nonlinear least
squares (curve fitting), and scalar root finding. Every routine follows the
same declarative options-object style and returns a plain result object, so
the same mental model carries across methods.

This notebook imports the local build. Once the package is published you would
import it from a CDN instead:

    import { minimize, curveFit, rootScalar } from 'https://esm.sh/@tangent.to/opt';
*/

// %% [javascript]

import { minimize, curveFit, rootScalar, minimizeScalar } from '../dist/index.js';

// The default method is Nelder-Mead, a derivative-free simplex search. It only
// needs the objective and a starting point, which makes it the safe first
// choice when you have no gradient.
minimize({
  f: (x) => (x[0] - 1) ** 2 + (x[1] - 3) ** 2,
  x0: [0, 0],
}).x; // near [1, 3]

// %% [markdown]
/*
## Minimizing a 2D function

The Booth function has a single minimum of 0 at [1, 3]. With no gradient
supplied, `minimize` runs Nelder-Mead, which reflects, expands, and contracts a
simplex of points downhill. The result object reports the located point `x`,
the objective value `fx`, the iteration count, and whether the convergence test
was met.
*/

// %% [javascript]

const booth = (x) =>
  (x[0] + 2 * x[1] - 7) ** 2 + (2 * x[0] + x[1] - 5) ** 2;

const boothResult = minimize({ f: booth, x0: [0, 0] });

({
  x: boothResult.x,
  fx: boothResult.fx,
  iterations: boothResult.iterations,
  converged: boothResult.converged,
  method: boothResult.method,
});

// %% [markdown]
/*
## Quasi-Newton with a gradient

When the objective is smooth and you can supply its gradient, L-BFGS is the
modern default. It builds a low-memory approximation of the curvature from the
history of gradients, so it converges far faster than a simplex on well-behaved
problems. Here it minimizes the Rosenbrock function from the classic hard start
[-1.2, 1], threading up the curved valley to the minimum at [1, 1].
*/

// %% [javascript]

const rosenbrock = (x) =>
  100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2;

const rosenbrockGrad = (x) => [
  -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0]),
  200 * (x[1] - x[0] ** 2),
];

const rosen = minimize({
  f: rosenbrock,
  grad: rosenbrockGrad,
  x0: [-1.2, 1],
  method: 'lbfgs',
});

({
  x: rosen.x,
  fx: rosen.fx,
  iterations: rosen.iterations,
  converged: rosen.converged,
});

// %% [markdown]
/*
## Curve fitting with uncertainties

`curveFit` solves a nonlinear least squares problem: given data `(x, y)` and a
model, it recovers the parameters that best fit the data (Levenberg-Marquardt).
Below we fit an exponential decay `a * exp(-b * x) + c` to synthetic data
generated from known parameters plus a little noise. Alongside the fitted
`params`, it returns `stdErr`, the one-sigma standard errors from the estimated
covariance, so you can judge how well each parameter is pinned down.
*/

// %% [javascript]

const model = (x, [a, b, c]) => a * Math.exp(-b * x) + c;

// Synthetic data: true params a = 5, b = 1.5, c = 0.5, with small
// deterministic noise so the notebook is reproducible.
const xs = Array.from({ length: 40 }, (_, i) => (i * 4) / 39);
const ys = xs.map(
  (x, i) => model(x, [5, 1.5, 0.5]) + 0.05 * Math.sin(7 * i),
);

const fit = curveFit({ model, x: xs, y: ys, p0: [1, 1, 0] });

({
  params: fit.params, // close to [5, 1.5, 0.5]
  stdErr: fit.stdErr,
  converged: fit.converged,
});

// %% [markdown]
/*
## Robust fitting against outliers

A plain least squares fit chases every point, so a few gross outliers can drag
the whole curve. Passing `loss: 'huber'` down-weights large residuals, keeping
the fit close to the bulk of the data. Here we corrupt three points and compare
the ordinary fit to the robust one; the robust `b` stays much closer to the
true value of 1.5.
*/

// %% [javascript]

const yOutliers = ys.slice();
yOutliers[10] = 20; // three gross outliers
yOutliers[20] = 18;
yOutliers[30] = 15;

const plainFit = curveFit({ model, x: xs, y: yOutliers, p0: [1, 1, 0] });
const robustFit = curveFit({
  model,
  x: xs,
  y: yOutliers,
  p0: [1, 1, 0],
  loss: 'huber',
  fScale: 0.5,
});

({
  plain_b: plainFit.params[1],
  robust_b: robustFit.params[1],
  true_b: 1.5,
});

// %% [markdown]
/*
## Scalar root finding and minimization

For one-dimensional problems the scalar routines are exact and fast.
`rootScalar` uses Brent-Dekker on a sign-changing bracket: cos crosses zero at
pi/2 inside [1, 2]. `minimizeScalar` uses Brent's method to locate a minimum;
the parabola `(x - 2)^2 + 1` bottoms out at x = 2.
*/

// %% [javascript]

const root = rootScalar((x) => Math.cos(x), { bracket: [1, 2] });
const minS = minimizeScalar((x) => (x - 2) ** 2 + 1, { bracket: [0, 5] });

({
  root_x: root.x, // pi / 2
  pi_over_2: Math.PI / 2,
  min_x: minS.x, // 2
  min_fx: minS.fx, // 1
});
