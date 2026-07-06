// ---
// title: Numerical optimization
// id: opt-optimization
// ---

// %% [markdown]
/*
`@tangent.to/opt` is a small, dependency-free optimization toolkit for
JavaScript: function minimization in one or many variables, nonlinear least
squares (curve fitting), and scalar root finding. Every routine follows the
same declarative options-object style and returns a plain result object, so
the same mental model carries across methods.
*/

// %% [javascript]

import * as __lib from 'https://esm.sh/@tangent.to/opt';
const minimize = __lib.minimize;
const curveFit = __lib.curveFit;
const rootScalar = __lib.rootScalar;
const minimizeScalar = __lib.minimizeScalar;

// The default method is Nelder-Mead, a derivative-free simplex search. It only
// needs the objective and a starting point, which makes it the safe first
// choice when you have no gradient.
minimize({
  f: (x) => (x[0] - 1) ** 2 + (x[1] - 3) ** 2,
  x0: [0, 0],
}).x; // near [1, 3]

// %% [javascript]

// Small helper used below: sample a 1-D slice of an objective through the
// located optimum along one coordinate, and mark the minimum with a dot.
// (Plot and d3 are preloaded globals in note.tangent.to, so no import.)
const objectiveSlice = (f, min, axis, [lo, hi]) => {
  const n = 200;
  const pts = Array.from({ length: n + 1 }, (_, i) => {
    const t = lo + ((hi - lo) * i) / n;
    const p = min.slice();
    p[axis] = t;
    return { x: t, y: f(p) };
  });
  return Plot.plot({
    width: 640,
    height: 320,
    x: { label: `x[${axis}]  (other coordinates fixed at the optimum)` },
    y: { label: 'f(x)' },
    marks: [
      Plot.line(pts, { x: 'x', y: 'y', stroke: 'steelblue' }),
      Plot.dot([{ x: min[axis], y: f(min) }], {
        x: 'x',
        y: 'y',
        fill: 'red',
        r: 5,
      }),
    ],
  });
};

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
A 1-D slice of the Booth function through the located optimum along x[0]; the
red dot marks the minimum `minimize` found.
*/

// %% [javascript]

objectiveSlice(booth, boothResult.x, 0, [-4, 6]);

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
A slice of the Rosenbrock function through the solution along x[0]; the red dot
sits at the minimum near [1, 1] that L-BFGS threaded up the curved valley to.
*/

// %% [javascript]

objectiveSlice(rosenbrock, rosen.x, 0, [-1.5, 1.5]);

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
The key visual: the noisy data points (blue) overlaid with the fitted
exponential decay `a·exp(-b·x) + c` drawn as a smooth red curve over the x-range.
*/

// %% [javascript]

const fitCurve = Array.from({ length: 200 }, (_, i) => {
  const x = (i * 4) / 199;
  return { x, y: model(x, fit.params) };
});

Plot.plot({
  width: 640,
  height: 340,
  x: { label: 'x' },
  y: { label: 'y' },
  marks: [
    Plot.dot(
      xs.map((x, i) => ({ x, y: ys[i] })),
      { x: 'x', y: 'y', fill: 'steelblue', r: 3 },
    ),
    Plot.line(fitCurve, { x: 'x', y: 'y', stroke: 'crimson', strokeWidth: 2 }),
  ],
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
Data with three gross outliers (gray). The ordinary least-squares fit (orange)
is dragged upward toward them, while the Huber-robust fit (red) stays with the
bulk of the data.
*/

// %% [javascript]

const plainCurve = Array.from({ length: 200 }, (_, i) => {
  const x = (i * 4) / 199;
  return { x, y: model(x, plainFit.params) };
});
const robustCurve = Array.from({ length: 200 }, (_, i) => {
  const x = (i * 4) / 199;
  return { x, y: model(x, robustFit.params) };
});

Plot.plot({
  width: 640,
  height: 340,
  x: { label: 'x' },
  y: { label: 'y' },
  marks: [
    Plot.dot(
      xs.map((x, i) => ({ x, y: yOutliers[i] })),
      { x: 'x', y: 'y', fill: '#888', r: 3 },
    ),
    Plot.line(plainCurve, { x: 'x', y: 'y', stroke: 'orange', strokeWidth: 2 }),
    Plot.line(robustCurve, { x: 'x', y: 'y', stroke: 'crimson', strokeWidth: 2 }),
  ],
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

// %% [markdown]
/*
`rootScalar` on cos over [0.5, 2.5]: the red dot sits where the blue curve
crosses zero (the horizontal rule), at the root π/2 that Brent-Dekker returned.
*/

// %% [javascript]

const cosPts = Array.from({ length: 200 }, (_, i) => {
  const x = 0.5 + ((2.5 - 0.5) * i) / 199;
  return { x, y: Math.cos(x) };
});

Plot.plot({
  width: 640,
  height: 300,
  x: { label: 'x' },
  y: { label: 'cos(x)' },
  marks: [
    Plot.ruleY([0]),
    Plot.line(cosPts, { x: 'x', y: 'y', stroke: 'steelblue' }),
    Plot.dot([{ x: root.x, y: 0 }], { x: 'x', y: 'y', fill: 'red', r: 5 }),
  ],
});

// %% [markdown]
/*
`minimizeScalar` on the parabola (x − 2)² + 1 over [0, 5]: the red dot marks the
located minimum at x = 2, f = 1.
*/

// %% [javascript]

const parab = Array.from({ length: 200 }, (_, i) => {
  const x = (5 * i) / 199;
  return { x, y: (x - 2) ** 2 + 1 };
});

Plot.plot({
  width: 640,
  height: 300,
  x: { label: 'x' },
  y: { label: 'f(x)' },
  marks: [
    Plot.line(parab, { x: 'x', y: 'y', stroke: 'steelblue' }),
    Plot.dot([{ x: minS.x, y: minS.fx }], { x: 'x', y: 'y', fill: 'red', r: 5 }),
  ],
});
