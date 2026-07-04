# tangent/opt

Declarative numerical optimization for JavaScript (ESM). Browser-first, zero
dependencies, runs in Node.js and Deno. Companion package to
[tangent/ds](https://github.com/tangent-to/ds).

- **Quasi-Newton**: L-BFGS with strong Wolfe line search
- **Derivative-free**: Nelder-Mead downhill simplex
- **Gradient-based**: gradient descent (optional backtracking line search),
  momentum, RMSProp, Adam
- **Scalar**: Brent minimization and golden section with auto-bracketing;
  Brent-Dekker and bisection root-finding
- **Least squares**: Levenberg-Marquardt and scipy-style `curveFit` with
  covariance / standard errors
- **Gradients optional**: pass an analytic gradient or Jacobian, return
  `{loss, gradient}` from your objective, or let central finite differences
  fill in

## Install

```bash
npm install @tangent.to/opt     # npm
deno add jsr:@tangent/opt       # Deno / JSR
```

## Usage

The declarative entry point takes a single spec object:

```javascript
import { minimize } from '@tangent.to/opt';

// Derivative-free (default method: Nelder-Mead)
const result = minimize({
  f: ([a, b]) => (1 - a) ** 2 + 100 * (b - a * a) ** 2,
  x0: [-1.2, 1],
});
// { x: [1, 1], fx: ~0, iterations, fevals, converged: true, method: 'neldermead' }

// Quasi-Newton (use this by default for smooth objectives)
const fit = minimize({
  f: (x) => x[0] ** 2 + x[1] ** 2,
  grad: (x) => [2 * x[0], 2 * x[1]], // optional; finite differences otherwise
  x0: [3, -2],
  method: 'lbfgs',
});
```

Scalar optimization, root-finding and curve fitting:

```javascript
import { curveFit, minimizeScalar, rootScalar } from '@tangent.to/opt';

minimizeScalar(Math.cos, { bracket: [2, 4] });          // x ≈ π
rootScalar(Math.cos, { bracket: [1, 2] });              // x ≈ π/2

const { params, stdErr } = curveFit({
  model: (x, [a, b, c]) => a * Math.exp(-b * x) + c,
  x: xdata,
  y: ydata,
  p0: [1, 1, 0],
});
```

Each method is also exported directly:

```javascript
import { adam, gradientDescent, nelderMead } from '@tangent.to/opt';

const r = nelderMead(f, x0, { maxIter: 2000, history: true });
```

### Result shape

All methods return:

| key          | description                                    |
| ------------ | ---------------------------------------------- |
| `x`          | best parameters found                          |
| `fx`         | objective value at `x`                         |
| `iterations` | iterations used                                |
| `converged`  | whether a tolerance criterion was met          |
| `history`    | per-iteration record (see each method's JSDoc) |

### tangent/ds compatibility

The classes formerly in `ds.core.optimize` are exported unchanged
(`GradientDescent`, `MomentumOptimizer`, `RMSProp`, `AdamOptimizer`,
`createOptimizer`), so `tangent/ds` re-exports this package as
`ds.core.optimize`:

```javascript
import { createOptimizer } from '@tangent.to/opt';

const opt = createOptimizer('adam', { learningRate: 0.05 });
const { x, history } = opt.minimize((x) => ({ loss, gradient }), x0);
```

## Validation against scipy

`tests_compare-to-scipy/` cross-checks every method against `scipy.optimize`:
Nelder-Mead vs scipy's Nelder-Mead, L-BFGS vs `L-BFGS-B` (matching iteration
counts), `minimizeScalar` vs `minimize_scalar`, `rootScalar` vs `brentq`
(1e-10 agreement), `curveFit` vs `curve_fit` (parameters *and* standard
errors to ~1e-9), gradient methods vs the BFGS reference optimum, and
`numericalGradient` vs `approx_fprime`. Requires
[uv](https://docs.astral.sh/uv/) and Node:

```bash
npm run test:scipy
```

## Roadmap

- Bounds (L-BFGS-B-style projection, box constraints for `leastSquares`)
- Conjugate gradient
- Global methods (differential evolution, basin hopping)

## License

MIT. The Nelder-Mead implementation is ported from
[fmin](https://github.com/benfred/fmin) (BSD-3-Clause) — see
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
