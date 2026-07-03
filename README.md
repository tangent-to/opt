# tangent/opt

Declarative numerical optimization for JavaScript (ESM). Browser-first, zero
dependencies, runs in Node.js and Deno. Companion package to
[tangent/ds](https://github.com/tangent-to/ds).

- **Derivative-free**: Nelder-Mead downhill simplex
- **Gradient-based**: gradient descent (optional backtracking line search),
  momentum, RMSProp, Adam
- **Gradients optional**: pass an analytic gradient, return
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

// Gradient-based
const fit = minimize({
  f: (x) => x[0] ** 2 + x[1] ** 2,
  grad: (x) => [2 * x[0], 2 * x[1]], // optional; finite differences otherwise
  x0: [3, -2],
  method: 'adam',
  learningRate: 0.1,
  maxIter: 5000,
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

## Roadmap

- `opt/minimize`: L-BFGS, conjugate gradient (Wolfe line search)
- `opt/scalar`: Brent minimization, golden section, bisection root-finding
- `opt/leastsq`: Levenberg-Marquardt, `curveFit`
- Bounds via projection

## License

MIT. The Nelder-Mead implementation is ported from
[fmin](https://github.com/benfred/fmin) (BSD-3-Clause) — see
[THIRD_PARTY_NOTICES.md](./THIRD_PARTY_NOTICES.md).
