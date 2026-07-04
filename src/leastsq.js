/**
 * Nonlinear least squares: Levenberg-Marquardt and a curve_fit-style wrapper.
 *
 * The cost is 0.5 * sum(r(p)^2). Each iteration solves the Marquardt-damped
 * normal equations
 *
 *   (J^T J + lambda * diag(J^T J)) delta = -J^T r
 *
 * accepting steps that reduce the cost (lambda /= lambdaDown) and rejecting
 * the rest (lambda *= lambdaUp, retry). Non-finite trial residuals are
 * treated as rejected steps, never as errors.
 */

import { solve } from './linsolve.js';
import { makeBoundsTransform, wrapResiduals } from './bounds.js';

const FD_STEP = 1e-6;

/**
 * Robust loss functions, scipy.optimize.least_squares-compatible:
 * cost = 0.5 * fScale^2 * sum(rho(z_i)) with z_i = (r_i / fScale)^2.
 * drho is rho'(z), used as the IRLS weight in the normal equations.
 */
const LOSSES = {
  linear: null,
  huber: {
    rho: (z) => (z <= 1 ? z : 2 * Math.sqrt(z) - 1),
    drho: (z) => (z <= 1 ? 1 : 1 / Math.sqrt(z)),
  },
  soft_l1: {
    rho: (z) => 2 * (Math.sqrt(1 + z) - 1),
    drho: (z) => 1 / Math.sqrt(1 + z),
  },
  cauchy: {
    rho: (z) => Math.log1p(z),
    drho: (z) => 1 / (1 + z),
  },
};

/**
 * Central finite-difference Jacobian of a residual function.
 * Step per component: h = FD_STEP * max(1, |p_i|).
 *
 * @param {Function} residuals - (p) => Array<number> of length m
 * @param {Array<number>} p - Point of length n
 * @param {number} m - Number of residuals
 * @returns {Array<Array<number>>} m-by-n Jacobian
 */
function fdJacobian(residuals, p, m) {
  const n = p.length;
  const J = new Array(m);
  for (let j = 0; j < m; j++) {
    J[j] = new Array(n);
  }
  const pi = p.slice();
  for (let i = 0; i < n; i++) {
    const h = FD_STEP * Math.max(1, Math.abs(p[i]));
    pi[i] = p[i] + h;
    const rPlus = residuals(pi);
    pi[i] = p[i] - h;
    const rMinus = residuals(pi);
    pi[i] = p[i];
    for (let j = 0; j < m; j++) {
      J[j][i] = (rPlus[j] - rMinus[j]) / (2 * h);
    }
  }
  return J;
}

/** Cost 0.5 * sum(r^2); NaN if any residual is non-finite. */
function halfSumSquares(r) {
  let sum = 0;
  for (let j = 0; j < r.length; j++) {
    sum += r[j] * r[j];
  }
  return 0.5 * sum;
}

/**
 * Robust cost 0.5 * fScale^2 * sum(rho((r/fScale)^2)); reduces to
 * halfSumSquares for the linear loss. NaN if any residual is non-finite.
 */
function robustCost(r, lossFn, fScale) {
  if (!lossFn) return halfSumSquares(r);
  let sum = 0;
  const s2 = fScale * fScale;
  for (let j = 0; j < r.length; j++) {
    sum += lossFn.rho((r[j] * r[j]) / s2);
  }
  return 0.5 * s2 * sum;
}

/** Per-residual IRLS weights rho'((r/fScale)^2), or null for the linear loss. */
function robustWeights(r, lossFn, fScale) {
  if (!lossFn) return null;
  const w = new Array(r.length);
  const s2 = fScale * fScale;
  for (let j = 0; j < r.length; j++) {
    w[j] = lossFn.drho((r[j] * r[j]) / s2);
  }
  return w;
}

/**
 * Form the (optionally IRLS-weighted) normal-equation pieces
 * J^T W J (n-by-n) and g = J^T W r (length n), W = diag(w).
 * With w = null this is plain J^T J and J^T r; with robust weights, g is
 * exactly the gradient of the robust cost, so the gTol test stays valid.
 *
 * @param {Array<Array<number>>} J - m-by-n Jacobian
 * @param {Array<number>} r - Residuals of length m
 * @param {Array<number>|null} [w] - Per-residual weights
 * @returns {{JTJ: Array<Array<number>>, g: Array<number>}}
 */
function normalEquations(J, r, w = null) {
  const m = J.length;
  const n = J[0].length;
  const JTJ = new Array(n);
  const g = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    JTJ[i] = new Array(n).fill(0);
  }
  for (let k = 0; k < m; k++) {
    const row = J[k];
    const wk = w === null ? 1 : w[k];
    for (let i = 0; i < n; i++) {
      g[i] += wk * row[i] * r[k];
      for (let j = i; j < n; j++) {
        JTJ[i][j] += wk * row[i] * row[j];
      }
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < i; j++) {
      JTJ[i][j] = JTJ[j][i];
    }
  }
  return { JTJ, g };
}

/**
 * Minimize 0.5 * sum(r(p)^2) with Levenberg-Marquardt.
 *
 * @param {Object} spec
 * @param {Function} spec.residuals - (p) => Array<number> of length m
 * @param {Array<number>} spec.x0 - Initial parameters of length n
 * @param {Function} [spec.jacobian] - (p) => m-by-n Array<Array<number>>;
 *   central finite differences on the residuals otherwise
 * @param {number} [spec.maxIter=200] - Maximum accepted iterations
 * @param {number} [spec.fTol=1e-10] - Relative cost reduction on an accepted step
 * @param {number} [spec.xTol=1e-10] - Max relative step component on an accepted step
 * @param {number} [spec.gTol=1e-10] - Inf-norm of the gradient J^T r
 * @param {number} [spec.lambda0=1e-3] - Initial damping
 * @param {number} [spec.lambdaUp=10] - Damping increase factor on rejection
 * @param {number} [spec.lambdaDown=10] - Damping decrease factor on acceptance
 * @param {string} [spec.loss='linear'] - 'linear' | 'huber' | 'soft_l1' | 'cauchy';
 *   robust losses down-weight outliers (IRLS, scipy least_squares semantics)
 * @param {number} [spec.fScale=1] - Residual scale at which the robust losses
 *   start to flatten (scipy's f_scale)
 * @param {Array<Array<number|null>>} [spec.bounds] - Per-parameter [lo, hi] box
 *   bounds (MINUIT transform; null/±Infinity for unbounded sides)
 * @param {boolean} [spec.history=false] - Record {cost, lambda} per accepted iteration
 * @returns {Object} {x, fx, residuals, iterations, fevals, converged, history?}
 */
export function leastSquares(spec = {}) {
  const {
    residuals,
    jacobian,
    x0,
    maxIter = 200,
    fTol = 1e-10,
    xTol = 1e-10,
    gTol = 1e-10,
    lambda0 = 1e-3,
    lambdaUp = 10,
    lambdaDown = 10,
    loss = 'linear',
    fScale = 1,
    bounds,
    history: trackHistory = false,
  } = spec;

  if (typeof residuals !== 'function') {
    throw new Error('leastSquares: spec.residuals must be a function');
  }
  if (!Array.isArray(x0) || x0.length === 0 || x0.some((v) => typeof v !== 'number')) {
    throw new Error('leastSquares: spec.x0 must be a non-empty array of numbers');
  }
  if (!(loss in LOSSES)) {
    throw new Error(
      `leastSquares: unknown loss '${loss}'. Available: ${Object.keys(LOSSES).join(', ')}`,
    );
  }
  const lossFn = LOSSES[loss];

  // Box bounds: solve in MINUIT-transformed internal space, report externally.
  const T = bounds ? makeBoundsTransform(bounds, x0.length) : null;
  if (T) {
    const wrapped = wrapResiduals(residuals, jacobian, T);
    const inner = leastSquares({
      ...spec,
      bounds: undefined,
      residuals: wrapped.residuals,
      jacobian: wrapped.jacobian,
      x0: T.toInternal(x0),
    });
    return { ...inner, x: T.toExternal(inner.x), bounded: true };
  }

  const n = x0.length;
  let fevals = 0;
  const evalResiduals = (p) => {
    fevals++;
    return residuals(p);
  };

  let x = x0.slice();
  let r = evalResiduals(x);
  if (!Array.isArray(r) || r.length === 0) {
    throw new Error('leastSquares: residuals must return a non-empty array of numbers');
  }
  const m = r.length;
  let cost = robustCost(r, lossFn, fScale);
  if (!Number.isFinite(cost)) {
    throw new Error('leastSquares: residuals are not finite at x0');
  }

  const evalJacobian = jacobian
    ? (p) => jacobian(p)
    : (p) => fdJacobian(evalResiduals, p, m);

  let lambda = lambda0;
  let converged = false;
  const history = trackHistory ? [] : null;
  let iteration = 0;

  outer:
  while (iteration < maxIter) {
    const J = evalJacobian(x);
    const { JTJ, g } = normalEquations(J, r, robustWeights(r, lossFn, fScale));

    let gMax = 0;
    for (let i = 0; i < n; i++) {
      gMax = Math.max(gMax, Math.abs(g[i]));
    }
    if (gMax < gTol) {
      converged = true;
      break;
    }

    // Inner loop: retry with increased damping until a step is accepted
    // or the damping safeguard trips.
    for (;;) {
      // A = J^T J + lambda * diag(J^T J); a zero diagonal entry gets
      // lambda itself so the damped system is never singular.
      const A = new Array(n);
      for (let i = 0; i < n; i++) {
        A[i] = JTJ[i].slice();
        A[i][i] = JTJ[i][i] !== 0 ? JTJ[i][i] * (1 + lambda) : lambda;
      }
      const rhs = new Array(n);
      for (let i = 0; i < n; i++) {
        rhs[i] = -g[i];
      }

      let delta = null;
      try {
        delta = solve(A, rhs);
      } catch {
        // Degenerate damped system: treat as a rejected step.
      }

      let trialCost = NaN;
      let xTrial = null;
      let rTrial = null;
      if (delta !== null) {
        xTrial = new Array(n);
        for (let i = 0; i < n; i++) {
          xTrial[i] = x[i] + delta[i];
        }
        rTrial = evalResiduals(xTrial);
        trialCost = robustCost(rTrial, lossFn, fScale);
      }

      if (Number.isFinite(trialCost) && trialCost < cost) {
        // Accepted step.
        const costReduction = (cost - trialCost) / cost;
        let maxRelStep = 0;
        for (let i = 0; i < n; i++) {
          maxRelStep = Math.max(maxRelStep, Math.abs(delta[i]) / Math.max(1, Math.abs(x[i])));
        }
        x = xTrial;
        r = rTrial;
        cost = trialCost;
        iteration++;
        if (trackHistory) {
          history.push({ cost, lambda });
        }
        lambda = Math.max(lambda / lambdaDown, 1e-12);
        if (costReduction < fTol || maxRelStep < xTol) {
          converged = true;
          break outer;
        }
        break;
      }

      // Rejected step (cost increase, or NaN/Infinity residuals).
      lambda *= lambdaUp;
      if (lambda > 1e12) {
        break outer;
      }
    }
  }

  const result = {
    x,
    fx: cost,
    residuals: r.slice(),
    iterations: iteration,
    fevals,
    converged,
  };
  if (trackHistory) {
    result.history = history;
  }
  return result;
}

/**
 * Fit a scalar model to (x, y) data, scipy.optimize.curve_fit style.
 *
 * @param {Object} spec
 * @param {Function} spec.model - (x, params) => number for a scalar datum x
 * @param {Array<number>} spec.x - Independent data
 * @param {Array<number>} spec.y - Dependent data, same length as x
 * @param {Array<number>} spec.p0 - Initial parameters
 * @param {...*} [spec.options] - Remaining keys are passed to leastSquares
 *   (jacobian, maxIter, fTol, xTol, gTol, lambda0, lambdaUp, lambdaDown, history)
 * @returns {Object} {params, cov, stdErr, fx, iterations, converged}
 */
export function curveFit(spec = {}) {
  const { model, x, y, p0, ...options } = spec;

  if (typeof model !== 'function') {
    throw new Error('curveFit: spec.model must be a function');
  }
  if (!Array.isArray(x) || !Array.isArray(y)) {
    throw new Error('curveFit: spec.x and spec.y must be arrays of numbers');
  }
  if (x.length !== y.length) {
    throw new Error(
      `curveFit: x and y must have the same length (got ${x.length} and ${y.length})`,
    );
  }
  if (!Array.isArray(p0) || p0.length === 0 || p0.some((v) => typeof v !== 'number')) {
    throw new Error('curveFit: spec.p0 must be a non-empty array of numbers');
  }

  const m = x.length;
  const residuals = (p) => {
    const r = new Array(m);
    for (let i = 0; i < m; i++) {
      r[i] = y[i] - model(x[i], p);
    }
    return r;
  };

  const result = leastSquares({ residuals, x0: p0, ...options });

  // Covariance estimate: cov = inv(J^T J) * s^2 with s^2 = 2*fx / (m - n),
  // solved column-by-column against the identity. Singular or
  // under-determined systems yield NaN entries instead of throwing.
  // The Gaussian formula is only valid for the linear loss; robust fits
  // report NaN (a sandwich estimator may be added later).
  const n = p0.length;
  const cov = new Array(n);
  const stdErr = new Array(n);
  let filled = false;
  const gaussian = !options.loss || options.loss === 'linear';
  if (m > n && gaussian) {
    const J = options.jacobian ? options.jacobian(result.x) : fdJacobian(residuals, result.x, m);
    const { JTJ } = normalEquations(J, result.residuals);
    const s2 = (2 * result.fx) / (m - n);
    try {
      const columns = new Array(n);
      for (let j = 0; j < n; j++) {
        const e = new Array(n).fill(0);
        e[j] = 1;
        columns[j] = solve(JTJ, e);
      }
      for (let i = 0; i < n; i++) {
        cov[i] = new Array(n);
        for (let j = 0; j < n; j++) {
          cov[i][j] = columns[j][i] * s2;
        }
        stdErr[i] = Math.sqrt(cov[i][i]);
      }
      filled = true;
    } catch {
      // Singular J^T J: fall through to the NaN fill.
    }
  }
  if (!filled) {
    for (let i = 0; i < n; i++) {
      cov[i] = new Array(n).fill(NaN);
      stdErr[i] = NaN;
    }
  }

  return {
    params: result.x,
    cov,
    stdErr,
    fx: result.fx,
    iterations: result.iterations,
    converged: result.converged,
  };
}
