/**
 * Box bounds via MINUIT-style parameter transforms.
 *
 * A bounded external parameter is mapped onto an unbounded internal one:
 *
 *   both bounds:  ext = lo + (hi - lo) * (sin(int) + 1) / 2
 *   lower only:   ext = lo - 1 + sqrt(int^2 + 1)
 *   upper only:   ext = hi + 1 - sqrt(int^2 + 1)
 *
 * Optimizers run in internal space; gradients are chain-ruled by the
 * transform derivative d ext / d int. This bounds every method at once
 * (L-BFGS, Nelder-Mead, Levenberg-Marquardt, ...) with no per-method
 * machinery — the approach MINUIT and lmfit have used for decades.
 *
 * Note: as an optimum approaches a bound the transform derivative goes to
 * zero, so active bounds are reached asymptotically (typically to 1e-4
 * or better of the bound, not exactly on it).
 */

const FREE = 0;
const LOWER = 1;
const UPPER = 2;
const BOTH = 3;

/**
 * Build a bounds transform, or return null when every parameter is free.
 *
 * @param {Array<Array<number|null>>} bounds - Per-parameter [lo, hi];
 *   null/undefined/±Infinity mean unbounded on that side
 * @param {number} n - Number of parameters
 * @returns {Object|null} {toInternal, toExternal, dExtDInt, clamp}
 */
export function makeBoundsTransform(bounds, n) {
  if (!Array.isArray(bounds) || bounds.length !== n) {
    throw new Error(`bounds must be an array of ${n} [lo, hi] pairs`);
  }

  const kind = new Array(n);
  const lo = new Array(n);
  const hi = new Array(n);
  let anyBounded = false;

  for (let i = 0; i < n; i++) {
    const pair = bounds[i] || [null, null];
    const l = pair[0] === null || pair[0] === undefined || pair[0] === -Infinity ? null : pair[0];
    const h = pair[1] === null || pair[1] === undefined || pair[1] === Infinity ? null : pair[1];
    if (l !== null && h !== null && !(l < h)) {
      throw new Error(`bounds[${i}]: lower bound must be strictly below upper (got [${l}, ${h}])`);
    }
    lo[i] = l;
    hi[i] = h;
    kind[i] = l === null ? (h === null ? FREE : UPPER) : h === null ? LOWER : BOTH;
    if (kind[i] !== FREE) anyBounded = true;
  }

  if (!anyBounded) return null;

  /** Nudge a point strictly inside the bounds (transforms stall exactly on a bound). */
  function clamp(xExt) {
    const out = xExt.slice();
    for (let i = 0; i < n; i++) {
      if (kind[i] === BOTH) {
        const eps = (hi[i] - lo[i]) * 1e-8;
        out[i] = Math.min(Math.max(out[i], lo[i] + eps), hi[i] - eps);
      } else if (kind[i] === LOWER) {
        const eps = 1e-8 * Math.max(1, Math.abs(lo[i]));
        out[i] = Math.max(out[i], lo[i] + eps);
      } else if (kind[i] === UPPER) {
        const eps = 1e-8 * Math.max(1, Math.abs(hi[i]));
        out[i] = Math.min(out[i], hi[i] - eps);
      }
    }
    return out;
  }

  function toInternal(xExt) {
    const xc = clamp(xExt);
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      switch (kind[i]) {
        case BOTH:
          out[i] = Math.asin(2 * (xc[i] - lo[i]) / (hi[i] - lo[i]) - 1);
          break;
        case LOWER:
          out[i] = Math.sqrt((xc[i] - lo[i] + 1) ** 2 - 1);
          break;
        case UPPER:
          out[i] = Math.sqrt((hi[i] - xc[i] + 1) ** 2 - 1);
          break;
        default:
          out[i] = xc[i];
      }
    }
    return out;
  }

  function toExternal(xInt) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      switch (kind[i]) {
        case BOTH:
          out[i] = lo[i] + (hi[i] - lo[i]) * (Math.sin(xInt[i]) + 1) / 2;
          break;
        case LOWER:
          out[i] = lo[i] - 1 + Math.sqrt(xInt[i] * xInt[i] + 1);
          break;
        case UPPER:
          out[i] = hi[i] + 1 - Math.sqrt(xInt[i] * xInt[i] + 1);
          break;
        default:
          out[i] = xInt[i];
      }
    }
    return out;
  }

  /** Derivative d ext_i / d int_i at an internal point. */
  function dExtDInt(xInt) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) {
      switch (kind[i]) {
        case BOTH:
          out[i] = (hi[i] - lo[i]) / 2 * Math.cos(xInt[i]);
          break;
        case LOWER:
          out[i] = xInt[i] / Math.sqrt(xInt[i] * xInt[i] + 1);
          break;
        case UPPER:
          out[i] = -xInt[i] / Math.sqrt(xInt[i] * xInt[i] + 1);
          break;
        default:
          out[i] = 1;
      }
    }
    return out;
  }

  return { toInternal, toExternal, dExtDInt, clamp };
}

/**
 * Wrap an objective (and optional gradient) so optimizers can run in
 * internal space. Supports both the scalar form (x) => number and the
 * combined form (x) => {loss, gradient}.
 *
 * @param {Function} f - External objective
 * @param {Function|undefined} grad - External gradient
 * @param {Object} T - Transform from makeBoundsTransform
 * @returns {{f: Function, grad: Function|undefined}}
 */
export function wrapObjective(f, grad, T) {
  const fInt = (xInt) => {
    const out = f(T.toExternal(xInt));
    if (out !== null && typeof out === 'object' && 'loss' in out) {
      const d = T.dExtDInt(xInt);
      return { loss: out.loss, gradient: out.gradient.map((gi, i) => gi * d[i]) };
    }
    return out;
  };
  const gradInt = grad
    ? (xInt) => {
      const d = T.dExtDInt(xInt);
      return grad(T.toExternal(xInt)).map((gi, i) => gi * d[i]);
    }
    : undefined;
  return { f: fInt, grad: gradInt };
}

/**
 * Wrap residuals (and optional Jacobian) for bounded least squares.
 *
 * @param {Function} residuals - External residuals (p) => Array<number>
 * @param {Function|undefined} jacobian - External Jacobian (p) => m-by-n
 * @param {Object} T - Transform from makeBoundsTransform
 * @returns {{residuals: Function, jacobian: Function|undefined}}
 */
export function wrapResiduals(residuals, jacobian, T) {
  const resInt = (pInt) => residuals(T.toExternal(pInt));
  const jacInt = jacobian
    ? (pInt) => {
      const d = T.dExtDInt(pInt);
      const J = jacobian(T.toExternal(pInt));
      return J.map((row) => row.map((v, i) => v * d[i]));
    }
    : undefined;
  return { residuals: resInt, jacobian: jacInt };
}
