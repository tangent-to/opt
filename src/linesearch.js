/**
 * Strong Wolfe line search (bracket + zoom with cubic interpolation),
 * after Nocedal & Wright, "Numerical Optimization", algorithms 3.5/3.6.
 *
 * Guarantees, on success, a step t with:
 *   f(x + t*p) <= f(x) + c1 * t * g0'p        (sufficient decrease)
 *   |g(x + t*p)'p| <= c2 * |g0'p|             (curvature)
 *
 * Non-finite trial values are treated as "step too long" and bracketed
 * away rather than propagated.
 */

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Cubic minimizer of a Hermite interpolant on [a, b] given f and f' at a
 * and f at b (and optionally f' at b). Falls back to the midpoint when the
 * interpolation is degenerate or lands outside a safeguarded interior.
 */
function interpolate(a, fa, da, b, fb, db) {
  // Try cubic interpolation using both derivatives when available
  if (db !== undefined && Number.isFinite(db)) {
    const d1 = da + db - 3 * (fa - fb) / (a - b);
    const disc = d1 * d1 - da * db;
    if (disc >= 0) {
      const d2 = Math.sign(b - a) * Math.sqrt(disc);
      const t = b - (b - a) * ((db + d2 - d1) / (db - da + 2 * d2));
      if (Number.isFinite(t)) return t;
    }
  }
  // Quadratic through fa, da, fb
  const t = a - (da * (b - a) * (b - a)) / (2 * (fb - fa - da * (b - a)));
  return Number.isFinite(t) ? t : (a + b) / 2;
}

/**
 * Strong Wolfe line search along direction p from x.
 *
 * @param {Function} evaluate - (x) => {loss, gradient}
 * @param {Array<number>} x - Starting point
 * @param {Array<number>|Float64Array} p - Descent direction (g0'p must be < 0)
 * @param {number} f0 - f(x)
 * @param {Array<number>} g0 - gradient at x
 * @param {Object} [options]
 * @param {number} [options.t0=1] - Initial step
 * @param {number} [options.c1=1e-4] - Sufficient-decrease constant
 * @param {number} [options.c2=0.9] - Curvature constant
 * @param {number} [options.maxIter=25] - Bracketing iterations
 * @param {number} [options.tMax=1e10] - Maximum step
 * @returns {Object} {success, t, fx, gradient, xNew, fevals}
 */
export function strongWolfeLineSearch(evaluate, x, p, f0, g0, options = {}) {
  const c1 = options.c1 !== undefined ? options.c1 : 1e-4;
  const c2 = options.c2 !== undefined ? options.c2 : 0.9;
  const maxIter = options.maxIter || 25;
  const tMax = options.tMax || 1e10;
  const n = x.length;

  const d0 = dot(g0, p);
  if (!(d0 < 0)) {
    return { success: false, t: 0, fx: f0, gradient: g0, xNew: x, fevals: 0 };
  }

  let fevals = 0;
  const xTrial = new Array(n);
  const phi = (t) => {
    for (let i = 0; i < n; i++) xTrial[i] = x[i] + t * p[i];
    const { loss, gradient } = evaluate(xTrial);
    fevals++;
    return { f: loss, g: gradient, dphi: dot(gradient, p) };
  };

  const fail = () => ({ success: false, t: 0, fx: f0, gradient: g0, xNew: x, fevals });
  const ok = (t, e) => ({
    success: true,
    t,
    fx: e.f,
    gradient: e.g,
    xNew: x.map((xi, i) => xi + t * p[i]),
    fevals,
  });

  /** Zoom phase: the Wolfe step is bracketed in [lo, hi] (in function-value order). */
  function zoom(tLo, eLo, tHi, eHi) {
    for (let j = 0; j < maxIter; j++) {
      let t = interpolate(tLo, eLo.f, eLo.dphi, tHi, eHi.f, eHi.dphi);
      const lo = Math.min(tLo, tHi);
      const hi = Math.max(tLo, tHi);
      // Safeguard: keep the trial strictly interior, else bisect
      if (!Number.isFinite(t) || t <= lo + 0.1 * (hi - lo) || t >= hi - 0.1 * (hi - lo)) {
        t = (tLo + tHi) / 2;
      }
      const e = phi(t);
      if (!Number.isFinite(e.f) || e.f > f0 + c1 * t * d0 || e.f >= eLo.f) {
        tHi = t;
        eHi = e;
      } else {
        if (Math.abs(e.dphi) <= -c2 * d0) return ok(t, e);
        if (e.dphi * (tHi - tLo) >= 0) {
          tHi = tLo;
          eHi = eLo;
        }
        tLo = t;
        eLo = e;
      }
      if (Math.abs(tHi - tLo) < 1e-16 * Math.max(1, Math.abs(tLo))) break;
    }
    // Interval collapsed: accept lo if it at least sufficiently decreases f
    return eLo.f < f0 + c1 * tLo * d0 && tLo > 0 ? ok(tLo, eLo) : fail();
  }

  let tPrev = 0;
  let ePrev = { f: f0, g: g0, dphi: d0 };
  let t = options.t0 !== undefined ? options.t0 : 1;

  for (let i = 0; i < maxIter; i++) {
    const e = phi(t);
    if (!Number.isFinite(e.f) || e.f > f0 + c1 * t * d0 || (i > 0 && e.f >= ePrev.f)) {
      return zoom(tPrev, ePrev, t, e);
    }
    if (Math.abs(e.dphi) <= -c2 * d0) return ok(t, e);
    if (e.dphi >= 0) return zoom(t, e, tPrev, ePrev);
    tPrev = t;
    ePrev = e;
    t = Math.min(2 * t, tMax);
    if (t >= tMax) return fail();
  }

  return fail();
}
