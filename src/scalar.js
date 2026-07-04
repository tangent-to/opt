/**
 * Scalar (univariate) minimization and root finding.
 *
 * minimizeScalar: Brent's minimization method (parabolic interpolation with
 * golden-section safeguards) or plain golden-section search. When no 3-point
 * bracket is supplied, a scipy-style downhill golden-ratio expansion
 * (with parabolic extrapolation and a grow limit) brackets the minimum first.
 *
 * rootScalar: the Brent-Dekker method (inverse quadratic interpolation and
 * secant steps with bisection safeguards) or plain bisection.
 *
 * Both are written from the standard algorithm descriptions and follow the
 * package's declarative options-object style, returning
 * {x, fx, iterations, fevals, converged}.
 */

const GOLD = 1.618034; // golden ratio, downhill expansion factor
const CGOLD = 0.3819660112501051; // 2 - phi, golden-section fraction
const MIN_TOL = 1e-11; // absolute floor added to relative x tolerances
const VERY_SMALL = 1e-21; // denominator guard in parabolic extrapolation

/**
 * Bracket a minimum by golden-ratio downhill expansion from two points,
 * with a parabolic extrapolation step and a grow limit (as in
 * scipy.optimize.bracket).
 *
 * @param {Function} fn - Counted objective: (x) => number
 * @param {number} xa0 - First starting point
 * @param {number} xb0 - Second starting point
 * @param {number} [growLimit=110] - Maximum grow factor per parabolic step
 * @param {number} [maxIter=1000] - Maximum number of expansions
 * @returns {Object} {xa, xb, xc, fa, fb, fc} with fb <= fa and fb <= fc
 */
function bracketMinimum(fn, xa0, xb0, growLimit = 110, maxIter = 1000) {
  let xa = xa0;
  let xb = xb0;
  let fa = fn(xa);
  let fb = fn(xb);
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) {
    throw new Error(
      `minimizeScalar: f is not finite at the initial bracket points ` +
        `(f(${xa}) = ${fa}, f(${xb}) = ${fb})`,
    );
  }
  if (fa < fb) {
    // Ensure the downhill direction is from xa to xb.
    let tmp = xa;
    xa = xb;
    xb = tmp;
    tmp = fa;
    fa = fb;
    fb = tmp;
  }
  let xc = xb + GOLD * (xb - xa);
  let fc = fn(xc);
  if (fc !== fc) fc = Infinity;

  let iteration = 0;
  while (fc < fb) {
    if (iteration >= maxIter) {
      throw new Error(
        `minimizeScalar: failed to bracket a minimum after ${maxIter} expansions from ` +
          `[${xa0}, ${xb0}] (the function may be monotonic or unbounded below)`,
      );
    }
    iteration++;

    // Parabolic extrapolation through (xa, fa), (xb, fb), (xc, fc).
    const tmp1 = (xb - xa) * (fb - fc);
    const tmp2 = (xb - xc) * (fb - fa);
    const val = tmp2 - tmp1;
    const denom = Math.abs(val) < VERY_SMALL
      ? 2 * VERY_SMALL
      : 2 * val;
    let w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom;
    const wlim = xb + growLimit * (xc - xb);
    let fw;

    if ((w - xc) * (xb - w) > 0) {
      // Parabolic point is between xb and xc.
      fw = fn(w);
      if (fw !== fw) fw = Infinity;
      if (fw < fc) {
        // Minimum between xb and xc.
        xa = xb;
        xb = w;
        fa = fb;
        fb = fw;
        break;
      } else if (fw > fb) {
        // Minimum between xa and w.
        xc = w;
        fc = fw;
        break;
      }
      // Parabolic step did not help: default golden-ratio expansion.
      w = xc + GOLD * (xc - xb);
      fw = fn(w);
      if (fw !== fw) fw = Infinity;
    } else if ((w - wlim) * (wlim - xc) >= 0) {
      // Parabolic point is at or beyond the grow limit: clamp to it.
      w = wlim;
      fw = fn(w);
      if (fw !== fw) fw = Infinity;
    } else if ((w - wlim) * (xc - w) > 0) {
      // Parabolic point is between xc and the grow limit.
      fw = fn(w);
      if (fw !== fw) fw = Infinity;
      if (fw < fc) {
        xb = xc;
        xc = w;
        w = xc + GOLD * (xc - xb);
        fb = fc;
        fc = fw;
        fw = fn(w);
        if (fw !== fw) fw = Infinity;
      }
    } else {
      // Reject the parabolic point: default golden-ratio expansion.
      w = xc + GOLD * (xc - xb);
      fw = fn(w);
      if (fw !== fw) fw = Infinity;
    }

    xa = xb;
    xb = xc;
    xc = w;
    fa = fb;
    fb = fc;
    fc = fw;

    if (!Number.isFinite(xc)) {
      throw new Error(
        `minimizeScalar: failed to bracket a minimum starting from [${xa0}, ${xb0}]; ` +
          'the expansion diverged (the function may be monotonic or unbounded below)',
      );
    }
  }

  const bracketed = (fb < fc && fb <= fa) || (fb < fa && fb <= fc);
  if (!bracketed || !Number.isFinite(fb)) {
    throw new Error(
      `minimizeScalar: failed to bracket a minimum starting from [${xa0}, ${xb0}] ` +
        '(the function may be monotonic)',
    );
  }
  return { xa, xb, xc, fa, fb, fc };
}

/**
 * Brent minimization on a bracketing triple (xa, xb, xc) with f(xb) known.
 * Parabolic interpolation is attempted each step and rejected in favor of a
 * golden-section step when unacceptable; steps are floored at
 * tol1 = xTol * |x| + MIN_TOL.
 *
 * @param {Function} fn - Counted objective: (x) => number
 * @param {number} xa - Bracket endpoint
 * @param {number} xb - Interior point with the lowest known f
 * @param {number} xc - Bracket endpoint
 * @param {number} fxb - f(xb)
 * @param {number} xTol - Relative tolerance on x
 * @param {number} maxIter - Maximum iterations
 * @returns {Object} {x, fx, iterations, converged}
 */
function brentMinimize(fn, xa, xb, xc, fxb, xTol, maxIter) {
  let a = xa < xc ? xa : xc;
  let b = xa < xc ? xc : xa;
  let x = xb;
  let w = xb;
  let v = xb;
  let fx = fxb;
  let fw = fxb;
  let fv = fxb;
  let d = 0;
  let e = 0;
  let converged = false;
  let iteration = 0;

  for (; iteration < maxIter; ++iteration) {
    const xm = 0.5 * (a + b);
    const tol1 = xTol * Math.abs(x) + MIN_TOL;
    const tol2 = 2 * tol1;

    if (Math.abs(x - xm) <= tol2 - 0.5 * (b - a)) {
      converged = true;
      break;
    }

    let useGolden = true;
    if (Math.abs(e) > tol1) {
      // Fit a parabola through (v, fv), (w, fw), (x, fx).
      const r = (x - w) * (fx - fv);
      let q = (x - v) * (fx - fw);
      let p = (x - v) * q - (x - w) * r;
      q = 2 * (q - r);
      if (q > 0) p = -p;
      q = Math.abs(q);
      const etemp = e;
      e = d;
      // Accept the parabolic step only if it is finite, smaller than half the
      // second-to-last step, and lands strictly inside the bracket.
      if (
        Number.isFinite(p) && Number.isFinite(q) && q !== 0 &&
        Math.abs(p) < Math.abs(0.5 * q * etemp) &&
        p > q * (a - x) && p < q * (b - x)
      ) {
        d = p / q;
        const u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          d = xm - x >= 0 ? tol1 : -tol1;
        }
        useGolden = false;
      }
    }
    if (useGolden) {
      e = x >= xm ? a - x : b - x;
      d = CGOLD * e;
    }

    // Never step by less than tol1.
    const u = Math.abs(d) >= tol1 ? x + d : x + (d >= 0 ? tol1 : -tol1);
    let fu = fn(u);
    if (!Number.isFinite(fu)) fu = Infinity;

    if (fu <= fx) {
      if (u >= x) {
        a = x;
      } else {
        b = x;
      }
      v = w;
      fv = fw;
      w = x;
      fw = fx;
      x = u;
      fx = fu;
    } else {
      if (u < x) {
        a = u;
      } else {
        b = u;
      }
      if (fu <= fw || w === x) {
        v = w;
        fv = fw;
        w = u;
        fw = fu;
      } else if (fu <= fv || v === x || v === w) {
        v = u;
        fv = fu;
      }
    }
  }

  return { x, fx, iterations: iteration, converged };
}

/**
 * Plain golden-section search on a bracketing triple (xa, xb, xc).
 *
 * @param {Function} fn - Counted objective: (x) => number
 * @param {number} xa - Bracket endpoint
 * @param {number} xb - Interior point with the lowest known f
 * @param {number} xc - Bracket endpoint
 * @param {number} fxb - f(xb)
 * @param {number} xTol - Relative tolerance on x
 * @param {number} maxIter - Maximum iterations
 * @returns {Object} {x, fx, iterations, converged}
 */
function goldenMinimize(fn, xa, xb, xc, fxb, xTol, maxIter) {
  const gr = 1 - CGOLD;
  let x0 = xa < xc ? xa : xc;
  let x3 = xa < xc ? xc : xa;
  let x1;
  let x2;
  let f1;
  let f2;

  // Place the new interior point in the larger of the two sub-intervals.
  if (Math.abs(x3 - xb) > Math.abs(xb - x0)) {
    x1 = xb;
    f1 = fxb;
    x2 = xb + CGOLD * (x3 - xb);
    f2 = fn(x2);
  } else {
    x2 = xb;
    f2 = fxb;
    x1 = xb - CGOLD * (xb - x0);
    f1 = fn(x1);
  }
  if (!Number.isFinite(f1)) f1 = Infinity;
  if (!Number.isFinite(f2)) f2 = Infinity;

  let converged = false;
  let iteration = 0;
  for (; iteration < maxIter; ++iteration) {
    if (Math.abs(x3 - x0) <= xTol * (Math.abs(x1) + Math.abs(x2)) + MIN_TOL) {
      converged = true;
      break;
    }
    if (f2 < f1) {
      x0 = x1;
      x1 = x2;
      x2 = gr * x2 + CGOLD * x3;
      f1 = f2;
      f2 = fn(x2);
      if (!Number.isFinite(f2)) f2 = Infinity;
    } else {
      x3 = x2;
      x2 = x1;
      x1 = gr * x1 + CGOLD * x0;
      f2 = f1;
      f1 = fn(x1);
      if (!Number.isFinite(f1)) f1 = Infinity;
    }
  }

  if (f1 < f2) {
    return { x: x1, fx: f1, iterations: iteration, converged };
  }
  return { x: x2, fx: f2, iterations: iteration, converged };
}

/**
 * Minimize a univariate function.
 *
 * @param {Function} f - Objective: (x: number) => number
 * @param {Object} [options]
 * @param {string} [options.method='brent'] - 'brent' or 'golden'
 * @param {Array<number>} [options.bracket=[0, 1]] - [a, b] to auto-bracket
 *   from, or a full bracketing triple [a, b, c] with f(b) <= f(a), f(b) <= f(c)
 * @param {number} [options.xTol=1e-8] - Relative tolerance on x
 * @param {number} [options.maxIter=500] - Maximum iterations
 * @returns {Object} {x, fx, iterations, fevals, converged}
 */
export function minimizeScalar(f, options = {}) {
  if (typeof f !== 'function') {
    throw new Error('minimizeScalar: f must be a function');
  }
  const method = options.method !== undefined ? String(options.method).toLowerCase() : 'brent';
  const xTol = options.xTol !== undefined ? options.xTol : 1e-8;
  const maxIter = options.maxIter !== undefined ? options.maxIter : 500;
  const bracket = options.bracket !== undefined ? options.bracket : [0, 1];

  if (method !== 'brent' && method !== 'golden') {
    throw new Error(
      `minimizeScalar: unknown method '${options.method}'. Available: brent, golden`,
    );
  }
  if (
    !Array.isArray(bracket) || bracket.length < 2 || bracket.length > 3 ||
    bracket.some((v) => typeof v !== 'number' || !Number.isFinite(v))
  ) {
    throw new Error('minimizeScalar: options.bracket must be [a, b] or [a, b, c] finite numbers');
  }

  let fevals = 0;
  const fn = (t) => {
    fevals++;
    return f(t);
  };

  let xa;
  let xb;
  let xc;
  let fb;
  if (bracket.length === 3) {
    xa = bracket[0];
    xb = bracket[1];
    xc = bracket[2];
    if (!((xa < xb && xb < xc) || (xa > xb && xb > xc))) {
      throw new Error(
        `minimizeScalar: bracket [${xa}, ${xb}, ${xc}] is not monotonically ordered`,
      );
    }
    const fa = fn(xa);
    fb = fn(xb);
    const fc = fn(xc);
    if (!Number.isFinite(fb) || !(fb <= fa) || !(fb <= fc)) {
      throw new Error(
        `minimizeScalar: bracket [${xa}, ${xb}, ${xc}] does not bracket a minimum ` +
          `(f values: ${fa}, ${fb}, ${fc}); f(b) must be finite and <= both f(a) and f(c)`,
      );
    }
  } else {
    const br = bracketMinimum(fn, bracket[0], bracket[1]);
    xa = br.xa;
    xb = br.xb;
    xc = br.xc;
    fb = br.fb;
  }

  const result = method === 'brent'
    ? brentMinimize(fn, xa, xb, xc, fb, xTol, maxIter)
    : goldenMinimize(fn, xa, xb, xc, fb, xTol, maxIter);
  result.fevals = fevals;
  return result;
}

/**
 * Brent-Dekker root finding: inverse quadratic interpolation and secant
 * steps, safeguarded by bisection. fa and fb must have opposite signs.
 *
 * @param {Function} fn - Counted function: (x) => number
 * @param {number} a0 - Bracket endpoint
 * @param {number} b0 - Bracket endpoint
 * @param {number} fa0 - f(a0)
 * @param {number} fb0 - f(b0)
 * @param {number} xTol - Absolute tolerance on x
 * @param {number} rTol - Relative tolerance on x
 * @param {number} maxIter - Maximum iterations
 * @returns {Object} {x, fx, iterations, converged}
 */
function brentRoot(fn, a0, b0, fa0, fb0, xTol, rTol, maxIter) {
  let a = a0;
  let b = b0;
  let fa = fa0;
  let fb = fb0;

  // Keep b as the better (smaller |f|) estimate.
  if (Math.abs(fa) < Math.abs(fb)) {
    let tmp = a;
    a = b;
    b = tmp;
    tmp = fa;
    fa = fb;
    fb = tmp;
  }
  let c = a;
  let fc = fa;
  let d = c;
  let mflag = true;
  let converged = false;
  let iteration = 0;

  for (; iteration < maxIter; ++iteration) {
    const tol = 2 * rTol * Math.abs(b) + xTol;
    if (fb === 0 || Math.abs(b - a) <= tol) {
      converged = true;
      break;
    }

    let s;
    if (fa !== fc && fb !== fc) {
      // Inverse quadratic interpolation.
      s = (a * fb * fc) / ((fa - fb) * (fa - fc)) +
        (b * fa * fc) / ((fb - fa) * (fb - fc)) +
        (c * fa * fb) / ((fc - fa) * (fc - fb));
    } else {
      // Secant step.
      s = b - fb * (b - a) / (fb - fa);
    }

    // Safeguards: fall back to bisection when the interpolated step is
    // suspect (outside [(3a + b) / 4, b], or not shrinking fast enough).
    const lo = (3 * a + b) / 4;
    const useBisection = !Number.isFinite(s) ||
      !((s > lo && s < b) || (s < lo && s > b)) ||
      (mflag && Math.abs(s - b) >= Math.abs(b - c) / 2) ||
      (!mflag && Math.abs(s - b) >= Math.abs(c - d) / 2) ||
      (mflag && Math.abs(b - c) < tol) ||
      (!mflag && Math.abs(c - d) < tol);
    if (useBisection) {
      s = 0.5 * (a + b);
      mflag = true;
    } else {
      mflag = false;
    }

    let fs = fn(s);
    if (!Number.isFinite(fs)) {
      // Non-finite interpolated value: fall back to the bisection midpoint.
      const mid = 0.5 * (a + b);
      if (s !== mid) {
        s = mid;
        mflag = true;
        fs = fn(s);
      }
      if (!Number.isFinite(fs)) {
        throw new Error(`rootScalar: f returned a non-finite value (${fs}) at x = ${s}`);
      }
    }

    d = c;
    c = b;
    fc = fb;
    if ((fa > 0) !== (fs > 0)) {
      b = s;
      fb = fs;
    } else {
      a = s;
      fa = fs;
    }
    if (Math.abs(fa) < Math.abs(fb)) {
      let tmp = a;
      a = b;
      b = tmp;
      tmp = fa;
      fa = fb;
      fb = tmp;
    }
  }

  return { x: b, fx: fb, iterations: iteration, converged };
}

/**
 * Plain bisection root finding. fa and fb must have opposite signs.
 *
 * @param {Function} fn - Counted function: (x) => number
 * @param {number} a0 - Bracket endpoint
 * @param {number} b0 - Bracket endpoint
 * @param {number} fa0 - f(a0)
 * @param {number} xTol - Absolute tolerance on x
 * @param {number} rTol - Relative tolerance on x
 * @param {number} maxIter - Maximum iterations
 * @returns {Object} {x, fx, iterations, converged}
 */
function bisectRoot(fn, a0, b0, fa0, xTol, rTol, maxIter) {
  let a = a0;
  let b = b0;
  let fa = fa0;
  let x = 0.5 * (a + b);
  let fx = NaN;
  let converged = false;
  let iteration = 0;

  for (; iteration < maxIter; ++iteration) {
    x = 0.5 * (a + b);
    fx = fn(x);
    if (!Number.isFinite(fx)) {
      throw new Error(`rootScalar: f returned a non-finite value (${fx}) at x = ${x}`);
    }
    const tol = 2 * rTol * Math.abs(x) + xTol;
    if (fx === 0 || 0.5 * Math.abs(b - a) <= tol) {
      converged = true;
      break;
    }
    if ((fa > 0) !== (fx > 0)) {
      b = x;
    } else {
      a = x;
      fa = fx;
    }
  }

  return { x, fx, iterations: iteration, converged };
}

/**
 * Find a root of a univariate function inside a sign-changing bracket.
 *
 * @param {Function} f - Function: (x: number) => number
 * @param {Object} [options]
 * @param {string} [options.method='brent'] - 'brent' or 'bisect'
 * @param {Array<number>} options.bracket - [a, b] with f(a) and f(b) of
 *   opposite signs (required)
 * @param {number} [options.xTol=1e-12] - Absolute tolerance on x
 * @param {number} [options.rTol=4 * Number.EPSILON] - Relative tolerance on x
 * @param {number} [options.maxIter=100] - Maximum iterations
 * @returns {Object} {x, fx, iterations, fevals, converged}
 */
export function rootScalar(f, options = {}) {
  if (typeof f !== 'function') {
    throw new Error('rootScalar: f must be a function');
  }
  const method = options.method !== undefined ? String(options.method).toLowerCase() : 'brent';
  const xTol = options.xTol !== undefined ? options.xTol : 1e-12;
  const rTol = options.rTol !== undefined ? options.rTol : 4 * Number.EPSILON;
  const maxIter = options.maxIter !== undefined ? options.maxIter : 100;
  const bracket = options.bracket;

  if (method !== 'brent' && method !== 'bisect') {
    throw new Error(
      `rootScalar: unknown method '${options.method}'. Available: brent, bisect`,
    );
  }
  if (
    !Array.isArray(bracket) || bracket.length !== 2 ||
    bracket.some((v) => typeof v !== 'number' || !Number.isFinite(v))
  ) {
    throw new Error('rootScalar: options.bracket = [a, b] (finite numbers) is required');
  }

  let fevals = 0;
  const fn = (t) => {
    fevals++;
    return f(t);
  };

  const a = bracket[0];
  const b = bracket[1];
  const fa = fn(a);
  const fb = fn(b);
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) {
    throw new Error(
      `rootScalar: f is not finite at the bracket endpoints ` +
        `(f(${a}) = ${fa}, f(${b}) = ${fb})`,
    );
  }
  if (fa === 0) {
    return { x: a, fx: fa, iterations: 0, fevals, converged: true };
  }
  if (fb === 0) {
    return { x: b, fx: fb, iterations: 0, fevals, converged: true };
  }
  if ((fa > 0) === (fb > 0)) {
    throw new Error(
      `rootScalar: f(a) and f(b) must have opposite signs, but ` +
        `f(${a}) = ${fa} and f(${b}) = ${fb}; a sign change across the bracket is required`,
    );
  }

  const result = method === 'brent'
    ? brentRoot(fn, a, b, fa, fb, xTol, rTol, maxIter)
    : bisectRoot(fn, a, b, fa, xTol, rTol, maxIter);
  result.fevals = fevals;
  return result;
}
