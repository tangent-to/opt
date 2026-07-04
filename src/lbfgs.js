/**
 * L-BFGS: limited-memory quasi-Newton minimization
 * (Nocedal & Wright, "Numerical Optimization", algorithm 7.4/7.5).
 *
 * Two-loop recursion over a ring buffer of the last `memory` curvature
 * pairs (s, y), with strong Wolfe line search. Ring storage and work
 * vectors use Float64Array and are allocated once per call.
 */

import { makeEvaluator } from './evaluate.js';
import { strongWolfeLineSearch } from './linesearch.js';

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm2(a) {
  return Math.sqrt(dot(a, a));
}

/**
 * Minimize a function with L-BFGS.
 *
 * @param {Function} f - Objective: (x) => number or (x) => {loss, gradient}
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options]
 * @param {Function} [options.grad] - Gradient: (x) => Array<number> (finite differences otherwise)
 * @param {number} [options.memory=10] - Number of curvature pairs kept
 * @param {number} [options.maxIter=1000] - Maximum iterations
 * @param {number} [options.tol=1e-6] - Convergence tolerance on the gradient norm
 * @param {number} [options.fTol=1e-12] - Relative function-decrease tolerance
 * @param {number} [options.c1=1e-4] - Line search sufficient-decrease constant
 * @param {number} [options.c2=0.9] - Line search curvature constant
 * @param {boolean} [options.verbose=false]
 * @returns {Object} {x, fx, iterations, fevals, converged, history}
 */
export function lbfgs(f, x0, options = {}) {
  const n = x0.length;
  const m = options.memory || 10;
  const maxIter = options.maxIter || 1000;
  const tol = options.tol || 1e-6;
  const fTol = options.fTol !== undefined ? options.fTol : 1e-12;
  const verbose = options.verbose || false;

  const evaluate = makeEvaluator(f, options.grad, options);
  let fevals = 0;
  const ev = (x) => {
    fevals++;
    return evaluate(x);
  };

  // Ring buffers of curvature pairs and the two-loop workspace
  const S = Array.from({ length: m }, () => new Float64Array(n));
  const Y = Array.from({ length: m }, () => new Float64Array(n));
  const rhoBuf = new Float64Array(m);
  const alphaBuf = new Float64Array(m);
  let stored = 0; // number of valid pairs
  let head = 0; // next slot to write

  const d = new Float64Array(n);
  const history = { loss: [], gradNorm: [] };

  let x = [...x0];
  let { loss: fx, gradient: g } = ev(x);
  let converged = false;
  let iteration = 0;

  /** d = -H*g via the two-loop recursion (γ-scaled identity as H0). */
  function computeDirection() {
    for (let i = 0; i < n; i++) d[i] = -g[i];
    if (stored === 0) return;

    for (let k = 0; k < stored; k++) {
      const idx = (head - 1 - k + m * 2) % m; // newest to oldest
      const a = rhoBuf[idx] * dot(S[idx], d);
      alphaBuf[idx] = a;
      const y = Y[idx];
      for (let i = 0; i < n; i++) d[i] -= a * y[i];
    }

    // H0 = γI with γ = s'y / y'y from the most recent pair
    const newest = (head - 1 + m) % m;
    const gamma = 1 / (rhoBuf[newest] * dot(Y[newest], Y[newest]));
    for (let i = 0; i < n; i++) d[i] *= gamma;

    for (let k = stored - 1; k >= 0; k--) {
      const idx = (head - 1 - k + m * 2) % m;
      const beta = rhoBuf[idx] * dot(Y[idx], d);
      const a = alphaBuf[idx];
      const s = S[idx];
      for (let i = 0; i < n; i++) d[i] += (a - beta) * s[i];
    }
  }

  for (; iteration < maxIter; iteration++) {
    history.loss.push(fx);
    const gNorm = norm2(g);
    history.gradNorm.push(gNorm);

    if (gNorm < tol) {
      converged = true;
      break;
    }

    computeDirection();

    // Guard: fall back to steepest descent if d is not a descent direction
    if (dot(g, d) >= 0) {
      stored = 0;
      head = 0;
      for (let i = 0; i < n; i++) d[i] = -g[i];
    }

    // Unit initial step once curvature is captured; scaled on iteration 0
    const t0 = stored === 0 ? Math.min(1, 1 / Math.max(gNorm, 1e-12)) : 1;
    let ls = strongWolfeLineSearch(ev, x, d, fx, g, {
      t0,
      c1: options.c1,
      c2: options.c2,
    });

    if (!ls.success && stored > 0) {
      // Curvature memory may be stale: restart from steepest descent
      stored = 0;
      head = 0;
      for (let i = 0; i < n; i++) d[i] = -g[i];
      ls = strongWolfeLineSearch(ev, x, d, fx, g, {
        t0: Math.min(1, 1 / Math.max(gNorm, 1e-12)),
        c1: options.c1,
        c2: options.c2,
      });
    }
    if (!ls.success) break; // converged stays false

    // Store the curvature pair when it keeps H positive definite
    const s = S[head];
    const y = Y[head];
    let ys = 0;
    for (let i = 0; i < n; i++) {
      s[i] = ls.xNew[i] - x[i];
      y[i] = ls.gradient[i] - g[i];
      ys += s[i] * y[i];
    }
    if (ys > 1e-10 * norm2(s) * norm2(y)) {
      rhoBuf[head] = 1 / ys;
      head = (head + 1) % m;
      if (stored < m) stored++;
    }

    const fxPrev = fx;
    x = ls.xNew;
    fx = ls.fx;
    g = ls.gradient;

    if (verbose && iteration % 10 === 0) {
      console.log(`Iter ${iteration}: loss=${fx.toFixed(8)}, grad_norm=${gNorm.toExponential(2)}`);
    }

    // Relative function-decrease stop (scipy factr-style)
    if (fxPrev - fx <= fTol * Math.max(Math.abs(fxPrev), Math.abs(fx), 1)) {
      converged = true;
      iteration++;
      history.loss.push(fx);
      history.gradNorm.push(norm2(g));
      break;
    }
  }

  return { x, fx, iterations: iteration, fevals, converged, history };
}
