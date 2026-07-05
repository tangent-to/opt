/**
 * Nelder-Mead downhill simplex minimization (derivative-free).
 *
 * Ported from fmin (https://github.com/benfred/fmin),
 * Copyright 2016, Ben Frederickson, BSD-3-Clause.
 * See THIRD_PARTY_NOTICES.md.
 *
 * Changes from the original: unified options/result shape, iteration and
 * function-evaluation counters, explicit convergence flag, separate fTol/xTol
 * options (the original read both from `minErrorDelta`), and support for
 * objectives in tangent/ds combined form ((x) => {loss, gradient}).
 */

import { weightedSum } from './blas1.js';
import { makeScalarEvaluator } from './evaluate.js';

/**
 * Minimize a function using the Nelder-Mead downhill simplex method.
 *
 * @param {Function} f - Objective: (x) => number (or (x) => {loss, ...})
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options]
 * @param {number} [options.maxIter=200*n] - Maximum iterations
 * @param {number} [options.fTol=1e-6] - Convergence tolerance on f spread across the simplex
 * @param {number} [options.xTol=1e-5] - Convergence tolerance on x spread across the simplex
 * @param {number} [options.rho=1] - Reflection coefficient
 * @param {number} [options.chi=2] - Expansion coefficient
 * @param {number} [options.psi=-0.5] - Contraction coefficient
 * @param {number} [options.sigma=0.5] - Reduction (shrink) coefficient
 * @param {number} [options.nonZeroDelta=1.05] - Relative perturbation for nonzero x0 entries
 * @param {number} [options.zeroDelta=1e-3] - Absolute perturbation for zero x0 entries
 * @param {boolean} [options.history=false] - Record {x, fx, simplex} per iteration
 * @returns {Object} {x, fx, iterations, fevals, converged, history?}
 */
export function nelderMead(f, x0, options = {}) {
  const n = x0.length;
  const maxIter = options.maxIter || n * 200;
  const fTol = options.fTol || options.tol || 1e-6;
  const xTol = options.xTol || 1e-5;
  const nonZeroDelta = options.nonZeroDelta || 1.05;
  const zeroDelta = options.zeroDelta || 1e-3;
  const rho = options.rho !== undefined ? options.rho : 1;
  const chi = options.chi !== undefined ? options.chi : 2;
  const psi = options.psi !== undefined ? options.psi : -0.5;
  const sigma = options.sigma !== undefined ? options.sigma : 0.5;
  const trackHistory = options.history || false;

  const evaluateScalar = makeScalarEvaluator(f);
  let fevals = 0;
  const fn = (x) => {
    fevals++;
    return evaluateScalar(x);
  };

  // Initialize the simplex: x0 plus one perturbed point per dimension.
  // Each vertex is an array with `fx` and `id` bookkeeping properties.
  const simplex = new Array(n + 1);
  simplex[0] = x0.slice();
  simplex[0].fx = fn(x0);
  simplex[0].id = 0;
  for (let i = 0; i < n; ++i) {
    const point = x0.slice();
    point[i] = point[i] ? point[i] * nonZeroDelta : zeroDelta;
    simplex[i + 1] = point;
    simplex[i + 1].fx = fn(point);
    simplex[i + 1].id = i + 1;
  }

  function updateSimplex(value) {
    for (let i = 0; i < value.length; i++) {
      simplex[n][i] = value[i];
    }
    simplex[n].fx = value.fx;
  }

  const sortOrder = (a, b) => a.fx - b.fx;

  const centroid = x0.slice();
  const reflected = x0.slice();
  const contracted = x0.slice();
  const expanded = x0.slice();

  const history = trackHistory ? [] : null;
  let converged = false;
  let iteration = 0;

  for (; iteration < maxIter; ++iteration) {
    simplex.sort(sortOrder);

    if (trackHistory) {
      // Copy the simplex (later iterations mutate it) and sort by vertex id
      // for a consistent order between iterations.
      const sortedSimplex = simplex.map((v) => {
        const state = v.slice();
        state.fx = v.fx;
        state.id = v.id;
        return state;
      });
      sortedSimplex.sort((a, b) => a.id - b.id);
      history.push({
        x: simplex[0].slice(),
        fx: simplex[0].fx,
        simplex: sortedSimplex,
      });
    }

    let maxDiff = 0;
    for (let i = 0; i < n; ++i) {
      maxDiff = Math.max(maxDiff, Math.abs(simplex[0][i] - simplex[1][i]));
    }

    if (Math.abs(simplex[0].fx - simplex[n].fx) < fTol && maxDiff < xTol) {
      converged = true;
      break;
    }

    // Centroid of all but the worst point
    for (let i = 0; i < n; ++i) {
      centroid[i] = 0;
      for (let j = 0; j < n; ++j) {
        centroid[i] += simplex[j][i];
      }
      centroid[i] /= n;
    }

    // Reflect the worst point past the centroid
    const worst = simplex[n];
    weightedSum(reflected, 1 + rho, centroid, -rho, worst);
    reflected.fx = fn(reflected);

    if (reflected.fx < simplex[0].fx) {
      // Best point seen so far: try expanding
      weightedSum(expanded, 1 + chi, centroid, -chi, worst);
      expanded.fx = fn(expanded);
      if (expanded.fx < reflected.fx) {
        updateSimplex(expanded);
      } else {
        updateSimplex(reflected);
      }
    } else if (reflected.fx >= simplex[n - 1].fx) {
      // Worse than the second worst: contract
      let shouldReduce = false;

      if (reflected.fx > worst.fx) {
        // Inside contraction
        weightedSum(contracted, 1 + psi, centroid, -psi, worst);
        contracted.fx = fn(contracted);
        if (contracted.fx < worst.fx) {
          updateSimplex(contracted);
        } else {
          shouldReduce = true;
        }
      } else {
        // Outside contraction
        weightedSum(contracted, 1 - psi * rho, centroid, psi * rho, worst);
        contracted.fx = fn(contracted);
        if (contracted.fx < reflected.fx) {
          updateSimplex(contracted);
        } else {
          shouldReduce = true;
        }
      }

      if (shouldReduce) {
        if (sigma >= 1) break;
        // Shrink the simplex toward the best point
        for (let i = 1; i < simplex.length; ++i) {
          weightedSum(simplex[i], 1 - sigma, simplex[0], sigma, simplex[i]);
          simplex[i].fx = fn(simplex[i]);
        }
      }
    } else {
      updateSimplex(reflected);
    }
  }

  simplex.sort(sortOrder);

  const result = {
    x: Array.from(simplex[0]),
    fx: simplex[0].fx,
    iterations: iteration,
    nfev: fevals,
    fevals, // deprecated alias for nfev
    converged,
    success: converged, // suite-wide "did it work" flag (matches ode)
  };
  if (trackHistory) {
    result.history = history;
  }
  return result;
}
