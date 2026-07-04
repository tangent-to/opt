/**
 * Declarative minimization front-end.
 *
 * A single entry point in the tangent/ds options-object style:
 *
 *   const result = minimize({
 *     f: (x) => (x[0] - 1) ** 2 + (x[1] + 2) ** 2,
 *     x0: [0, 0],
 *     method: 'neldermead',        // default
 *     maxIter: 500,
 *   });
 *   // result: {x, fx, iterations, converged, method, ...}
 *
 * Gradient-based methods use `grad` when provided, the combined
 * (x) => {loss, gradient} form when f returns one, and central finite
 * differences otherwise.
 */

import { nelderMead } from './neldermead.js';
import { lbfgs } from './lbfgs.js';
import { adam, gradientDescent } from './gradient.js';
import { makeBoundsTransform, wrapObjective } from './bounds.js';

/**
 * The curated roster: textbook-intuitive plus modern best practice.
 * - neldermead: derivative-free, geometric intuition
 * - lbfgs: the modern default for smooth objectives
 * - gd: the teaching baseline for gradient descent
 * - adam: the modern default for stochastic/ML-style objectives
 * momentumDescent and rmsprop remain importable (and in the ds compat
 * layer) but are deliberately not part of the declarative roster.
 */
const METHODS = {
  neldermead: nelderMead,
  'nelder-mead': nelderMead,
  lbfgs: lbfgs,
  'l-bfgs': lbfgs,
  gd: gradientDescent,
  sgd: gradientDescent,
  gradient_descent: gradientDescent,
  adam: adam,
};

/**
 * List the available minimization methods.
 * @returns {Array<string>}
 */
export function methods() {
  return Object.keys(METHODS);
}

/**
 * Minimize a scalar function of one or more variables.
 *
 * @param {Object} spec
 * @param {Function} spec.f - Objective: (x) => number or (x) => {loss, gradient}
 * @param {Array<number>} spec.x0 - Initial parameters
 * @param {string} [spec.method='neldermead'] - One of methods()
 * @param {Function} [spec.grad] - Gradient: (x) => Array<number> (gradient methods only)
 * @param {...*} [spec.options] - Remaining keys are passed to the method
 *   (maxIter, tol, learningRate, fTol, xTol, history, verbose, ...)
 * @returns {Object} {x, fx, iterations, converged, method, ...}
 */
export function minimize(spec = {}) {
  const { f, x0, method = 'neldermead', bounds, ...options } = spec;

  if (typeof f !== 'function') {
    throw new Error('minimize: spec.f must be a function');
  }
  if (!Array.isArray(x0) || x0.length === 0 || x0.some((v) => typeof v !== 'number')) {
    throw new Error('minimize: spec.x0 must be a non-empty array of numbers');
  }

  const key = String(method).toLowerCase();
  const minimizer = METHODS[key];
  if (!minimizer) {
    throw new Error(
      `minimize: unknown method '${method}'. Available: ${Object.keys(METHODS).join(', ')}`,
    );
  }

  // Box bounds: run the method in MINUIT-transformed internal space.
  // history/gradNorm are then reported in internal coordinates.
  const T = bounds ? makeBoundsTransform(bounds, x0.length) : null;
  if (T) {
    const wrapped = wrapObjective(f, options.grad, T);
    const result = minimizer(wrapped.f, T.toInternal(x0), { ...options, grad: wrapped.grad });
    result.x = T.toExternal(result.x);
    result.method = key;
    result.bounded = true;
    return result;
  }

  const result = minimizer(f, x0, options);
  result.method = key;
  return result;
}
