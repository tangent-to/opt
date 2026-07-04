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
import { adam, gradientDescent, momentumDescent, rmsprop } from './gradient.js';

const METHODS = {
  neldermead: nelderMead,
  'nelder-mead': nelderMead,
  lbfgs: lbfgs,
  'l-bfgs': lbfgs,
  gd: gradientDescent,
  sgd: gradientDescent,
  gradient_descent: gradientDescent,
  momentum: momentumDescent,
  rmsprop: rmsprop,
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
  const { f, x0, method = 'neldermead', ...options } = spec;

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

  const result = minimizer(f, x0, options);
  result.method = key;
  return result;
}
