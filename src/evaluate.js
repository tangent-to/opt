/**
 * Objective-function normalization.
 *
 * All optimizers work against a single evaluator contract:
 *   evaluate(x) => { loss: number, gradient: Array<number> }
 *
 * Users may supply either:
 *   - f: (x) => number, plus an optional grad: (x) => Array<number>
 *   - f: (x) => { loss, gradient }  (tangent/ds loss-function style)
 *
 * When no gradient is available, central finite differences are used.
 */

import { numericalGradient } from './numdiff.js';

/**
 * Build a normalized evaluator from a user-supplied objective.
 *
 * @param {Function} f - Objective: (x) => number or (x) => {loss, gradient}
 * @param {Function} [grad] - Optional gradient: (x) => Array<number>
 * @param {Object} [options]
 * @param {number} [options.h] - Finite-difference step size
 * @returns {Function} evaluate(x) => {loss, gradient}
 */
export function makeEvaluator(f, grad, options = {}) {
  return function evaluate(x) {
    const out = f(x);

    // tangent/ds combined form: f returns {loss, gradient}
    if (out !== null && typeof out === 'object' && 'loss' in out) {
      return out;
    }

    if (typeof out !== 'number') {
      throw new Error(
        'Objective must return a number or an object of the form {loss, gradient}',
      );
    }

    return {
      loss: out,
      gradient: grad ? grad(x) : numericalGradient(f, x, options),
    };
  };
}

/**
 * Build a scalar-only evaluator (for derivative-free methods).
 *
 * @param {Function} f - Objective: (x) => number or (x) => {loss, gradient}
 * @returns {Function} (x) => number
 */
export function makeScalarEvaluator(f) {
  return function evaluateScalar(x) {
    const out = f(x);
    if (out !== null && typeof out === 'object' && 'loss' in out) {
      return out.loss;
    }
    if (typeof out !== 'number') {
      throw new Error(
        'Objective must return a number or an object of the form {loss, gradient}',
      );
    }
    return out;
  };
}
