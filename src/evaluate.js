/**
 * Objective-function normalization.
 *
 * All optimizers work against a single evaluator contract:
 *   evaluate(x) => { loss: number, gradient: Array<number> }
 *
 * Users may supply either:
 *   - f: (x) => number, plus an optional grad: (x) => Array<number>
 *   - f: (x) => { loss, gradient }  (tangent/ds loss-function style)
 *   - f: (x) => { value, gradient } (what @tangent.to/grad's valueAndGrad
 *     and compile return, so one of those is an objective as it stands)
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

    // Combined forms: {loss, gradient} from tangent/ds, {value, gradient} from grad.
    if (out !== null && typeof out === 'object') {
      if ('loss' in out) return out;
      if ('value' in out) return { loss: out.value, gradient: out.gradient };
    }

    if (typeof out !== 'number') {
      throw new Error(
        'Objective must return a number or an object of the form {loss, gradient} or {value, gradient}',
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
    if (out !== null && typeof out === 'object') {
      if ('loss' in out) return out.loss;
      if ('value' in out) return out.value;
    }
    if (typeof out !== 'number') {
      throw new Error(
        'Objective must return a number or an object of the form {loss, gradient} or {value, gradient}',
      );
    }
    return out;
  };
}
