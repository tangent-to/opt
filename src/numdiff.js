/**
 * Numerical differentiation utilities
 */

/**
 * Approximate the gradient of a scalar function by central finite differences.
 *
 * @param {Function} f - Scalar function (x: Array<number>) => number
 * @param {Array<number>} x - Point at which to evaluate the gradient
 * @param {Object} [options]
 * @param {number} [options.h=1e-6] - Step size
 * @returns {Array<number>} Gradient approximation
 */
export function numericalGradient(f, x, options = {}) {
  const h = options.h || 1e-6;
  const n = x.length;
  const grad = new Array(n);
  const xi = x.slice();

  for (let i = 0; i < n; i++) {
    xi[i] = x[i] + h;
    const fPlus = f(xi);
    xi[i] = x[i] - h;
    const fMinus = f(xi);
    xi[i] = x[i];
    grad[i] = (fPlus - fMinus) / (2 * h);
  }

  return grad;
}
