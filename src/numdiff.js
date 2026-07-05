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

/**
 * Central finite-difference Jacobian of a vector-valued function.
 * Per-component step h_j = step * max(1, |x_j|).
 *
 * @param {Function} f - (x: Array<number>) => Array<number> of length m
 * @param {Array<number>} x - Point of length n
 * @param {Object} [options]
 * @param {number} [options.step=1e-6] - Relative step size
 * @returns {Array<Array<number>>} m-by-n Jacobian
 */
export function numericalJacobian(f, x, options = {}) {
  const step = options.step || 1e-6;
  const n = x.length;
  const xi = x.slice();
  const cols = new Array(n);
  let m = -1;
  for (let j = 0; j < n; j++) {
    const h = step * Math.max(1, Math.abs(x[j]));
    xi[j] = x[j] + h;
    const fPlus = f(xi);
    xi[j] = x[j] - h;
    const fMinus = f(xi);
    xi[j] = x[j];
    if (m < 0) m = fPlus.length;
    const col = new Array(m);
    for (let i = 0; i < m; i++) col[i] = (fPlus[i] - fMinus[i]) / (2 * h);
    cols[j] = col;
  }
  // Transpose columns into an m-by-n row-major matrix
  const J = new Array(m);
  for (let i = 0; i < m; i++) {
    J[i] = new Array(n);
    for (let j = 0; j < n; j++) J[i][j] = cols[j][i];
  }
  return J;
}

/**
 * Central finite-difference Hessian of a scalar function.
 *
 * @param {Function} f - (x: Array<number>) => number
 * @param {Array<number>} x - Point of length n
 * @param {Object} [options]
 * @param {number} [options.step=1e-4] - Relative step size
 * @returns {Array<Array<number>>} n-by-n symmetric Hessian
 */
export function numericalHessian(f, x, options = {}) {
  const h = options.step || 1e-4;
  const n = x.length;
  const step = x.map((v) => h * Math.max(1, Math.abs(v)));
  const f0 = f(x);
  const H = Array.from({ length: n }, () => new Array(n).fill(0));
  const e = x.slice();
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      if (i === j) {
        e[i] = x[i] + step[i];
        const fp = f(e);
        e[i] = x[i] - step[i];
        const fm = f(e);
        e[i] = x[i];
        H[i][i] = (fp - 2 * f0 + fm) / (step[i] * step[i]);
      } else {
        e[i] = x[i] + step[i];
        e[j] = x[j] + step[j];
        const fpp = f(e);
        e[j] = x[j] - step[j];
        const fpm = f(e);
        e[i] = x[i] - step[i];
        const fmm = f(e);
        e[j] = x[j] + step[j];
        const fmp = f(e);
        e[i] = x[i];
        e[j] = x[j];
        H[i][j] = (fpp - fpm - fmp + fmm) / (4 * step[i] * step[j]);
        H[j][i] = H[i][j];
      }
    }
  }
  return H;
}
