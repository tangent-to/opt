/**
 * Dense linear-system solver.
 *
 * Gaussian elimination with scaled partial pivoting for the small n-by-n
 * systems that arise in Levenberg-Marquardt normal equations. Inputs are
 * copied, never mutated.
 */

/**
 * Solve the linear system A x = b.
 *
 * @param {Array<Array<number>>} A - n-by-n coefficient matrix (not mutated)
 * @param {Array<number>} b - Right-hand side of length n (not mutated)
 * @returns {Array<number>} Solution vector x
 * @throws {Error} If A is not square, b has the wrong length, or A is singular
 */
export function solve(A, b) {
  if (!Array.isArray(A) || A.length === 0) {
    throw new Error('solve: A must be a non-empty square matrix');
  }
  const n = A.length;
  for (let i = 0; i < n; i++) {
    if (!Array.isArray(A[i]) || A[i].length !== n) {
      throw new Error('solve: A must be a non-empty square matrix');
    }
  }
  if (!Array.isArray(b) || b.length !== n) {
    throw new Error('solve: b must be an array with the same length as A');
  }

  // Working copies: elimination happens in place on these.
  const M = new Array(n);
  for (let i = 0; i < n; i++) {
    M[i] = A[i].slice();
  }
  const x = b.slice();

  // Row scale factors for scaled partial pivoting.
  const scale = new Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let j = 0; j < n; j++) {
      s = Math.max(s, Math.abs(M[i][j]));
    }
    if (s === 0) {
      throw new Error('solve: matrix is singular (zero row)');
    }
    scale[i] = s;
  }

  for (let k = 0; k < n; k++) {
    // Pick the pivot row maximizing |M[i][k]| / scale[i].
    let pivotRow = k;
    let pivotSize = Math.abs(M[k][k]) / scale[k];
    for (let i = k + 1; i < n; i++) {
      const size = Math.abs(M[i][k]) / scale[i];
      if (size > pivotSize) {
        pivotSize = size;
        pivotRow = i;
      }
    }
    if (!(pivotSize >= 1e-300)) {
      throw new Error('solve: matrix is singular to working precision');
    }
    if (pivotRow !== k) {
      const rowTmp = M[k];
      M[k] = M[pivotRow];
      M[pivotRow] = rowTmp;
      const bTmp = x[k];
      x[k] = x[pivotRow];
      x[pivotRow] = bTmp;
      const sTmp = scale[k];
      scale[k] = scale[pivotRow];
      scale[pivotRow] = sTmp;
    }

    const pivot = M[k][k];
    for (let i = k + 1; i < n; i++) {
      const factor = M[i][k] / pivot;
      if (factor === 0) continue;
      M[i][k] = 0;
      for (let j = k + 1; j < n; j++) {
        M[i][j] -= factor * M[k][j];
      }
      x[i] -= factor * x[k];
    }
  }

  // Back substitution.
  for (let i = n - 1; i >= 0; i--) {
    let sum = x[i];
    for (let j = i + 1; j < n; j++) {
      sum -= M[i][j] * x[j];
    }
    x[i] = sum / M[i][i];
  }

  return x;
}
