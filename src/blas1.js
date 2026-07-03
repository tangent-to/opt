/**
 * Dense vector helpers (BLAS level-1 style).
 *
 * Ported from fmin (https://github.com/benfred/fmin),
 * Copyright 2016, Ben Frederickson, BSD-3-Clause.
 * See THIRD_PARTY_NOTICES.md.
 */

/**
 * Dot product of two vectors
 * @param {Array<number>} a
 * @param {Array<number>} b
 * @returns {number}
 */
export function dot(a, b) {
  let ret = 0;
  for (let i = 0; i < a.length; ++i) {
    ret += a[i] * b[i];
  }
  return ret;
}

/**
 * Euclidean norm of a vector
 * @param {Array<number>} a
 * @returns {number}
 */
export function norm2(a) {
  return Math.sqrt(dot(a, a));
}

/**
 * In-place weighted sum: out = w1 * v1 + w2 * v2
 * @param {Array<number>} out - Output vector (mutated)
 * @param {number} w1
 * @param {Array<number>} v1
 * @param {number} w2
 * @param {Array<number>} v2
 */
export function weightedSum(out, w1, v1, w2, v2) {
  for (let i = 0; i < out.length; ++i) {
    out[i] = w1 * v1[i] + w2 * v2[i];
  }
}

/**
 * In-place scale: out = c * value
 * @param {Array<number>} out - Output vector (mutated)
 * @param {Array<number>} value
 * @param {number} c
 */
export function scale(out, value, c) {
  for (let i = 0; i < out.length; ++i) {
    out[i] = value[i] * c;
  }
}
