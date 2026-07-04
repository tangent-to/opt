import { describe, expect, it } from 'vitest';
import { curveFit, leastSquares, makeBoundsTransform, minimize } from '../src/index.js';
import { booth, rosenbrock, rosenbrockGrad } from './functions.js';

describe('makeBoundsTransform', () => {
  it('round-trips interior points for all bound kinds', () => {
    const T = makeBoundsTransform([[0, 10], [2, null], [null, 5], [null, null]], 4);
    const xExt = [3.7, 8.1, -12.4, 0.5];
    const back = T.toExternal(T.toInternal(xExt));
    for (let i = 0; i < 4; i++) {
      expect(back[i]).toBeCloseTo(xExt[i], 6);
    }
  });

  it('returns null when every parameter is free', () => {
    expect(makeBoundsTransform([[null, null], [null, null]], 2)).toBeNull();
  });

  it('throws on inverted or degenerate bounds and wrong length', () => {
    expect(() => makeBoundsTransform([[3, 1]], 1)).toThrow(/strictly below/);
    expect(() => makeBoundsTransform([[2, 2]], 1)).toThrow(/strictly below/);
    expect(() => makeBoundsTransform([[0, 1]], 2)).toThrow(/array of 2/);
  });

  it('clamps points on or outside the bounds strictly inside', () => {
    const T = makeBoundsTransform([[0, 1]], 1);
    const inside = T.toExternal(T.toInternal([5]));
    expect(inside[0]).toBeLessThan(1);
    expect(inside[0]).toBeGreaterThan(0);
  });
});

describe('minimize with bounds', () => {
  it('leaves an interior optimum untouched (lbfgs)', () => {
    const result = minimize({
      f: booth,
      x0: [2, 2],
      method: 'lbfgs',
      bounds: [[0, 10], [0, 10]],
    });
    expect(result.bounded).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 4);
    expect(result.x[1]).toBeCloseTo(3, 4);
  });

  it('finds the active-bound optimum of Rosenbrock on a box (lbfgs)', () => {
    // Unconstrained min (1,1) lies outside; constrained min is (0.5, 0.25).
    const result = minimize({
      f: rosenbrock,
      grad: rosenbrockGrad,
      x0: [0, 0],
      method: 'lbfgs',
      bounds: [[-0.5, 0.5], [-0.5, 0.5]],
    });
    expect(result.x[0]).toBeCloseTo(0.5, 3);
    expect(result.x[1]).toBeCloseTo(0.25, 3);
    expect(result.x[0]).toBeLessThanOrEqual(0.5);
  });

  it('respects a one-sided bound with an active constraint (lbfgs)', () => {
    // min (x-5)^2 with x <= 2  ->  x = 2
    const result = minimize({
      f: (x) => (x[0] - 5) ** 2,
      x0: [0],
      method: 'lbfgs',
      bounds: [[null, 2]],
    });
    expect(result.x[0]).toBeCloseTo(2, 3);
    expect(result.x[0]).toBeLessThanOrEqual(2);
  });

  it('works with Nelder-Mead too (same transform, no gradients)', () => {
    const result = minimize({
      f: rosenbrock,
      x0: [0, 0],
      bounds: [[-0.5, 0.5], [-0.5, 0.5]],
      maxIter: 4000,
    });
    expect(result.x[0]).toBeCloseTo(0.5, 3);
    expect(result.x[1]).toBeCloseTo(0.25, 3);
  });

  it('never evaluates f outside the bounds', () => {
    let violations = 0;
    const f = (x) => {
      if (x[0] < 0 || x[0] > 1) violations++;
      return (x[0] - 5) ** 2;
    };
    minimize({ f, x0: [0.5], method: 'lbfgs', bounds: [[0, 1]] });
    expect(violations).toBe(0);
  });

  it('starts from an x0 outside the bounds by clamping it inside', () => {
    const result = minimize({
      f: booth,
      x0: [50, -50],
      method: 'lbfgs',
      bounds: [[0, 10], [0, 10]],
    });
    expect(result.x[0]).toBeCloseTo(1, 3);
    expect(result.x[1]).toBeCloseTo(3, 3);
  });
});

describe('leastSquares / curveFit with bounds', () => {
  it('hits an active bound when the unconstrained fit lies outside', () => {
    // OLS slope through the origin is ~2; bound it at 1.5.
    const xs = [1, 2, 3, 4, 5];
    const ys = xs.map((v) => 2 * v);
    const result = leastSquares({
      residuals: (p) => xs.map((v, i) => ys[i] - p[0] * v),
      x0: [1],
      bounds: [[null, 1.5]],
    });
    expect(result.bounded).toBe(true);
    expect(result.x[0]).toBeCloseTo(1.5, 3);
    expect(result.x[0]).toBeLessThanOrEqual(1.5);
  });

  it('curveFit recovers interior parameters unchanged under inactive bounds', () => {
    const xs = Array.from({ length: 81 }, (_, i) => i * 0.05);
    const ys = xs.map((v) => 2.5 * Math.exp(-1.3 * v) + 0.5);
    const { params, converged } = curveFit({
      model: (v, p) => p[0] * Math.exp(-p[1] * v) + p[2],
      x: xs,
      y: ys,
      p0: [1, 1, 0.1],
      bounds: [[0, null], [0, null], [null, null]],
    });
    expect(converged).toBe(true);
    expect(params[0]).toBeCloseTo(2.5, 5);
    expect(params[1]).toBeCloseTo(1.3, 5);
    expect(params[2]).toBeCloseTo(0.5, 5);
  });
});
