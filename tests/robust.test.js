import { describe, expect, it } from 'vitest';
import { curveFit, leastSquares } from '../src/index.js';

/**
 * Deterministic linear data y = 2x + 1 with small noise and three gross
 * outliers. Robust losses should shrug the outliers off; plain least
 * squares gets dragged.
 */
function outlierData() {
  const xs = [];
  const ys = [];
  // "noise" from a fixed table so the test is deterministic
  const noise = [0.03, -0.05, 0.02, 0.04, -0.01, -0.04, 0.05, 0.01, -0.02, -0.03];
  for (let i = 0; i < 30; i++) {
    const x = i / 3;
    xs.push(x);
    ys.push(2 * x + 1 + noise[i % noise.length]);
  }
  ys[5] += 30;
  ys[14] -= 25;
  ys[23] += 40;
  return { xs, ys };
}

const linearModel = (x, p) => p[0] * x + p[1];

describe('robust losses', () => {
  const { xs, ys } = outlierData();
  const fit = (loss) =>
    curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0], loss, fScale: 0.1 });

  it('plain least squares is dragged by outliers', () => {
    const { params } = curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0] });
    const err = Math.abs(params[0] - 2) + Math.abs(params[1] - 1);
    expect(err).toBeGreaterThan(0.5); // visibly wrong
  });

  for (const loss of ['huber', 'soft_l1', 'cauchy']) {
    it(`${loss} loss recovers the true line despite outliers`, () => {
      const { params, converged } = fit(loss);
      expect(converged).toBe(true);
      expect(params[0]).toBeCloseTo(2, 1);
      expect(params[1]).toBeCloseTo(1, 1);
      // and is much closer than the non-robust fit
      const plain = curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0] }).params;
      const robustErr = Math.abs(params[0] - 2) + Math.abs(params[1] - 1);
      const plainErr = Math.abs(plain[0] - 2) + Math.abs(plain[1] - 1);
      expect(robustErr).toBeLessThan(plainErr / 5);
    });
  }

  it('linear loss stays the exact default (identical to omitting loss)', () => {
    const a = curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0] });
    const b = curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0], loss: 'linear' });
    expect(a.params[0]).toBe(b.params[0]);
    expect(a.params[1]).toBe(b.params[1]);
  });

  it('robust fits report NaN stdErr (Gaussian formula does not apply)', () => {
    const { stdErr } = fit('huber');
    expect(stdErr.every(Number.isNaN)).toBe(true);
    const plain = curveFit({ model: linearModel, x: xs, y: ys, p0: [1, 0] });
    expect(plain.stdErr.every(Number.isFinite)).toBe(true);
  });

  it('throws on unknown loss names', () => {
    expect(() =>
      leastSquares({ residuals: (p) => [p[0]], x0: [1], loss: 'tukey' }),
    ).toThrow(/unknown loss 'tukey'/);
  });

  it('robust loss composes with bounds', () => {
    const { params } = curveFit({
      model: linearModel,
      x: xs,
      y: ys,
      p0: [1, 0],
      loss: 'huber',
      fScale: 0.1,
      bounds: [[0, 5], [null, null]],
    });
    expect(params[0]).toBeCloseTo(2, 1);
    expect(params[1]).toBeCloseTo(1, 1);
  });
});
