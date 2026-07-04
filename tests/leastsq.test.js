import { describe, expect, it } from 'vitest';
import { solve } from '../src/linsolve.js';
import { curveFit, leastSquares } from '../src/leastsq.js';

describe('solve', () => {
  it('solves a known 3x3 system', () => {
    const A = [
      [2, 1, -1],
      [-3, -1, 2],
      [-2, 1, 2],
    ];
    const b = [8, -11, -3];
    const x = solve(A, b);
    expect(x[0]).toBeCloseTo(2, 10);
    expect(x[1]).toBeCloseTo(3, 10);
    expect(x[2]).toBeCloseTo(-1, 10);
  });

  it('handles a system that needs pivoting (zero on the diagonal)', () => {
    const A = [
      [0, 1, 1],
      [2, 0, 1],
      [1, 1, 0],
    ];
    const b = [4, 7, 3]; // x = [2, 1, 3]
    const x = solve(A, b);
    expect(x[0]).toBeCloseTo(2, 10);
    expect(x[1]).toBeCloseTo(1, 10);
    expect(x[2]).toBeCloseTo(3, 10);
  });

  it('does not mutate its inputs', () => {
    const A = [
      [0, 1],
      [1, 0],
    ];
    const b = [4, 5];
    solve(A, b);
    expect(A).toEqual([[0, 1], [1, 0]]);
    expect(b).toEqual([4, 5]);
  });

  it('throws a clear error on a singular matrix', () => {
    const A = [
      [1, 2],
      [2, 4],
    ];
    expect(() => solve(A, [1, 2])).toThrow(/singular/);
  });

  it('throws on a zero matrix', () => {
    expect(() => solve([[0, 0], [0, 0]], [1, 1])).toThrow(/singular/);
  });

  it('throws on shape mismatches', () => {
    expect(() => solve([[1, 2]], [1])).toThrow(/square/);
    expect(() => solve([[1, 0], [0, 1]], [1])).toThrow(/length/);
  });
});

// Rosenbrock in least-squares form: cost = 0.5 * (r1^2 + r2^2)
const rosenbrockResiduals = (p) => [1 - p[0], 10 * (p[1] - p[0] * p[0])];
const rosenbrockJacobian = (p) => [
  [-1, 0],
  [-20 * p[0], 10],
];

describe('leastSquares', () => {
  it('requires residuals and x0', () => {
    expect(() => leastSquares({ x0: [0] })).toThrow(/residuals/);
    expect(() => leastSquares({ residuals: (p) => [p[0]] })).toThrow(/x0/);
  });

  it('minimizes Rosenbrock with an analytic jacobian', () => {
    const result = leastSquares({
      residuals: rosenbrockResiduals,
      jacobian: rosenbrockJacobian,
      x0: [-1.2, 1],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x[0] - 1)).toBeLessThan(1e-8);
    expect(Math.abs(result.x[1] - 1)).toBeLessThan(1e-8);
    expect(result.fx).toBeLessThan(1e-16);
  });

  it('minimizes Rosenbrock with a finite-difference jacobian', () => {
    const result = leastSquares({
      residuals: rosenbrockResiduals,
      x0: [-1.2, 1],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x[0] - 1)).toBeLessThan(1e-8);
    expect(Math.abs(result.x[1] - 1)).toBeLessThan(1e-8);
  });

  it('recovers exact parameters of an exponential decay from noiseless data', () => {
    const trueP = [2, 0.5];
    const t = Array.from({ length: 31 }, (_, i) => i * 0.1);
    const data = t.map((ti) => trueP[0] * Math.exp(-trueP[1] * ti));
    const result = leastSquares({
      residuals: (p) => t.map((ti, i) => data[i] - p[0] * Math.exp(-p[1] * ti)),
      x0: [1, 1],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x[0] - trueP[0])).toBeLessThan(1e-6);
    expect(Math.abs(result.x[1] - trueP[1])).toBeLessThan(1e-6);
  });

  it('reports sane iteration and evaluation counts', () => {
    const result = leastSquares({
      residuals: rosenbrockResiduals,
      jacobian: rosenbrockJacobian,
      x0: [-1.2, 1],
    });
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.iterations).toBeLessThan(200);
    expect(result.fevals).toBeGreaterThan(result.iterations);
    expect(result.residuals).toHaveLength(2);
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('records strictly decreasing cost in history', () => {
    const result = leastSquares({
      residuals: rosenbrockResiduals,
      jacobian: rosenbrockJacobian,
      x0: [-1.2, 1],
      history: true,
    });
    expect(result.history).toHaveLength(result.iterations);
    for (const entry of result.history) {
      expect(entry).toHaveProperty('cost');
      expect(entry).toHaveProperty('lambda');
    }
    for (let i = 1; i < result.history.length; i++) {
      expect(result.history[i].cost).toBeLessThan(result.history[i - 1].cost);
    }
  });

  it('treats non-finite trial residuals as rejected steps, not crashes', () => {
    // sqrt residual: NaN whenever a trial step drives p[0] + p[1]*t negative.
    const t = Array.from({ length: 21 }, (_, i) => i * 0.1);
    const data = t.map((ti) => Math.sqrt(1 + 0.5 * ti));
    const result = leastSquares({
      residuals: (p) => t.map((ti, i) => data[i] - Math.sqrt(p[0] + p[1] * ti)),
      x0: [2, 1],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x[0] - 1)).toBeLessThan(1e-6);
    expect(Math.abs(result.x[1] - 0.5)).toBeLessThan(1e-6);
  });

  it('throws if residuals are not finite at x0', () => {
    expect(() =>
      leastSquares({ residuals: (p) => [Math.sqrt(-1 - p[0] * p[0])], x0: [1] })
    ).toThrow(/finite/);
  });
});

describe('curveFit', () => {
  it('recovers exponential model parameters from noiseless data', () => {
    const [a, b, c] = [2.5, 1.3, 0.5];
    const x = [];
    for (let xi = 0; xi <= 4 + 1e-12; xi += 0.05) {
      x.push(xi);
    }
    const y = x.map((xi) => a * Math.exp(-b * xi) + c);
    const result = curveFit({
      model: (xi, p) => p[0] * Math.exp(-p[1] * xi) + p[2],
      x,
      y,
      p0: [1, 1, 0],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.params[0] - a)).toBeLessThan(1e-6);
    expect(Math.abs(result.params[1] - b)).toBeLessThan(1e-6);
    expect(Math.abs(result.params[2] - c)).toBeLessThan(1e-6);
    for (const se of result.stdErr) {
      expect(Number.isFinite(se)).toBe(true);
      expect(se).toBeLessThan(1e-3);
    }
    expect(result.cov).toHaveLength(3);
    expect(result.cov[0]).toHaveLength(3);
  });

  it('matches ordinary least squares closed form for a linear model', () => {
    const x = Array.from({ length: 25 }, (_, i) => i * 0.25);
    // Deterministic scatter so the fit is non-trivial.
    const y = x.map((xi, i) => 1.5 + 0.7 * xi + 0.1 * Math.sin(3 * i));

    // Closed-form OLS: slope and intercept.
    const m = x.length;
    const sx = x.reduce((s, v) => s + v, 0);
    const sy = y.reduce((s, v) => s + v, 0);
    const sxx = x.reduce((s, v) => s + v * v, 0);
    const sxy = x.reduce((s, v, i) => s + v * y[i], 0);
    const slope = (m * sxy - sx * sy) / (m * sxx - sx * sx);
    const intercept = (sy - slope * sx) / m;

    const result = curveFit({
      model: (xi, p) => p[0] + p[1] * xi,
      x,
      y,
      p0: [0, 0],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.params[0] - intercept)).toBeLessThan(1e-6);
    expect(Math.abs(result.params[1] - slope)).toBeLessThan(1e-6);
    expect(Number.isFinite(result.stdErr[0])).toBe(true);
    expect(Number.isFinite(result.stdErr[1])).toBe(true);
  });

  it('throws a clear message on mismatched x/y lengths', () => {
    expect(() =>
      curveFit({
        model: (xi, p) => p[0] * xi,
        x: [1, 2, 3],
        y: [1, 2],
        p0: [1],
      })
    ).toThrow(/same length/);
  });

  it('throws if p0 is missing', () => {
    expect(() =>
      curveFit({
        model: (xi, p) => p[0] * xi,
        x: [1, 2, 3],
        y: [1, 2, 3],
      })
    ).toThrow(/p0/);
  });

  it('throws if model is missing', () => {
    expect(() => curveFit({ x: [1], y: [1], p0: [1] })).toThrow(/model/);
  });

  it('converges for a model with a restricted domain when started inside it', () => {
    const x = Array.from({ length: 41 }, (_, i) => i * 0.1);
    const y = x.map((xi) => 2 * Math.sqrt(1 + 0.8 * xi));
    const result = curveFit({
      model: (xi, p) => p[0] * Math.sqrt(1 + p[1] * xi),
      x,
      y,
      p0: [1, 1],
    });
    expect(result.converged).toBe(true);
    expect(Math.abs(result.params[0] - 2)).toBeLessThan(1e-6);
    expect(Math.abs(result.params[1] - 0.8)).toBeLessThan(1e-6);
  });

  it('fills the covariance with NaN when J^T J is singular', () => {
    // Redundant parameters: model depends only on p[0] + p[1].
    const x = [0, 1, 2, 3];
    const y = [1, 2, 3, 4];
    const result = curveFit({
      model: (xi, p) => (p[0] + p[1]) * xi,
      x,
      y,
      p0: [0.5, 0.5],
      maxIter: 20,
    });
    expect(result.cov[0][0]).toBeNaN();
    expect(result.stdErr[0]).toBeNaN();
    expect(result.stdErr[1]).toBeNaN();
  });
});
