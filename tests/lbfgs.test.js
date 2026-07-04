import { describe, expect, it } from 'vitest';
import { lbfgs } from '../src/lbfgs.js';
import { minimize } from '../src/minimize.js';
import { beale, booth, himmelblau, rosenbrock, rosenbrockGrad, sphere, sphereGrad } from './functions.js';

describe('lbfgs', () => {
  it('minimizes Rosenbrock from the classic start point with analytic gradient', () => {
    const result = lbfgs(rosenbrock, [-1.2, 1], { grad: rosenbrockGrad });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 6);
    expect(result.x[1]).toBeCloseTo(1, 6);
    expect(result.fx).toBeLessThan(1e-12);
    // Quasi-Newton should be dramatically cheaper than first-order methods
    expect(result.iterations).toBeLessThan(100);
  });

  it('minimizes Rosenbrock with finite-difference gradients', () => {
    const result = lbfgs(rosenbrock, [-1.2, 1]);
    expect(result.x[0]).toBeCloseTo(1, 4);
    expect(result.x[1]).toBeCloseTo(1, 4);
  });

  it('minimizes a high-dimensional quadratic quickly', () => {
    const n = 100;
    const x0 = Array.from({ length: n }, (_, i) => (i % 2 ? 3 : -2));
    const result = lbfgs(sphere, x0, { grad: sphereGrad });
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-10);
    expect(result.iterations).toBeLessThan(20);
  });

  it('minimizes an ill-conditioned quadratic (condition number 1e6)', () => {
    const scales = [1, 1e2, 1e3];
    const f = (x) => scales.reduce((s, c, i) => s + c * x[i] * x[i], 0);
    const grad = (x) => x.map((xi, i) => 2 * scales[i] * xi);
    const result = lbfgs(f, [1, 1, 1], { grad, maxIter: 500 });
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-8);
  });

  it('minimizes Booth, Himmelblau and Beale', () => {
    for (const [f, x0, xmin] of [
      [booth, [0, 0], [1, 3]],
      [himmelblau, [4, 3], [3, 2]],
      [beale, [1, 1], [3, 0.5]],
    ]) {
      const result = lbfgs(f, x0);
      expect(result.converged).toBe(true);
      expect(result.x[0]).toBeCloseTo(xmin[0], 4);
      expect(result.x[1]).toBeCloseTo(xmin[1], 4);
    }
  });

  it('accepts the combined {loss, gradient} objective form', () => {
    const lossFn = (x) => ({ loss: rosenbrock(x), gradient: rosenbrockGrad(x) });
    const result = lbfgs(lossFn, [-1.2, 1]);
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 6);
  });

  it('handles objectives with non-finite regions', () => {
    // f(x) = -log(x0) + x0 + x1^2, domain x0 > 0, min at x0=1, x1=0
    const f = (x) => -Math.log(x[0]) + x[0] + x[1] * x[1];
    const grad = (x) => [-1 / x[0] + 1, 2 * x[1]];
    const result = lbfgs(f, [3, 1], { grad });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 6);
    expect(result.x[1]).toBeCloseTo(0, 6);
  });

  it('uses far fewer function evaluations than Nelder-Mead on Rosenbrock', () => {
    const nm = minimize({ f: rosenbrock, x0: [-1.2, 1], maxIter: 5000 });
    const lb = lbfgs(rosenbrock, [-1.2, 1], { grad: rosenbrockGrad });
    expect(lb.fevals).toBeLessThan(nm.fevals / 2);
  });

  it('respects the memory option and tracks history', () => {
    const result = lbfgs(rosenbrock, [-1.2, 1], { grad: rosenbrockGrad, memory: 3 });
    expect(result.converged).toBe(true);
    expect(result.history.loss.length).toBeGreaterThan(1);
    expect(result.history.gradNorm.length).toBe(result.history.loss.length);
    // loss history is non-increasing (Wolfe guarantees decrease)
    for (let i = 1; i < result.history.loss.length; i++) {
      expect(result.history.loss[i]).toBeLessThanOrEqual(result.history.loss[i - 1] + 1e-15);
    }
  });

  it('is dispatchable through minimize()', () => {
    const result = minimize({
      f: rosenbrock,
      grad: rosenbrockGrad,
      x0: [-1.2, 1],
      method: 'lbfgs',
    });
    expect(result.method).toBe('lbfgs');
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 5);
  });

  it('does not mutate x0', () => {
    const x0 = [-1.2, 1];
    lbfgs(rosenbrock, x0, { grad: rosenbrockGrad });
    expect(x0).toEqual([-1.2, 1]);
  });
});
