import { describe, expect, it } from 'vitest';
import { methods, minimize } from '../src/index.js';
import { booth, rosenbrock, rosenbrockGrad, sphere, sphereGrad } from './functions.js';

describe('minimize (declarative API)', () => {
  it('defaults to Nelder-Mead', () => {
    const result = minimize({ f: booth, x0: [0, 0] });
    expect(result.method).toBe('neldermead');
    expect(result.x[0]).toBeCloseTo(1, 3);
    expect(result.x[1]).toBeCloseTo(3, 3);
  });

  it('dispatches to adam with a gradient', () => {
    const result = minimize({
      f: sphere,
      grad: sphereGrad,
      x0: [3, -2],
      method: 'adam',
      learningRate: 0.1,
      maxIter: 5000,
    });
    expect(result.method).toBe('adam');
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-10);
  });

  it('passes method options through (Nelder-Mead)', () => {
    const result = minimize({ f: rosenbrock, x0: [-1.2, 1], maxIter: 2000, history: true });
    expect(result.fx).toBeLessThan(1e-6);
    expect(result.history.length).toBeGreaterThan(0);
  });

  it('accepts gradient descent aliases', () => {
    for (const method of ['gd', 'sgd', 'gradient_descent']) {
      const result = minimize({
        f: sphere,
        grad: sphereGrad,
        x0: [1, 1],
        method,
        learningRate: 0.1,
        maxIter: 2000,
      });
      expect(result.converged).toBe(true);
    }
  });

  it('is case-insensitive about method names', () => {
    const result = minimize({ f: booth, x0: [0, 0], method: 'Nelder-Mead' });
    expect(result.method).toBe('nelder-mead');
    expect(result.fx).toBeLessThan(1e-6);
  });

  it('throws on unknown methods', () => {
    expect(() => minimize({ f: sphere, x0: [1], method: 'bfgs' })).toThrow(/unknown method/);
  });

  it('validates f and x0', () => {
    expect(() => minimize({ x0: [1] })).toThrow(/f must be a function/);
    expect(() => minimize({ f: sphere })).toThrow(/x0 must be a non-empty array/);
    expect(() => minimize({ f: sphere, x0: [] })).toThrow(/x0 must be a non-empty array/);
    expect(() => minimize({ f: sphere, x0: [1, 'a'] })).toThrow(/x0 must be a non-empty array/);
  });

  it('lists available methods', () => {
    expect(methods()).toContain('neldermead');
    expect(methods()).toContain('adam');
  });

  it('does not mutate x0', () => {
    const x0 = [0, 0];
    minimize({ f: booth, x0 });
    expect(x0).toEqual([0, 0]);
  });
});
