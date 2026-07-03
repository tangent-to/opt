import { describe, expect, it } from 'vitest';
import { adam, gradientDescent, momentumDescent, numericalGradient, rmsprop } from '../src/index.js';
import { rosenbrock, rosenbrockGrad, sphere, sphereGrad } from './functions.js';

describe('gradientDescent', () => {
  it('minimizes the sphere with an explicit gradient', () => {
    const result = gradientDescent(sphere, [3, -2], {
      grad: sphereGrad,
      learningRate: 0.1,
      maxIter: 2000,
    });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(0, 5);
    expect(result.x[1]).toBeCloseTo(0, 5);
  });

  it('supports backtracking line search', () => {
    const result = gradientDescent(sphere, [2, 2], {
      grad: sphereGrad,
      lineSearch: true,
      maxIter: 500,
    });
    expect(result.history.learningRate.length).toBeGreaterThan(0);
    // line search picks larger steps than the default 0.01 learning rate
    expect(Math.max(...result.history.learningRate)).toBeGreaterThan(0.01);
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-10);
  });

  it('falls back to numerical gradients when none is given', () => {
    const result = gradientDescent(sphere, [1, 1], {
      learningRate: 0.1,
      maxIter: 2000,
      tol: 1e-4,
    });
    expect(result.fx).toBeLessThan(1e-6);
  });

  it('accepts the combined {loss, gradient} form', () => {
    const lossFn = (x) => ({ loss: sphere(x), gradient: sphereGrad(x) });
    const result = gradientDescent(lossFn, [2, 2], { learningRate: 0.1, maxIter: 2000 });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(0, 5);
  });
});

describe('momentumDescent', () => {
  it('minimizes the sphere', () => {
    const result = momentumDescent(sphere, [3, -2], {
      grad: sphereGrad,
      learningRate: 0.05,
      maxIter: 2000,
    });
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-10);
  });
});

describe('rmsprop', () => {
  it('minimizes the sphere', () => {
    const result = rmsprop(sphere, [3, -2], {
      grad: sphereGrad,
      learningRate: 0.05,
      maxIter: 5000,
    });
    expect(result.fx).toBeLessThan(1e-6);
  });
});

describe('adam', () => {
  it('minimizes the sphere', () => {
    const result = adam(sphere, [3, -2], {
      grad: sphereGrad,
      learningRate: 0.1,
      maxIter: 5000,
    });
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-10);
  });

  it('makes progress on Rosenbrock', () => {
    const result = adam(rosenbrock, [-1.2, 1], {
      grad: rosenbrockGrad,
      learningRate: 0.01,
      maxIter: 5000,
    });
    expect(result.fx).toBeLessThan(0.1);
  });

  it('tracks loss and gradient norm history', () => {
    const result = adam(sphere, [1, 1], { grad: sphereGrad, maxIter: 50 });
    expect(result.history.loss.length).toBeGreaterThan(0);
    expect(result.history.gradNorm.length).toBe(result.history.loss.length);
  });
});

describe('numericalGradient', () => {
  it('approximates the sphere gradient', () => {
    const g = numericalGradient(sphere, [1, -2, 3]);
    const expected = sphereGrad([1, -2, 3]);
    for (let i = 0; i < g.length; i++) {
      expect(g[i]).toBeCloseTo(expected[i], 5);
    }
  });
});
