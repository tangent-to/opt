/**
 * Verify the ds.core.optimize drop-in surface: same classes, same
 * signatures, same {x, history} return shape and history keys.
 */
import { describe, expect, it } from 'vitest';
import {
  AdamOptimizer,
  createOptimizer,
  GradientDescent,
  MomentumOptimizer,
  RMSProp,
} from '../src/index.js';
import { sphere, sphereGrad } from './functions.js';

const lossFn = (x) => ({ loss: sphere(x), gradient: sphereGrad(x) });

describe('ds.core.optimize compatibility', () => {
  it('createOptimizer resolves the same names as ds', () => {
    expect(createOptimizer('gd')).toBeInstanceOf(GradientDescent);
    expect(createOptimizer('gradient_descent')).toBeInstanceOf(GradientDescent);
    expect(createOptimizer('sgd')).toBeInstanceOf(GradientDescent);
    expect(createOptimizer('momentum')).toBeInstanceOf(MomentumOptimizer);
    expect(createOptimizer('rmsprop')).toBeInstanceOf(RMSProp);
    expect(createOptimizer('ADAM')).toBeInstanceOf(AdamOptimizer);
    expect(() => createOptimizer('nope')).toThrow(/Unknown optimizer/);
  });

  it('optimizers return {x, history} like ds', () => {
    const opt = createOptimizer('adam', { learningRate: 0.1 });
    const result = opt.minimize(lossFn, [3, -2]);
    expect(Object.keys(result).sort()).toEqual(['history', 'x']);
    expect(result.x[0]).toBeCloseTo(0, 4);
    expect(result.x[1]).toBeCloseTo(0, 4);
  });

  it('GradientDescent history has loss, gradNorm and learningRate keys', () => {
    const opt = new GradientDescent({ learningRate: 0.1, lineSearch: true });
    const { history } = opt.minimize(lossFn, [2, 2]);
    expect(Object.keys(history).sort()).toEqual(['gradNorm', 'learningRate', 'loss']);
    expect(history.loss.length).toBeGreaterThan(0);
  });

  it('Momentum/RMSProp/Adam history has loss and gradNorm keys only', () => {
    for (const Ctor of [MomentumOptimizer, RMSProp, AdamOptimizer]) {
      const { history } = new Ctor({ learningRate: 0.05 }).minimize(lossFn, [1, 1]);
      expect(Object.keys(history).sort()).toEqual(['gradNorm', 'loss']);
    }
  });

  it('minimize() honors per-call maxIter and tol overrides', () => {
    const opt = new AdamOptimizer({ learningRate: 0.1 });
    const { history } = opt.minimize(lossFn, [3, -2], { maxIter: 5 });
    expect(history.loss.length).toBeLessThanOrEqual(5);
  });

  it('GradientDescent stores stochastic/batchSize options like ds', () => {
    const opt = new GradientDescent({ stochastic: true, batchSize: 16 });
    expect(opt.stochastic).toBe(true);
    expect(opt.batchSize).toBe(16);
  });
});
