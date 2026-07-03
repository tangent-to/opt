import { describe, expect, it } from 'vitest';
import { nelderMead } from '../src/index.js';
import { beale, booth, himmelblau, rosenbrock } from './functions.js';

describe('nelderMead', () => {
  it('minimizes the Rosenbrock function from the classic start point', () => {
    const result = nelderMead(rosenbrock, [-1.2, 1], { maxIter: 2000 });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 3);
    expect(result.x[1]).toBeCloseTo(1, 3);
    expect(result.fx).toBeLessThan(1e-6);
  });

  it('minimizes the Booth function', () => {
    const result = nelderMead(booth, [0, 0]);
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(1, 3);
    expect(result.x[1]).toBeCloseTo(3, 3);
  });

  it('finds a Himmelblau minimum', () => {
    const result = nelderMead(himmelblau, [4, 3]);
    expect(result.converged).toBe(true);
    expect(result.fx).toBeLessThan(1e-6);
  });

  it('minimizes the Beale function', () => {
    const result = nelderMead(beale, [1, 1], { maxIter: 2000 });
    expect(result.converged).toBe(true);
    expect(result.x[0]).toBeCloseTo(3, 2);
    expect(result.x[1]).toBeCloseTo(0.5, 2);
  });

  it('reports iterations and function evaluations', () => {
    const result = nelderMead(booth, [0, 0]);
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.fevals).toBeGreaterThan(result.iterations);
  });

  it('records history when requested', () => {
    const result = nelderMead(booth, [0, 0], { history: true });
    expect(result.history.length).toBe(result.iterations + 1);
    expect(result.history[0]).toHaveProperty('x');
    expect(result.history[0]).toHaveProperty('fx');
    expect(result.history[0].simplex).toHaveLength(3);
    // history fx is non-increasing (best vertex per iteration)
    for (let i = 1; i < result.history.length; i++) {
      expect(result.history[i].fx).toBeLessThanOrEqual(result.history[i - 1].fx);
    }
  });

  it('accepts objectives in combined {loss, gradient} form', () => {
    const lossFn = (x) => ({ loss: booth(x), gradient: null });
    const result = nelderMead(lossFn, [0, 0]);
    expect(result.x[0]).toBeCloseTo(1, 3);
    expect(result.x[1]).toBeCloseTo(3, 3);
  });

  it('handles zero-valued starting coordinates', () => {
    const result = nelderMead(booth, [0, 0]);
    expect(Number.isFinite(result.fx)).toBe(true);
  });

  it('returns a plain array for x', () => {
    const result = nelderMead(booth, [0, 0]);
    expect(Array.isArray(result.x)).toBe(true);
    expect(result.x).not.toHaveProperty('fx');
  });
});
