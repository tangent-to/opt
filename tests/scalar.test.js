import { describe, expect, it } from 'vitest';
import { minimizeScalar, rootScalar } from '../src/scalar.js';

describe('minimizeScalar', () => {
  it('minimizes cos(x) near x = pi', () => {
    const result = minimizeScalar(Math.cos, { bracket: [2, 4] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(Math.PI, 6);
    expect(result.fx).toBeCloseTo(-1, 10);
  });

  it('handles the flat minimum of (x - 2)^4', () => {
    const result = minimizeScalar((x) => (x - 2) ** 4, {});
    expect(result.converged).toBe(true);
    expect(Math.abs(result.x - 2)).toBeLessThan(1e-3);
    expect(result.fx).toBeLessThan(1e-12);
  });

  it('minimizes a parabola with an explicit 3-point bracket', () => {
    const result = minimizeScalar((x) => (x - 1) ** 2, { bracket: [-2, 0.5, 3] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(1, 7);
    expect(result.fx).toBeCloseTo(0, 12);
  });

  it('golden and brent agree on the same problem', () => {
    const brent = minimizeScalar(Math.cos, { bracket: [2, 4], method: 'brent' });
    const golden = minimizeScalar(Math.cos, { bracket: [2, 4], method: 'golden' });
    expect(brent.converged).toBe(true);
    expect(golden.converged).toBe(true);
    expect(Math.abs(brent.x - golden.x)).toBeLessThan(1e-5);
    expect(golden.x).toBeCloseTo(Math.PI, 5);
  });

  it('auto-brackets from the default [0, 1] when the minimum is far away', () => {
    const result = minimizeScalar((x) => (x - 15) ** 2);
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(15, 6);
  });

  it('auto-brackets downhill to the left as well', () => {
    const result = minimizeScalar((x) => (x + 12) ** 2 + 3);
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(-12, 6);
    expect(result.fx).toBeCloseTo(3, 8);
  });

  it('throws a clear error when bracketing a monotonic function fails', () => {
    expect(() => minimizeScalar((x) => x)).toThrow(/bracket/i);
    expect(() => minimizeScalar((x) => -x)).toThrow(/bracket/i);
  });

  it('throws when f is non-finite at the initial bracket points', () => {
    expect(() => minimizeScalar((x) => Math.log(x), { bracket: [-2, -1] }))
      .toThrow(/not finite/i);
  });

  it('throws on an unknown method and on a malformed bracket', () => {
    expect(() => minimizeScalar((x) => x * x, { method: 'newton' })).toThrow(/unknown method/i);
    expect(() => minimizeScalar((x) => x * x, { bracket: [0] })).toThrow(/bracket/i);
    expect(() => minimizeScalar((x) => x * x, { bracket: [3, 0, 1] })).toThrow(/ordered/i);
  });

  it('reports iterations and function evaluations', () => {
    const result = minimizeScalar(Math.cos, { bracket: [2, 4] });
    expect(result.iterations).toBeGreaterThan(0);
    expect(result.fevals).toBeGreaterThan(result.iterations);
    expect(Number.isInteger(result.iterations)).toBe(true);
    expect(Number.isInteger(result.fevals)).toBe(true);
  });

  it('brent uses far fewer fevals than golden on a smooth quartic', () => {
    const quartic = (x) => (x - 1.3) ** 4 + (x - 1.3) ** 2;
    const brent = minimizeScalar(quartic, { bracket: [0, 1, 3], method: 'brent' });
    const golden = minimizeScalar(quartic, { bracket: [0, 1, 3], method: 'golden' });
    expect(brent.converged).toBe(true);
    expect(golden.converged).toBe(true);
    expect(brent.x).toBeCloseTo(1.3, 6);
    expect(golden.x).toBeCloseTo(1.3, 6);
    expect(brent.fevals).toBeLessThan(golden.fevals / 2);
  });

  it('respects maxIter and reports converged = false when it runs out', () => {
    const result = minimizeScalar(Math.cos, { bracket: [2, 3, 4], method: 'golden', maxIter: 3 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(3);
  });

  it('falls back to safeguarded steps when f goes non-finite inside the bracket', () => {
    // Finite at the bracket triple, NaN in a pocket left of the minimum.
    const f = (x) => (x > 2.6 && x < 2.9 ? NaN : (x - 3) ** 2);
    const result = minimizeScalar(f, { bracket: [2.5, 3.5, 4.5] });
    expect(Number.isFinite(result.fx)).toBe(true);
    expect(result.x).toBeCloseTo(3, 4);
  });
});

describe('rootScalar', () => {
  it('finds the root of cos(x) at pi/2 in [1, 2] to 1e-12', () => {
    const result = rootScalar(Math.cos, { bracket: [1, 2] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(Math.PI / 2, 11);
    expect(Math.abs(result.fx)).toBeLessThan(1e-10);
  });

  it('solves the classic Brent test x^3 - 2x - 5 = 0 in [2, 3]', () => {
    const result = rootScalar((x) => x ** 3 - 2 * x - 5, { bracket: [2, 3] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(2.0945514815423265, 12);
  });

  it('short-circuits when an endpoint is exactly zero', () => {
    const f = (x) => x * (x - 2);
    const atA = rootScalar(f, { bracket: [0, 1] });
    expect(atA.x).toBe(0);
    expect(Math.abs(atA.fx)).toBe(0);
    expect(atA.iterations).toBe(0);
    expect(atA.converged).toBe(true);

    const atB = rootScalar(f, { bracket: [1, 2] });
    expect(atB.x).toBe(2);
    expect(atB.fx).toBe(0);
    expect(atB.iterations).toBe(0);
    expect(atB.converged).toBe(true);
  });

  it('throws when the bracket has no sign change, mentioning the bracket values', () => {
    expect(() => rootScalar((x) => x * x + 1, { bracket: [-1, 1] }))
      .toThrow(/sign change/i);
    try {
      rootScalar((x) => x * x + 1, { bracket: [-1, 1] });
      expect.unreachable();
    } catch (err) {
      expect(err.message).toContain('-1');
      expect(err.message).toContain('1');
      expect(err.message).toMatch(/sign/i);
    }
  });

  it('requires a bracket', () => {
    expect(() => rootScalar(Math.cos, {})).toThrow(/bracket/i);
    expect(() => rootScalar(Math.cos, { bracket: [1] })).toThrow(/bracket/i);
  });

  it('bisect agrees with brent', () => {
    const brent = rootScalar(Math.cos, { bracket: [1, 2], method: 'brent' });
    const bisect = rootScalar(Math.cos, { bracket: [1, 2], method: 'bisect' });
    expect(brent.converged).toBe(true);
    expect(bisect.converged).toBe(true);
    expect(Math.abs(brent.x - bisect.x)).toBeLessThan(1e-10);
    expect(bisect.x).toBeCloseTo(Math.PI / 2, 10);
  });

  it('handles a non-Lipschitz sign * sqrt function', () => {
    const f = (x) => Math.sign(x - 1.5) * Math.sqrt(Math.abs(x - 1.5));
    const result = rootScalar(f, { bracket: [0, 2] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(1.5, 9);
  });

  it('handles log(x) - 1 in [0.1, 10] despite the NaN region of log', () => {
    const result = rootScalar((x) => Math.log(x) - 1, { bracket: [0.1, 10] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(Math.E, 10);
  });

  it('throws when f is non-finite at a bracket endpoint', () => {
    expect(() => rootScalar((x) => Math.log(x) - 1, { bracket: [-1, 10] }))
      .toThrow(/not finite/i);
  });

  it('falls back to bisection when f goes non-finite during brent iteration', () => {
    // NaN pocket strictly inside the bracket, away from the root at x = 1.
    const f = (x) => (x > 1.7 && x < 1.9 ? NaN : x - 1);
    const result = rootScalar(f, { bracket: [0.5, 2.5] });
    expect(result.converged).toBe(true);
    expect(result.x).toBeCloseTo(1, 10);
  });

  it('reports iterations and fevals, and brent needs fewer fevals than bisect', () => {
    const f = (x) => x ** 3 - 2 * x - 5;
    const brent = rootScalar(f, { bracket: [2, 3], method: 'brent' });
    const bisect = rootScalar(f, { bracket: [2, 3], method: 'bisect' });
    expect(brent.iterations).toBeGreaterThan(0);
    expect(brent.fevals).toBeGreaterThan(0);
    expect(bisect.x).toBeCloseTo(brent.x, 10);
    expect(brent.fevals).toBeLessThan(bisect.fevals);
  });

  it('respects maxIter and reports converged = false when it runs out', () => {
    const result = rootScalar(Math.cos, { bracket: [1, 2], method: 'bisect', maxIter: 3 });
    expect(result.converged).toBe(false);
    expect(result.iterations).toBe(3);
  });

  it('throws on an unknown method', () => {
    expect(() => rootScalar(Math.cos, { bracket: [1, 2], method: 'newton' }))
      .toThrow(/unknown method/i);
  });
});
