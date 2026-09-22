/**
 * The update rules as steps. Two things are pinned: a driver and a hand-run
 * loop of its step follow the same trajectory to the bit, so there is one
 * copy of each rule; and a step behaves as a training loop needs it to,
 * lazily creating its state, working on typed arrays, and reading its
 * options on every call.
 */

import { describe, expect, it } from 'vitest';
import {
  adam, adamStep, gradientDescent, gradientStep, lbfgs, momentumDescent, momentumStep, rmsprop, rmspropStep,
} from '../src/index.js';

const sphere = (x) => ({ loss: x.reduce((s, v) => s + v * v, 0), gradient: x.map((v) => 2 * v) });
const x0 = [3, -2, 0.5];

/** Run a step by hand for `k` iterations, the way a training loop would. */
function byHand(step, k, options) {
  const x = [...x0];
  let state = null;
  for (let i = 0; i < k; i++) state = step(x, sphere(x).gradient, state, options);
  return x;
}

describe('steps: the drivers are built from them', () => {
  const k = 25;
  const cases = [
    ['gradientDescent', gradientDescent, gradientStep, {}],
    ['momentumDescent', momentumDescent, momentumStep, { momentum: 0.8 }],
    ['rmsprop', rmsprop, rmspropStep, { decay: 0.95 }],
    ['adam', adam, adamStep, { beta1: 0.8, beta2: 0.99 }],
  ];
  for (const [name, driver, step, extra] of cases) {
    it(`${name} and its step follow one trajectory`, () => {
      const lr = 0.05;
      const driven = driver(sphere, x0, { learningRate: lr, maxIter: k, tol: 0, ...extra });
      const manual = byHand(step, k, { learningRate: lr, ...extra });
      expect(driven.x).toEqual(manual);
    });
  }
});

describe('steps: what a training loop needs', () => {
  it('creates its state on the first call, sized to x, and returns it', () => {
    const x = Float64Array.from(x0);
    const s = adamStep(x, [1, 1, 1], null, { learningRate: 0.1 });
    expect(s.m).toHaveLength(3);
    expect(s.v).toHaveLength(3);
    expect(s.t).toBe(1);
    const s2 = adamStep(x, [1, 1, 1], s, { learningRate: 0.1 });
    expect(s2).toBe(s);
    expect(s.t).toBe(2);
    expect(gradientStep(x, [1, 1, 1])).toBeNull();
  });

  it('moves a typed array in place', () => {
    const x = Float64Array.from([1, 2]);
    gradientStep(x, Float64Array.from([1, 1]), null, { learningRate: 0.5 });
    expect(Array.from(x)).toEqual([0.5, 1.5]);
    const y = Float64Array.from([1, 2]);
    momentumStep(y, [1, 1], null, { learningRate: 0.5, momentum: 0.9 });
    expect(Array.from(y)).toEqual([0.5, 1.5]);
  });

  it("Adam's first step has magnitude lr in every coordinate, whatever the gradient scale", () => {
    const x = [0, 0, 0];
    adamStep(x, [1e-3, -5, 1e4], null, { learningRate: 0.1 });
    // epsilon (1e-8) shaves 1e-6 off the smallest coordinate's step; 4 places is the claim.
    x.forEach((v) => expect(Math.abs(v)).toBeCloseTo(0.1, 4));
    expect(Math.sign(x[1])).toBe(1);
  });

  it('reads the learning rate on every call, so a schedule is a number changed between steps', () => {
    const opts = { learningRate: 1 };
    const x = [10];
    gradientStep(x, [1], null, opts);
    opts.learningRate = 0.1;
    gradientStep(x, [1], null, opts);
    expect(x[0]).toBeCloseTo(8.9, 12);
  });

  it('keeps one state per tensor when several share an options object', () => {
    const opts = { learningRate: 0.1 };
    const W = [1, 1], b = [1];
    let sW = null, sb = null;
    for (let i = 0; i < 3; i++) {
      sW = adamStep(W, [1, 1], sW, opts);
      sb = adamStep(b, [1], sb, opts);
    }
    expect(sW.t).toBe(3);
    expect(sb.t).toBe(3);
    expect(W[0]).toBeCloseTo(b[0], 12);   // same gradient history, same path
  });
});

describe('evaluator: the form grad returns', () => {
  it('{value, gradient} is an objective as it stands', () => {
    const f = (x) => ({ value: x[0] * x[0] + x[1] * x[1], gradient: [2 * x[0], 2 * x[1]] });
    const r = lbfgs(f, [3, -2]);
    expect(r.fx).toBeLessThan(1e-10);
    const a = adam(f, [3, -2], { learningRate: 0.1, maxIter: 500 });
    expect(a.fx).toBeLessThan(1e-3);
  });
});
