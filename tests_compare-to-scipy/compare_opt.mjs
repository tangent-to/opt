#!/usr/bin/env node
/**
 * Helper script to run @tangent.to/opt minimizers for comparison with
 * scipy.optimize. Reads a JSON spec, prints a JSON result.
 *
 * Usage:
 *   node tests_compare-to-scipy/compare_opt.mjs /path/to/spec.json
 *
 * Spec shape:
 *   {
 *     "function": "rosenbrock",     // name from tests/functions.js
 *     "x0": [-1.2, 1],
 *     "method": "neldermead",
 *     "grad": true,                  // pass the analytic gradient if available
 *     "options": { "maxIter": 2000 }
 *   }
 */

import { readFileSync } from 'node:fs';
import { curveFit, minimize, minimizeScalar, numericalGradient, rootScalar } from '../src/index.js';
import * as functions from '../tests/functions.js';

const SCALAR_FUNCS = {
  cos: Math.cos,
  quartic: (x) => (x - 2) ** 4,
  cubic: (x) => x ** 3 - 2 * x - 5,
  logroot: (x) => Math.log(x) - 1,
  shifted_parabola: (x) => (x - 7.3) ** 2 + 1,
};

const MODELS = {
  exp_decay: (x, p) => p[0] * Math.exp(-p[1] * x) + p[2],
  sigmoid: (x, p) => p[0] / (1 + Math.exp(-p[1] * (x - p[2]))),
  linear: (x, p) => p[0] * x + p[1],
};

const GRADS = {
  sphere: functions.sphereGrad,
  rosenbrock: functions.rosenbrockGrad,
};

const specPath = process.argv[2];
if (!specPath) {
  console.error('usage: compare_opt.mjs <spec.json>');
  process.exit(1);
}

const spec = JSON.parse(readFileSync(specPath, 'utf8'));

if (spec.mode === 'minimize_scalar' || spec.mode === 'root_scalar') {
  const fn = SCALAR_FUNCS[spec.function];
  if (!fn) {
    console.error(`unknown scalar function: ${spec.function}`);
    process.exit(1);
  }
  const runner = spec.mode === 'minimize_scalar' ? minimizeScalar : rootScalar;
  const result = runner(fn, spec.options || {});
  process.stdout.write(JSON.stringify({
    x: result.x,
    fx: result.fx,
    iterations: result.iterations,
    fevals: result.fevals,
    converged: result.converged,
  }));
  process.exit(0);
}

if (spec.mode === 'curve_fit') {
  const model = MODELS[spec.model];
  if (!model) {
    console.error(`unknown model: ${spec.model}`);
    process.exit(1);
  }
  const result = curveFit({ model, x: spec.x, y: spec.y, p0: spec.p0, ...(spec.options || {}) });
  process.stdout.write(JSON.stringify({
    params: result.params,
    stdErr: result.stdErr,
    fx: result.fx,
    converged: result.converged,
  }));
  process.exit(0);
}

const f = functions[spec.function];
if (!f) {
  console.error(`unknown test function: ${spec.function}`);
  process.exit(1);
}

if (spec.mode === 'gradient') {
  // Compare numericalGradient against scipy.optimize.approx_fprime
  const g = numericalGradient(f, spec.x0);
  process.stdout.write(JSON.stringify({ gradient: g }));
  process.exit(0);
}

const result = minimize({
  f,
  grad: spec.grad ? GRADS[spec.function] : undefined,
  x0: spec.x0,
  method: spec.method || 'neldermead',
  ...(spec.options || {}),
});

process.stdout.write(JSON.stringify({
  x: result.x,
  fx: result.fx,
  iterations: result.iterations,
  converged: result.converged,
}));
