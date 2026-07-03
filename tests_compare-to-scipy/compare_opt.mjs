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
import { minimize, numericalGradient } from '../src/index.js';
import * as functions from '../tests/functions.js';

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
