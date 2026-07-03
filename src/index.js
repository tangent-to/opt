/**
 * @tangent.to/opt - Declarative numerical optimization for JavaScript (ESM)
 *
 * Function minimization for the browser, Node, and Deno. Companion package
 * to @tangent.to/ds.
 */

export { methods, minimize } from './minimize.js';
export { nelderMead } from './neldermead.js';
export { adam, backtrackingLineSearch, gradientDescent, momentumDescent, rmsprop } from './gradient.js';
export { numericalGradient } from './numdiff.js';

// tangent/ds compatibility layer (ds.core.optimize drop-in)
export {
  AdamOptimizer,
  createOptimizer,
  GradientDescent,
  MomentumOptimizer,
  RMSProp,
} from './compat.js';

import { methods, minimize } from './minimize.js';
import { nelderMead } from './neldermead.js';
import { adam, gradientDescent, momentumDescent, rmsprop } from './gradient.js';
import { numericalGradient } from './numdiff.js';
import {
  AdamOptimizer,
  createOptimizer,
  GradientDescent,
  MomentumOptimizer,
  RMSProp,
} from './compat.js';

export default {
  minimize,
  methods,
  nelderMead,
  gradientDescent,
  momentumDescent,
  rmsprop,
  adam,
  numericalGradient,
  GradientDescent,
  MomentumOptimizer,
  RMSProp,
  AdamOptimizer,
  createOptimizer,
};
