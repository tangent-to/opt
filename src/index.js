/**
 * @tangent.to/opt - Declarative numerical optimization for JavaScript (ESM)
 *
 * Function minimization, scalar optimization/root-finding and nonlinear
 * least squares for the browser, Node, and Deno. Companion package to
 * @tangent.to/ds.
 */

export { methods, minimize } from './minimize.js';
export { nelderMead } from './neldermead.js';
export { lbfgs } from './lbfgs.js';
export { adam, backtrackingLineSearch, gradientDescent, momentumDescent, rmsprop } from './gradient.js';
export { strongWolfeLineSearch } from './linesearch.js';
export { minimizeScalar, rootScalar } from './scalar.js';
export { curveFit, leastSquares } from './leastsq.js';
export { solve } from './linsolve.js';
export { makeBoundsTransform } from './bounds.js';
export { numericalGradient, numericalHessian, numericalJacobian } from './numdiff.js';

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
import { lbfgs } from './lbfgs.js';
import { adam, gradientDescent, momentumDescent, rmsprop } from './gradient.js';
import { minimizeScalar, rootScalar } from './scalar.js';
import { curveFit, leastSquares } from './leastsq.js';
import { solve } from './linsolve.js';
import { numericalGradient, numericalHessian, numericalJacobian } from './numdiff.js';
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
  lbfgs,
  gradientDescent,
  momentumDescent,
  rmsprop,
  adam,
  minimizeScalar,
  rootScalar,
  leastSquares,
  curveFit,
  solve,
  numericalGradient,
  numericalHessian,
  numericalJacobian,
  GradientDescent,
  MomentumOptimizer,
  RMSProp,
  AdamOptimizer,
  createOptimizer,
};
