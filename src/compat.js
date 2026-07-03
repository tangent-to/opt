/**
 * tangent/ds compatibility layer.
 *
 * Drop-in equivalents of the classes formerly in ds.core.optimize, so that
 * tangent/ds can re-export this module unchanged:
 *
 *   optimizer.minimize(lossFn, x0, options) => {x, history}
 *
 * where lossFn is the combined form (x) => {loss, gradient}. New code should
 * prefer the declarative minimize() or the functional methods directly.
 */

import { adam, gradientDescent, momentumDescent, rmsprop } from './gradient.js';

class Optimizer {
  constructor(options = {}) {
    this.learningRate = options.learningRate || 0.01;
    this.maxIter = options.maxIter || 1000;
    this.tol = options.tol || 1e-6;
    this.verbose = options.verbose || false;
  }

  /**
   * Minimize a loss function
   * @param {Function} lossFn - Function that returns {loss, gradient}
   * @param {Array<number>} x0 - Initial parameters
   * @param {Object} options - Additional options
   * @returns {Object} {x, history}
   */
  minimize(_lossFn, _x0, _options = {}) {
    throw new Error('minimize() must be implemented by subclass');
  }

  _options(options) {
    return {
      learningRate: this.learningRate,
      maxIter: options.maxIter || this.maxIter,
      tol: options.tol || this.tol,
      verbose: this.verbose,
    };
  }
}

/**
 * Gradient Descent (batch; optional backtracking line search)
 */
export class GradientDescent extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.stochastic = options.stochastic || false;
    this.batchSize = options.batchSize || 32;
    this.lineSearch = options.lineSearch || false;
  }

  minimize(lossFn, x0, options = {}) {
    const { x, history } = gradientDescent(lossFn, x0, {
      ...this._options(options),
      lineSearch: this.lineSearch,
    });
    return { x, history };
  }
}

/**
 * Momentum Optimizer
 */
export class MomentumOptimizer extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.momentum = options.momentum || 0.9;
  }

  minimize(lossFn, x0, options = {}) {
    const { x, history } = momentumDescent(lossFn, x0, {
      ...this._options(options),
      momentum: this.momentum,
    });
    return { x, history: { loss: history.loss, gradNorm: history.gradNorm } };
  }
}

/**
 * RMSProp Optimizer
 */
export class RMSProp extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.decay = options.decay || 0.9;
    this.epsilon = options.epsilon || 1e-8;
  }

  minimize(lossFn, x0, options = {}) {
    const { x, history } = rmsprop(lossFn, x0, {
      ...this._options(options),
      decay: this.decay,
      epsilon: this.epsilon,
    });
    return { x, history: { loss: history.loss, gradNorm: history.gradNorm } };
  }
}

/**
 * Adam Optimizer (Adaptive Moment Estimation)
 */
export class AdamOptimizer extends Optimizer {
  constructor(options = {}) {
    super(options);
    this.beta1 = options.beta1 || 0.9;
    this.beta2 = options.beta2 || 0.999;
    this.epsilon = options.epsilon || 1e-8;
  }

  minimize(lossFn, x0, options = {}) {
    const { x, history } = adam(lossFn, x0, {
      ...this._options(options),
      beta1: this.beta1,
      beta2: this.beta2,
      epsilon: this.epsilon,
    });
    return { x, history: { loss: history.loss, gradNorm: history.gradNorm } };
  }
}

/**
 * Convenience function to create optimizer by name
 * @param {string} name - Optimizer name
 * @param {Object} options - Optimizer options
 * @returns {Optimizer} Optimizer instance
 */
export function createOptimizer(name, options = {}) {
  const optimizers = {
    gd: GradientDescent,
    gradient_descent: GradientDescent,
    sgd: GradientDescent,
    momentum: MomentumOptimizer,
    rmsprop: RMSProp,
    adam: AdamOptimizer,
  };

  const OptimizerClass = optimizers[name.toLowerCase()];
  if (!OptimizerClass) {
    throw new Error(
      `Unknown optimizer: ${name}. Available: ${Object.keys(optimizers).join(', ')}`,
    );
  }

  return new OptimizerClass(options);
}
