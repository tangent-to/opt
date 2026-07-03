/**
 * Gradient-based minimizers: gradient descent, momentum, RMSProp, Adam.
 *
 * Moved from tangent/ds (ds.core.optimize). Update rules are unchanged;
 * the API is functional and all methods share the evaluator contract from
 * evaluate.js, so objectives may be (x) => number with an optional separate
 * gradient, or the combined (x) => {loss, gradient} form used in tangent/ds.
 */

import { makeEvaluator } from './evaluate.js';

function gradNorm(gradient) {
  let sum = 0;
  for (let i = 0; i < gradient.length; i++) {
    sum += gradient[i] * gradient[i];
  }
  return Math.sqrt(sum);
}

/**
 * Backtracking line search satisfying the Armijo condition.
 *
 * @param {Function} evaluate - (x) => {loss, gradient}
 * @param {Array<number>} x - Current point
 * @param {Array<number>} gradient - Gradient at x
 * @param {number} currentLoss - Loss at x
 * @returns {number} Step size
 */
export function backtrackingLineSearch(evaluate, x, gradient, currentLoss) {
  const alpha = 0.3; // Armijo condition constant
  const beta = 0.8; // Reduction factor
  let t = 1.0;

  let gradNormSq = 0;
  for (let i = 0; i < gradient.length; i++) {
    gradNormSq += gradient[i] * gradient[i];
  }

  for (let i = 0; i < 20; i++) {
    const xNew = x.map((xi, j) => xi - t * gradient[j]);
    const { loss: newLoss } = evaluate(xNew);
    if (newLoss <= currentLoss - alpha * t * gradNormSq) {
      return t;
    }
    t *= beta;
  }

  return t;
}

/**
 * Shared descent loop.
 *
 * @param {Object} spec
 * @param {Function} spec.init - (n) => state
 * @param {Function} spec.step - ({x, gradient, state, iter, lr}) mutates x in place
 * @param {boolean} [spec.trackLearningRate] - Record per-iteration step sizes
 * @param {Function} f - Objective
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} options
 * @returns {Object} {x, fx, iterations, converged, history}
 */
function descend(spec, f, x0, options = {}) {
  const learningRate = options.learningRate || 0.01;
  const maxIter = options.maxIter || 1000;
  const tol = options.tol || 1e-6;
  const verbose = options.verbose || false;
  const lineSearch = options.lineSearch || false;

  const evaluate = makeEvaluator(f, options.grad, options);

  const x = [...x0];
  const state = spec.init(x.length);

  const history = { loss: [], gradNorm: [] };
  if (spec.trackLearningRate) {
    history.learningRate = [];
  }

  let converged = false;
  let iteration = 0;
  let lastLoss = NaN;

  for (; iteration < maxIter; iteration++) {
    const { loss, gradient } = evaluate(x);
    lastLoss = loss;

    history.loss.push(loss);
    const norm = gradNorm(gradient);
    history.gradNorm.push(norm);

    if (norm < tol) {
      converged = true;
      if (verbose) {
        console.log(`Converged at iteration ${iteration}, loss: ${loss.toFixed(6)}`);
      }
      break;
    }

    let lr = learningRate;
    if (lineSearch && spec.trackLearningRate) {
      lr = backtrackingLineSearch(evaluate, x, gradient, loss);
    }
    if (spec.trackLearningRate) {
      history.learningRate.push(lr);
    }

    spec.step({ x, gradient, state, iter: iteration, lr });

    if (verbose && iteration % 100 === 0) {
      console.log(`Iter ${iteration}: loss=${loss.toFixed(6)}, grad_norm=${norm.toFixed(6)}`);
    }
  }

  const fx = converged ? lastLoss : evaluate(x).loss;

  return { x, fx, iterations: iteration, converged, history };
}

/**
 * Plain gradient descent, with optional backtracking line search.
 *
 * @param {Function} f - Objective: (x) => number or (x) => {loss, gradient}
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options] - {grad, learningRate, maxIter, tol, lineSearch, verbose}
 * @returns {Object} {x, fx, iterations, converged, history}
 */
export function gradientDescent(f, x0, options = {}) {
  return descend(
    {
      init: () => null,
      step: ({ x, gradient, lr }) => {
        for (let i = 0; i < x.length; i++) {
          x[i] -= lr * gradient[i];
        }
      },
      trackLearningRate: true,
    },
    f,
    x0,
    options,
  );
}

/**
 * Gradient descent with momentum.
 *
 * @param {Function} f - Objective
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options] - {grad, learningRate, maxIter, tol, momentum, verbose}
 * @returns {Object} {x, fx, iterations, converged, history}
 */
export function momentumDescent(f, x0, options = {}) {
  const momentum = options.momentum || 0.9;
  return descend(
    {
      init: (n) => ({ velocity: new Array(n).fill(0) }),
      step: ({ x, gradient, state, lr }) => {
        for (let i = 0; i < x.length; i++) {
          state.velocity[i] = momentum * state.velocity[i] + lr * gradient[i];
          x[i] -= state.velocity[i];
        }
      },
    },
    f,
    x0,
    options,
  );
}

/**
 * RMSProp.
 *
 * @param {Function} f - Objective
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options] - {grad, learningRate, maxIter, tol, decay, epsilon, verbose}
 * @returns {Object} {x, fx, iterations, converged, history}
 */
export function rmsprop(f, x0, options = {}) {
  const decay = options.decay || 0.9;
  const epsilon = options.epsilon || 1e-8;
  return descend(
    {
      init: (n) => ({ cache: new Array(n).fill(0) }),
      step: ({ x, gradient, state, lr }) => {
        for (let i = 0; i < x.length; i++) {
          state.cache[i] = decay * state.cache[i] + (1 - decay) * gradient[i] * gradient[i];
          x[i] -= lr * gradient[i] / (Math.sqrt(state.cache[i]) + epsilon);
        }
      },
    },
    f,
    x0,
    options,
  );
}

/**
 * Adam (adaptive moment estimation).
 *
 * @param {Function} f - Objective
 * @param {Array<number>} x0 - Initial parameters
 * @param {Object} [options] - {grad, learningRate, maxIter, tol, beta1, beta2, epsilon, verbose}
 * @returns {Object} {x, fx, iterations, converged, history}
 */
export function adam(f, x0, options = {}) {
  const beta1 = options.beta1 || 0.9;
  const beta2 = options.beta2 || 0.999;
  const epsilon = options.epsilon || 1e-8;
  return descend(
    {
      init: (n) => ({ m: new Array(n).fill(0), v: new Array(n).fill(0) }),
      step: ({ x, gradient, state, iter, lr }) => {
        const { m, v } = state;
        for (let i = 0; i < x.length; i++) {
          m[i] = beta1 * m[i] + (1 - beta1) * gradient[i];
          v[i] = beta2 * v[i] + (1 - beta2) * gradient[i] * gradient[i];
        }
        const t = iter + 1;
        for (let i = 0; i < x.length; i++) {
          const mHat = m[i] / (1 - Math.pow(beta1, t));
          const vHat = v[i] / (1 - Math.pow(beta2, t));
          x[i] -= lr * mHat / (Math.sqrt(vHat) + epsilon);
        }
      },
    },
    f,
    x0,
    options,
  );
}
