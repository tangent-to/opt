/**
 * Update rules, one step at a time.
 *
 * The drivers in gradient.js run a whole minimization: they own the loop,
 * evaluate the objective, test convergence. A training loop that samples its
 * own mini-batches cannot use a driver, because the driver would be calling
 * the objective; it needs the rule alone. So the rules live here, and the
 * drivers are built from them, which keeps one copy of each.
 *
 * Every step has the same shape:
 *
 *   state = step(x, gradient, state, options)
 *
 * It moves `x` in place, keeps its moments in `state` (created on the first
 * call when `state` is null, and sized to `x`), and reads its options on
 * every call, so a learning-rate schedule is a caller changing one number
 * between two steps. `x` and `gradient` may be plain arrays or typed arrays,
 * of the same length; a caller with several parameter tensors keeps one
 * state per tensor.
 */

/** @typedef {{ learningRate?: number }} StepOptions */

/**
 * Plain gradient descent: `x -= lr · g`. Stateless; returns `null` so it has
 * the signature of the others.
 *
 * @param {number[]|Float64Array} x - moved in place
 * @param {number[]|Float64Array} gradient
 * @param {null} [_state]
 * @param {StepOptions} [options]
 * @returns {null}
 */
export function gradientStep(x, gradient, _state = null, options = {}) {
  const lr = options.learningRate ?? 0.01;
  for (let i = 0; i < x.length; i++) x[i] -= lr * gradient[i];
  return null;
}

/**
 * Gradient descent with momentum: `v = μ v + lr · g; x -= v`.
 *
 * @param {number[]|Float64Array} x - moved in place
 * @param {number[]|Float64Array} gradient
 * @param {{ velocity: Float64Array }|null} [state]
 * @param {StepOptions & { momentum?: number }} [options]
 * @returns {{ velocity: Float64Array }}
 */
export function momentumStep(x, gradient, state = null, options = {}) {
  const lr = options.learningRate ?? 0.01;
  const mu = options.momentum ?? 0.9;
  const s = state ?? { velocity: new Float64Array(x.length) };
  const v = s.velocity;
  for (let i = 0; i < x.length; i++) {
    v[i] = mu * v[i] + lr * gradient[i];
    x[i] -= v[i];
  }
  return s;
}

/**
 * RMSProp: a running mean of squared gradients scales each coordinate's step.
 *
 * @param {number[]|Float64Array} x - moved in place
 * @param {number[]|Float64Array} gradient
 * @param {{ cache: Float64Array }|null} [state]
 * @param {StepOptions & { decay?: number, epsilon?: number }} [options]
 * @returns {{ cache: Float64Array }}
 */
export function rmspropStep(x, gradient, state = null, options = {}) {
  const lr = options.learningRate ?? 0.01;
  const decay = options.decay ?? 0.9;
  const eps = options.epsilon ?? 1e-8;
  const s = state ?? { cache: new Float64Array(x.length) };
  const c = s.cache;
  for (let i = 0; i < x.length; i++) {
    const g = gradient[i];
    c[i] = decay * c[i] + (1 - decay) * g * g;
    x[i] -= lr * g / (Math.sqrt(c[i]) + eps);
  }
  return s;
}

/**
 * Adam (Kingma and Ba, 2015): bias-corrected first and second moments. The
 * step count lives in the state, so bias correction is right however many
 * tensors share one options object.
 *
 * @param {number[]|Float64Array} x - moved in place
 * @param {number[]|Float64Array} gradient
 * @param {{ m: Float64Array, v: Float64Array, t: number }|null} [state]
 * @param {StepOptions & { beta1?: number, beta2?: number, epsilon?: number }} [options]
 * @returns {{ m: Float64Array, v: Float64Array, t: number }}
 */
export function adamStep(x, gradient, state = null, options = {}) {
  const lr = options.learningRate ?? 0.01;
  const b1 = options.beta1 ?? 0.9;
  const b2 = options.beta2 ?? 0.999;
  const eps = options.epsilon ?? 1e-8;
  const s = state ?? { m: new Float64Array(x.length), v: new Float64Array(x.length), t: 0 };
  const { m, v } = s;
  s.t += 1;
  const c1 = 1 - Math.pow(b1, s.t);
  const c2 = 1 - Math.pow(b2, s.t);
  for (let i = 0; i < x.length; i++) {
    const g = gradient[i];
    m[i] = b1 * m[i] + (1 - b1) * g;
    v[i] = b2 * v[i] + (1 - b2) * g * g;
    x[i] -= lr * (m[i] / c1) / (Math.sqrt(v[i] / c2) + eps);
  }
  return s;
}
