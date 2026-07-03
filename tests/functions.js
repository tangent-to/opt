/**
 * Standard optimization test functions with known minima.
 */

/** Sphere: min 0 at origin */
export function sphere(x) {
  return x.reduce((sum, xi) => sum + xi * xi, 0);
}

export function sphereGrad(x) {
  return x.map((xi) => 2 * xi);
}

/** Rosenbrock: min 0 at (1, 1) */
export function rosenbrock(x) {
  const [a, b] = x;
  return (1 - a) ** 2 + 100 * (b - a * a) ** 2;
}

export function rosenbrockGrad(x) {
  const [a, b] = x;
  return [
    -2 * (1 - a) - 400 * a * (b - a * a),
    200 * (b - a * a),
  ];
}

/** Booth: min 0 at (1, 3) */
export function booth(x) {
  const [a, b] = x;
  return (a + 2 * b - 7) ** 2 + (2 * a + b - 5) ** 2;
}

/** Himmelblau: min 0 at four points, incl. (3, 2) */
export function himmelblau(x) {
  const [a, b] = x;
  return (a * a + b - 11) ** 2 + (a + b * b - 7) ** 2;
}

/** Beale: min 0 at (3, 0.5) */
export function beale(x) {
  const [a, b] = x;
  return (1.5 - a + a * b) ** 2 +
    (2.25 - a + a * b * b) ** 2 +
    (2.625 - a + a * b * b * b) ** 2;
}
