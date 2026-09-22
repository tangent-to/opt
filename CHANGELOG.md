# Changelog

Notable changes to `@tangent.to/opt`. This file starts at 0.2.0; for earlier
releases see the [git history](https://github.com/tangent-to/opt/commits/main)
and the [release tags](https://github.com/tangent-to/opt/releases).

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Update rules as steps.** `gradientStep`, `momentumStep`, `rmspropStep`
  and `adamStep`, each `state = step(x, gradient, state, options)`: moves `x`
  in place (a plain or typed array), creates its state on the first call,
  reads `learningRate` and the rule's constants on every call. For a loop
  that owns its own sampling, a mini-batch training loop being the case. The
  drivers `gradientDescent`, `momentumDescent`, `rmsprop` and `adam` are now
  built from them and follow the same trajectories as before.
- Objectives may return `{ value, gradient }`, the form `@tangent.to/grad`'s
  `valueAndGrad` and `compile` return, alongside `{ loss, gradient }`.
