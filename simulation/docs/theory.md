# Theory

This note derives the forward model, the least-squares waveform misfit, and the
exact discrete adjoint-state gradient used in `ringfwi/fwi.py`.

## 1. Forward model

The constant-density acoustic wave equation is written in squared-slowness form

    m(x) d2p/dt2 = laplacian(p) + f(x, t),      m(x) = 1 / c(x)^2,

where `p` is pressure, `c` is sound speed, and `f` is the source. The operator
`m d2/dt2 - laplacian` is self-adjoint in space, which keeps the discrete
adjoint stencil identical to the forward one.

Discretising with a second-order leapfrog scheme in time and a five-point
Laplacian `L` in space (spacing `h`, step `dt`, and `S = dt^2 / m`) gives

    p^n = 2 p^{n-1} - p^{n-2} + S ( L p^{n-1} + f^{n-1} ),    n = 1 .. nt-1,

with `p^0 = 0` at rest. Sources are injected into the forcing term at their grid
points; receivers sample `p^n` at their grid points, `d_syn^n = R p^n`.

Stability follows the Courant condition `c_max dt / h <= 1 / sqrt(2)` in 2D.

## 2. Misfit

For observed data `d^n` the least-squares waveform misfit is

    J = 1/2  sum_n  || R p^n - d^n ||^2  =  1/2 sum_n || r^n ||^2,

summed over all sources and receivers, with residual `r^n = R p^n - d^n`.

## 3. Adjoint-state gradient

Introduce an adjoint field `lam^n` as the Lagrange multiplier of the discrete
constraint

    c^n = p^n - 2 p^{n-1} + p^{n-2} - S ( L p^{n-1} + f^{n-1} ) = 0.

Stationarity of the Lagrangian `J + sum_n lam^n . c^n` with respect to `p^k`
gives the time-reversed adjoint recursion

    lam^k = 2 lam^{k+1} - lam^{k+2} + L ( S lam^{k+1} ) - R^T r^k,

run backward for `k = nt-1 .. 1` with `lam = 0` beyond the final step. The
residual `R^T r^k` acts as an adjoint source injected at the receivers.

The gradient of `J` with respect to the model follows from
`dc^n / dS_i = -(L p^{n-1} + f^{n-1})_i`. Converting from `S` to `m` through
`S = dt^2 / m` and using the update relation
`L p^{n-1} + f^{n-1} = (p^n - 2 p^{n-1} + p^{n-2}) / S` collapses the constants
and yields the compact imaging condition

    g_i = (1 / m_i)  sum_n  lam^n_i ( p^n - 2 p^{n-1} + p^{n-2} )_i.

This is the zero-lag correlation of the adjoint field with the second time
difference of the forward field, weighted by `1 / m`.

## 4. Why the discrete adjoint

Deriving the adjoint from the *discrete* forward scheme (rather than
discretising the continuous adjoint equation) makes the gradient the exact
transpose of the forward operator. The practical consequence is that the
adjoint-state gradient matches a central finite-difference gradient to
machine precision, which `tests/test_gradient.py` checks. A gradient that is
only approximately consistent with the forward model slows or stalls the
optimisation, so this exactness matters in practice, not just in principle.

## 5. Optimisation

Each iteration:

1. Compute `J` and `g` from one forward and one adjoint solve per source.
2. Precondition `g` by masking to the update region and Gaussian smoothing to
   suppress the source and receiver imprint.
3. Take a backtracking line search along `-g` so the misfit never increases.
4. Apply physical bounds on `c` within the update region.

Steepest descent with a line search is enough for the single-inclusion
demonstration. Quasi-Newton updates (l-BFGS) and multiscale frequency
continuation are the standard next steps for stronger contrasts and are listed
on the roadmap.

## References

- J. Virieux, *Geophysics* 51(4), 889-901, 1986.
- A. Tarantola, *Geophysics* 49(8), 1259-1266, 1984.
- R.-E. Plessix, *Geophysical Journal International* 167(2), 495-503, 2006.
