# Time-dependent PPDONet

This repo keeps the original PPDONet code inside `PPDONET/`. The new project is
about turning the steady-state surrogate into something that can do
time-dependent disk evolution.

The main outside inspiration is this PINN disk paper:
[Neural Networks as Surrogate Solvers for Time-Dependent Accretion Disk
Dynamics](https://arxiv.org/html/2509.20447). The useful part for us is not
"let a huge neural net solve the whole movie at once". The useful part is more
practical: split time into short windows, hard-code the initial condition in
each window, use periodic coordinates for theta, rescale outputs, and train the
physics loss locally.

I think the operator version should be a few-step evolution map.

Original PPDONet is basically

```text
G_ss: mu = (alpha, h0, q) -> x_ss(r, theta),
```

where `x` is one of `log_sigma`, `v_r`, or `v_theta`. For the time-dependent
case, instead of only asking for

```text
G: (mu, t) -> x(r, theta, t),
```

we can learn a local propagator

```text
Phi_dt: (mu, x_n) -> x_{n+1}.
```

Then a long simulation is just a few learned operator steps:

```text
x_K = Phi_dtK o ... o Phi_dt2 o Phi_dt1 (mu, x_0).
```

This is close to their time-marching PINN idea, but the object we learn is
reusable. We do not want to train a separate PINN for every time window if the
goal is a fast surrogate.

A first ansatz can be

```text
x_hat(mu, x_n, r, theta, tau)
  = x_n(r, theta) + A(tau) * N_theta(mu, Enc(x_n), r, sin(theta), cos(theta), tau),
```

where `tau in [0, 1]` is local time inside the window. Choose `A(0)=0`, for
example `A(tau)=tau` or `1-exp(-c tau)`. Then the initial condition is exact:

```text
x_hat(mu, x_n, r, theta, 0) = x_n(r, theta).
```

That is probably worth doing from day one. It removes one annoying failure mode:
the model cannot drift at the left edge of every time window.

The loss can start simple:

```text
L = L_data + lambda_pde L_hydro + lambda_cont L_cont + lambda_ss L_steady.
```

`L_data` is just matching FARGO snapshots. `L_hydro` is the PINN-style residual
loss if we decide to use the disk equations. `L_cont` keeps adjacent windows
from disagreeing. `L_steady` is the PPDONet-specific extra idea: at late time,
softly anchor the rollout toward the old steady-state network,

```text
x_hat(mu, T) approx G_ss(mu).
```

This anchor may be biased, but it gives the time-dependent model a sane
long-time target. It also makes the old baseline useful instead of only being
background reading.

For implementation, I would keep version 1 small: encode `x_n` with a few sensor
values or a downsampled grid, keep the coordinate branch as
`(r, sin(theta), cos(theta), tau)`, and train a one-step/few-step operator. If
that works on coarse snapshots, then add the PDE residual and loss-balancing
tricks from the PINN paper.

## Mini demo

I added a small smoke demo in `scripts/time_conditioned_operator_demo.py`.

It does not use real time-dependent FARGO data yet. Instead, it asks a controlled
question: if the target field changes over local time `tau`, does putting `tau`
into the operator coordinate branch help? The demo builds a synthetic transient
from an analytic initial disk profile to the bundled steady PPDONet `log_sigma`
prediction, then compares two tiny operator nets:

```text
with time:    N(mu, r, sin(theta), cos(theta), tau)
without time: N(mu, r, sin(theta), cos(theta))
```

Both models use the hard initial-condition form

```text
x_hat = x_0 + tau * residual
```

so `tau=0` is exact.

Run:

```bash
python3.10 -m venv .venv
.venv/bin/pip install -r requirements-demo.txt
.venv/bin/python scripts/time_conditioned_operator_demo.py
```

One smoke result on a `12 x 24` grid:

```text
held-out time RMSE:
  with tau    0.0327
  without tau 0.0691
```

So yes, time as a coordinate variable is doing something reasonable: it roughly
halves interpolation error across unseen times in this toy setup. The less nice
part is held-out parameter generalization, which is still weak here. That points
to the next real issue: time-conditioning helps temporal interpolation, but the
branch encoding of `mu` and `x_n` needs more work before this becomes a serious
operator surrogate.
