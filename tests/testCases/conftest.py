"""
Shared fixtures, constants, and closed-form helpers for the msrDynamics
test suite. Covers both PKE and thermal-hydraulics domains in one file.

Auto-discovered by pytest as the conftest at the rootdir of the test tree.
For the test files in ``pke/`` and ``th/`` to import from this file via
``from conftest import X``, ``pytest.ini`` adds ``testCases/`` to
``pythonpath`` (requires pytest ≥ 7.0).

Importable from sibling test files via:

    # PKE
    from conftest import BETA, LAMBDA, build_one_group_system, ...
    # TH
    from conftest import (ANALYTICAL_TOL, build_delayed_advective_chain,
                          cstr_chain_step_response_with_delay, solve_th, ...)

The two domains share tolerance constants but otherwise occupy disjoint
namespaces (different helper names, different builder names, separate
solver wrappers ``solve_pke`` and ``solve_th``).
"""
import numpy as np
import pytest
from jitcdde import t as jitcdde_t
from msrDynamics import Node, System


# ===========================================================================
# Shared tolerances (both domains)
# ===========================================================================

ABS_TOL_TIGHT, REL_TOL_TIGHT = 1e-12, 1e-10
ABS_TOL_LOOSE, REL_TOL_LOOSE = 1e-10, 1e-5

ANALYTICAL_TOL       = 1e-4   # tight: closed-form, no delay
DELAY_ANALYTICAL_TOL = 1e-3   # looser: closed-form, with delay

# max_delay regimes for System.solve() — apply to either domain
MAX_DELAY_STATIC = 1e-3       # forces adaptive integrate() after 1st sample
MAX_DELAY_FLOW   = 1e10       # always integrate_blindly (default for DDEs)


# ===========================================================================
# PKE constants
# ===========================================================================

# Point-kinetics parameters (one-group reduction)
BETA   = 0.0065      # delayed-neutron fraction
LAMBDA = 4e-4        # prompt-neutron generation time (s)
LAM    = 0.1         # one-group avg precursor decay constant (1/s)
N0     = 1.0         # nominal fractional neutron concentration


# ===========================================================================
# PKE closed-form helpers — static
# ===========================================================================

def one_group_pke_closed_form(t, n0, C0, rho, beta, Lambda, lam):
    """Analytical solution of the one-group PKE for constant rho.

    Solves the linear 2x2 system

        d[n,C]/dt = A·[n,C],   A = [[(rho-beta)/Lambda,  lam], [beta/Lambda, -lam]]

    via eigendecomposition. Returns (n(t), C(t)).
    """
    a = (rho - beta) / Lambda
    b = lam
    c = beta / Lambda
    d = -lam
    trace = a + d
    delta = np.sqrt((a - d) ** 2 + 4.0 * b * c)
    mu1 = 0.5 * (trace + delta)
    mu2 = 0.5 * (trace - delta)
    n_dot0 = a * n0 + b * C0
    A1 = (n_dot0 - mu2 * n0) / (mu1 - mu2)
    A2 = (mu1 * n0 - n_dot0) / (mu1 - mu2)
    n = A1 * np.exp(mu1 * t) + A2 * np.exp(mu2 * t)
    C = (A1 * (mu1 - a) * np.exp(mu1 * t)
         + A2 * (mu2 - a) * np.exp(mu2 * t)) / b
    return n, C


def precursor_closed_form(t, C0, n_const, beta, Lambda, lam):
    """Isolated precursor equation with constant neutron source.

        dC/dt = beta·n_const/Lambda − lam·C
        ⇒ C(t) = C_ss + (C0 − C_ss)·exp(−lam·t),   C_ss = beta·n_const/(Lambda·lam).
    """
    C_ss = beta * n_const / (Lambda * lam)
    return C_ss + (C0 - C_ss) * np.exp(-lam * t)


def reactivity_ramp_closed_form(t, rho0, sources, coeffs):
    """Reactivity linear ramp from constant feedback rate.

        d(rho)/dt = Σ_i coeff_i · source_i
        ⇒ rho(t) = rho0 + t · Σ_i coeff_i · source_i.
    """
    rate = sum(c * s for c, s in zip(coeffs, sources))
    return rho0 + rate * t


def dndt_decay_closed_form(t, nd0, n_const, n0, rel_yield, lam):
    """Generic source-decay equation:

        dn_d/dt = (n/n0)·rel_yield − lam·n_d
        ⇒ if source > 0:  n_d(t) = n_d_eq + (nd0 − n_d_eq)·exp(−lam·t),
                          n_d_eq = rel_yield·(n_const/n0) / lam.
             if source = 0:  n_d(t) = nd0·exp(−lam·t).
    """
    source = rel_yield * (n_const / n0)
    if source == 0:
        return nd0 * np.exp(-lam * t)
    nd_eq = source / lam
    return nd_eq + (nd0 - nd_eq) * np.exp(-lam * t)


# ===========================================================================
# PKE closed-form helpers — circulating fuel
# ===========================================================================

def circulating_C_ss(n0, beta, Lambda, lam, t_c, t_l):
    """SS precursor concentration for circulating-fuel one-group PKE.

    At SS with constant n=n0 and C(t-t_l)=C_ss, the governing equation

        dC/dt = β·n0/Λ − λ·C − C/t_c + C(t−t_l)·exp(−λ·t_l)/t_c

    reduces to

        C_ss = β·n0 / (Λ·[λ + (1 − exp(−λ·t_l))/t_c]).
    """
    return beta * n0 / (Lambda * (lam + (1.0 - np.exp(-lam * t_l)) / t_c))


def effective_beta_factor(lam, t_c, t_l):
    """Reduction factor on the delayed-neutron contribution from circulation.

        ζ = λ·C_ss·Λ / (β·n0) = λ·t_c / [λ·t_c + 1 − exp(−λ·t_l)]

    Bounded above by 1 (no circulation) and approaches 0 as t_l → ∞.
    """
    return lam * t_c / (lam * t_c + 1.0 - np.exp(-lam * t_l))


def circulating_rho_crit(beta, lam, t_c, t_l):
    """Critical reactivity for one-group circulating-fuel PKE."""
    return beta * (1.0 - effective_beta_factor(lam, t_c, t_l))


def circulating_rho_crit_multi(betas, lams, t_c, t_l):
    """Critical reactivity for multi-group circulating-fuel PKE."""
    return sum(
        beta * (1.0 - effective_beta_factor(lam, t_c, t_l))
        for beta, lam in zip(betas, lams))


# ===========================================================================
# PKE builders
# ===========================================================================

def build_one_group_system(rho0, n0=N0, C0=None):
    """Build a static (non-circulating) one-group PKE system.

    If ``C0`` is None, precursors are initialized at the rho=0 equilibrium.
    """
    if C0 is None:
        C0 = BETA * n0 / (LAMBDA * LAM)
    sys = System()
    n   = Node(name='n',   y0=n0)
    C   = Node(name='C',   y0=C0)
    rho = Node(name='rho', y0=rho0)
    sys.add_nodes([n, C, rho])
    n.set_dndt(r=rho.y(), beta_eff=BETA, Lambda=LAMBDA,
               lam=[LAM], C=[C.y()])
    C.set_dcdt(n=n.y(), beta=BETA, Lambda=LAMBDA, lam=LAM)
    return sys, n, C, rho


def build_circulating_one_group_system(rho0, t_c, t_l, n0=N0, C0=None,
                                       force_steady_state=False,
                                       max_delay=MAX_DELAY_FLOW):
    """Build a circulating-fuel one-group PKE system.

    If ``C0`` is None, precursors are initialized at the circulating
    steady state given by ``circulating_C_ss``.
    """
    if C0 is None:
        C0 = circulating_C_ss(n0, BETA, LAMBDA, LAM, t_c, t_l)
    sys = System()
    n   = Node(name='n',   y0=n0)
    C   = Node(name='C',   y0=C0)
    rho = Node(name='rho', y0=rho0)
    sys.add_nodes([n, C, rho])
    n.set_dndt(r=rho.y(), beta_eff=BETA, Lambda=LAMBDA,
               lam=[LAM], C=[C.y()])
    C.set_dcdt(n=n.y(), beta=BETA, Lambda=LAMBDA, lam=LAM,
               flow=True, t_c=t_c, t_l=t_l,
               force_steady_state=force_steady_state,
               max_delay=max_delay)
    return sys, n, C, rho


# ===========================================================================
# PKE solver wrapper
# ===========================================================================

def solve_pke(sys, T, max_delay=MAX_DELAY_STATIC, tight=True):
    """Drive sys.solve with a consistent tolerance pair.

    Default ``max_delay`` is ``MAX_DELAY_STATIC = 1e-3``, which forces
    adaptive ``integrate()`` immediately — the right choice for static
    PKE (no real delay). Pass ``max_delay=MAX_DELAY_FLOW`` for circulating
    delay tests where ``integrate_blindly`` is the appropriate path.
    """
    abs_tol = ABS_TOL_TIGHT if tight else ABS_TOL_LOOSE
    rel_tol = REL_TOL_TIGHT if tight else REL_TOL_LOOSE
    sys.solve(T, max_delay=max_delay, populate_nodes=True,
              abs_tol=abs_tol, rel_tol=rel_tol)


# ===========================================================================
# TH closed-form helpers — pipe-delay
# ===========================================================================

def cstr_upstream_step_response(t, T0, T_in, alpha):
    """T1(t) for an upstream CSTR being heated toward T_in by undelayed
    advection from a constant reservoir:

        dT1/dt = α·(T_in − T1),   α = W/m,   T1(0) = T0
        ⇒  T1(t) = T_in + (T0 − T_in)·exp(−α·t).
    """
    return T_in + (T0 - T_in) * np.exp(-alpha * t)


def cstr_chain_step_response_with_delay(t, T0, T_in, alpha, tau):
    """T2(t) for a two-node chain with a pipe delay between upstream and
    downstream advection. With constant_past initialization at T0:

        dT2/dt = α·(T1(t−τ) − T2),
        T2(t < τ) = T0,
        T2(t ≥ τ) = T_in + (T0 − T_in)·exp(−α·(t−τ))·(1 + α·(t−τ)).

    Reduces to the standard two-CSTR-chain solution when τ = 0.
    """
    out = np.full_like(np.asarray(t, dtype=float), float(T0))
    t_arr = np.asarray(t, dtype=float)
    mask = t_arr >= tau
    u = t_arr[mask] - tau
    out[mask] = T_in + (T0 - T_in) * np.exp(-alpha * u) * (1.0 + alpha * u)
    return out


# ===========================================================================
# TH builders — two-node delayed configurations shared across smoke/basic/function
# ===========================================================================

def build_delayed_advective_chain(T0=300.0, T_in=400.0, tau=5.0,
                                   m=1.0, W=0.01, scp=1000.0,
                                   T0_up=None):
    """Two-node advective chain with a pipe delay between upstream and
    downstream:

        up: dT/dt = (T_in    − T_up)·W/m              (undelayed inlet)
        dn: dT/dt = (T_up(t−τ) − T_dn)·W/m            (delayed source)

    Returns (sys, up, dn).

    If ``T0_up`` is omitted, upstream IC = T0 so the constant history of
    the delayed call equals downstream IC, and downstream holds at T0
    during the pre-delay window 0 ≤ t < τ. Override ``T0_up = T_in`` to
    model an upstream held at the inlet from t = 0.
    """
    if T0_up is None:
        T0_up = T0
    sys = System()
    up = Node(m=m, scp=scp, W=W, y0=T0_up, name='up')
    dn = Node(m=m, scp=scp, W=W, y0=T0,    name='down')
    sys.add_nodes([up, dn])
    up.set_dTdt_advective(source=T_in)
    dn.set_dTdt_advective(source=up.y(tau=jitcdde_t - tau))
    return sys, up, dn


def build_undelayed_advective_chain(T0=300.0, T_in=400.0,
                                     m=1.0, W=0.01, scp=1000.0):
    """Same advective chain but with ``up.y()`` as the downstream source
    (no delay machinery). For τ → 0 cross-checks against the delayed
    builder configured with ``tau=0``."""
    sys = System()
    up = Node(m=m, scp=scp, W=W, y0=T0, name='up')
    dn = Node(m=m, scp=scp, W=W, y0=T0, name='down')
    sys.add_nodes([up, dn])
    up.set_dTdt_advective(source=T_in)
    dn.set_dTdt_advective(source=up.y())
    return sys, up, dn


def build_delayed_convective_chain(T0=300.0, T_in=400.0, tau=5.0,
                                    m=1.0, scp=1000.0, W=0.01, hA=10.0):
    """Upstream ramps to T_in via undelayed advection; downstream is
    *convectively* coupled to the delayed upstream temperature.
    Verifies the y(t−τ) plumbing through ``set_dTdt_convective`` rather
    than ``set_dTdt_advective``."""
    sys = System()
    up = Node(m=m, scp=scp, W=W, y0=T0, name='up')
    dn = Node(m=m, scp=scp,      y0=T0, name='down')
    sys.add_nodes([up, dn])
    up.set_dTdt_advective(source=T_in)
    dn.set_dTdt_convective(source=[up.y(tau=jitcdde_t - tau)], hA=[hA])
    return sys, up, dn


# ===========================================================================
# TH solver wrapper
# ===========================================================================

def solve_th(sys, T, max_delay=MAX_DELAY_STATIC, tight=True):
    """Drive sys.solve with a consistent tolerance pair.

    Default ``max_delay = MAX_DELAY_STATIC = 1e-3`` forces adaptive
    ``integrate()`` after the first sample, matching ``solve_pke``.
    This is the right choice for non-delay TH tests — using the
    ``System.solve`` default of ``1e10`` keeps the solver in
    ``integrate_blindly`` for the entire window and causes memory
    blow-up on long integrations (e.g. the 200,000 s convective runs).

    Pass ``max_delay >= τ_max`` for delay tests.
    """
    abs_tol = ABS_TOL_TIGHT if tight else ABS_TOL_LOOSE
    rel_tol = REL_TOL_TIGHT if tight else REL_TOL_LOOSE
    sys.solve(T, max_delay=max_delay, populate_nodes=True,
              abs_tol=abs_tol, rel_tol=rel_tol)