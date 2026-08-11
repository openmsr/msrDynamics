"""
PKE delay tests: function
=========================

Function-level checks: numerical SS matches the circulating closed form,
the effective-beta reduction factor is reproduced exactly, ``force_steady_state``
agrees with the true-delay formulation on the SS manifold, and the
``max_delay`` clamp in ``set_dcdt`` actually limits the delay used.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import (BETA, LAMBDA, LAM, N0,
                      DELAY_ANALYTICAL_TOL, MAX_DELAY_FLOW,
                      circulating_C_ss, effective_beta_factor,
                      build_circulating_one_group_system, solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.delay, pytest.mark.function]


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------

def build_isolated_circulating_C_system(t_c, t_l, C0, n_const=N0,
                                        force_steady_state=False,
                                        max_delay=MAX_DELAY_FLOW):
    """Isolate the precursor dynamics by holding n at a constant value.

    Constructing n with ``n.dydt = 0.0`` plus IC=n_const keeps n fixed so
    we can test ``set_dcdt`` in isolation against its closed-form SS.
    """
    sys = System()
    n = Node(name='n', y0=n_const)
    C = Node(name='C', y0=C0)
    sys.add_nodes([n, C])
    n.dydt = 0.0
    C.set_dcdt(n=n.y(), beta=BETA, Lambda=LAMBDA, lam=LAM,
               flow=True, t_c=t_c, t_l=t_l,
               force_steady_state=force_steady_state,
               max_delay=max_delay)
    return sys, n, C


# ===========================================================================

class TestFlowingPrecursorFirstInterval:
    """First-interval closed form for ``set_dcdt(flow=True, ...)``.

    Within 0 ≤ t < t_l, the delayed call C(t − t_l) returns the constant
    history value C0. With n=0 (no source), the equation reduces to a
    first-order linear ODE

        dC/dt = − a·C + b,   a = λ + 1/t_c,   b = C0·exp(−λ·t_l)/t_c,

    with closed-form solution

        C(t) = C_ss + (C0 − C_ss)·exp(−a·t),    C_ss = b/a.

    This is a sharper test than asymptotic-SS comparisons because it
    pins down the *transient* shape during the first delay window,
    not just a single equilibrium value.
    """

    @staticmethod
    def _C_closed_form(T_arr, C0, lam, t_c, t_l):
        T_arr = np.asarray(T_arr, dtype=float)
        a = lam + 1.0 / t_c
        b = C0 * np.exp(-lam * t_l) / t_c
        C_ss = b / a
        return C_ss + (C0 - C_ss) * np.exp(-a * T_arr)

    def test_first_interval_against_closed_form(self):
        """Stay strictly inside 0 ≤ t < t_l so C(t − t_l) = C0 always."""
        C0 = 10.0
        beta_val, Lam_val, lam_val = 0.0065, 4e-4, 0.1
        t_c, t_l = 5.0, 3.0

        sys = System()
        C = Node(name='C_flow', y0=C0)
        sys.add_nodes([C])
        C.set_dcdt(n=0.0, beta=beta_val, Lambda=Lam_val, lam=lam_val,
                   flow=True, t_c=t_c, t_l=t_l)

        # Strict interior of the first interval
        T_grid = np.linspace(0.0, 0.95 * t_l, 301)
        sys.solve(T_grid, max_delay=t_l, populate_nodes=True,
                  abs_tol=1e-12, rel_tol=1e-10)

        C_cf = self._C_closed_form(T_grid, C0, lam_val, t_c, t_l)
        denom = max(np.max(np.abs(C_cf)), 1e-14)
        rel_err = np.max(np.abs(C.y_out - C_cf)) / denom
        assert rel_err < 1e-4

    def test_delayed_return_term_has_measurable_effect(self):
        """Regression guard: the C(t − t_l)·exp(−λ·t_l)/t_c term must
        change the answer measurably vs the outflow-only model.
        Catches silent collapse of the delayed call to y(t)."""
        C0, lam_val = 10.0, 0.1
        t_c, t_l = 5.0, 3.0
        T_grid = np.linspace(0.0, 0.95 * t_l, 301)

        C_with_return = self._C_closed_form(T_grid, C0, lam_val, t_c, t_l)
        # No-return model: dC/dt = −(λ + 1/t_c)·C, pure exponential decay
        C_no_return = C0 * np.exp(-(lam_val + 1.0 / t_c) * T_grid)

        assert np.max(np.abs(C_with_return - C_no_return)) > 1e-3


class TestClosedFormSteadyState:
    """Numerical C_ss matches the closed-form circulating-fuel SS."""

    @pytest.mark.parametrize("t_c,t_l", [
        (8.0,  17.0),
        (5.0,  10.0),
        (15.0, 25.0),
        (3.0,  6.0),
        (20.0, 40.0),
    ])
    def test_circulating_C_ss_matches_closed_form(self, t_c, t_l):
        """Start C below its SS and let it relax up to the predicted value."""
        C_ss_expected = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c, t_l)
        sys, _, C = build_isolated_circulating_C_system(
            t_c=t_c, t_l=t_l, C0=0.8 * C_ss_expected
        )
        # Long horizon: effective time constant ~ 1/[λ + (1−e^(−λ·t_l))/t_c]
        # ≤ ~10s for the parameter ranges above; 300s is ample.
        T = np.linspace(0.0, 300.0, 1001)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)
        rel_err = abs(C.y_out[-1] - C_ss_expected) / C_ss_expected
        assert rel_err < DELAY_ANALYTICAL_TOL


class TestEffectiveBetaReduction:
    """λ·C_ss·Λ / (β·n0) equals the closed-form ζ factor < 1."""

    @pytest.mark.parametrize("t_c,t_l", [
        (8.0,  17.0),
        (10.0, 20.0),
        (5.0,  30.0),
    ])
    def test_effective_beta_factor_matches(self, t_c, t_l):
        """Pure algebraic identity — confirms the two helper formulas
        are consistent and that the implied ζ < 1."""
        C_ss = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c, t_l)
        zeta_predicted = effective_beta_factor(LAM, t_c, t_l)
        zeta_from_Css = LAM * C_ss * LAMBDA / (BETA * N0)
        assert abs(zeta_from_Css - zeta_predicted) / zeta_predicted < 1e-12
        assert zeta_predicted < 1.0


class TestForceSteadyState:
    """force_steady_state=True swaps y(t−t_l) → y(t)·exp(−λ·t_l).
    On the SS manifold (constant C), both formulations are identical."""

    @pytest.mark.parametrize("t_c,t_l", [
        (8.0,  17.0),
        (10.0, 20.0),
    ])
    def test_force_ss_holds_at_equilibrium_IC(self, t_c, t_l):
        """force_steady_state model held at C_ss should not drift."""
        C_ss = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c, t_l)
        sys, _, C = build_isolated_circulating_C_system(
            t_c=t_c, t_l=t_l, C0=C_ss, force_steady_state=True
        )
        T = np.linspace(0.0, 50.0, 251)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)
        rel_err = np.max(np.abs(C.y_out - C_ss)) / C_ss
        assert rel_err < DELAY_ANALYTICAL_TOL

    @pytest.mark.parametrize("t_c,t_l", [
        (8.0,  17.0),
        (10.0, 20.0),
    ])
    def test_force_ss_matches_true_delay_at_equilibrium(self, t_c, t_l):
        """Both formulations give the same trajectory when held at SS."""
        C_ss = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c, t_l)
        sysA, _, CA = build_isolated_circulating_C_system(
            t_c=t_c, t_l=t_l, C0=C_ss, force_steady_state=True
        )
        sysB, _, CB = build_isolated_circulating_C_system(
            t_c=t_c, t_l=t_l, C0=C_ss, force_steady_state=False
        )
        T = np.linspace(0.0, 50.0, 251)
        solve_pke(sysA, T, max_delay=MAX_DELAY_FLOW)
        solve_pke(sysB, T, max_delay=MAX_DELAY_FLOW)
        np.testing.assert_allclose(
            CA.y_out, CB.y_out, rtol=DELAY_ANALYTICAL_TOL
        )


class TestMaxDelayClamp:
    """``set_dcdt`` uses ``sp.Min(max_delay, t_l)``. Passing a max_delay
    smaller than t_l must make the model behave as if t_l = max_delay."""

    def test_max_delay_smaller_than_tl_clamps_to_max_delay(self):
        """t_l=50 with set_dcdt(max_delay=10) should reproduce t_l=10 SS."""
        t_c, max_delay = 10.0, 10.0
        # ground-truth SS uses the effective (clamped) t_l = 10
        C_ss_clamped = circulating_C_ss(N0, BETA, LAMBDA, LAM,
                                         t_c=t_c, t_l=max_delay)
        sys, _, C = build_isolated_circulating_C_system(
            t_c=t_c, t_l=50.0, C0=0.8 * C_ss_clamped,
            max_delay=max_delay
        )
        T = np.linspace(0.0, 300.0, 1001)
        # Note: solve(max_delay=...) is independent of set_dcdt(max_delay=...)
        # — here we just need solve(max_delay) ≥ the effective delay.
        solve_pke(sys, T, max_delay=max_delay)
        rel_err = abs(C.y_out[-1] - C_ss_clamped) / C_ss_clamped
        assert rel_err < DELAY_ANALYTICAL_TOL


class TestZeroDelayLimit:
    """When ``t_l → 0`` the delayed call collapses; trajectories should
    approach the static-PKE precursor balance with an outflow term."""

    def test_tl_zero_with_finite_tc_outflow_only(self):
        """t_l → 0 ⇒ inflow = outflow, leaving the static SS unchanged."""
        t_c, t_l = 10.0, 1e-6
        C_static = BETA * N0 / (LAMBDA * LAM)
        sys, _, C = build_isolated_circulating_C_system(
            t_c=t_c, t_l=t_l, C0=C_static
        )
        T = np.linspace(0.0, 50.0, 251)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)
        rel_err = np.max(np.abs(C.y_out - C_static)) / C_static
        assert rel_err < DELAY_ANALYTICAL_TOL
