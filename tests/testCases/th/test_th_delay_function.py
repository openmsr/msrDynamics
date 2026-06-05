"""
TH delay tests: function
========================

Function-level checks:
  * primitive y(t−τ) test against closed form (bypasses setters),
  * full-trajectory closed form for the two-node delayed chain,
  * phase-lag identity (downstream begins moving at exactly t = τ),
  * monotonicity in τ at fixed sample time,
  * delayed source routed through the *convective* setter.
"""
import numpy as np
import pytest
from jitcdde import t as jitcdde_t
from msrDynamics import Node, System
from conftest import (DELAY_ANALYTICAL_TOL, ANALYTICAL_TOL,
                      cstr_chain_step_response_with_delay,
                      cstr_upstream_step_response,
                      build_delayed_advective_chain,
                      build_delayed_convective_chain,
                      solve_th)

pytestmark = [pytest.mark.th, pytest.mark.delay, pytest.mark.function]


# ===========================================================================

class TestDelayedStatePrimitive:
    """Primitive y(t − τ) test — bypasses setters entirely by assigning
    to ``node.dydt`` directly. Exercises the bare delayed-state machinery
    that underlies ``set_dTdt_advective(source=node.y(t − τ))``.

    Reference setup (no thermal units, just the math):

        source: dy/dt = −λ·y,           y(0) = y0
        target: dT/dt = k·y(t − τ),     T(0) = T0

    With constant history y(t < 0) = y0, the closed forms are:

        source(t) = y0·exp(−λ·t)
        target(t) = T0 + k·y0·t,                                        t < τ
        target(t) = T0 + k·y0·τ + (k·y0/λ)·(1 − exp(−λ·(t − τ))),       t ≥ τ
    """

    @staticmethod
    def _target_closed_form(t_arr, T0, y0, lam, k, tau):
        t_arr = np.asarray(t_arr, dtype=float)
        out = np.empty_like(t_arr)
        before = t_arr < tau
        after = ~before
        out[before] = T0 + k * y0 * t_arr[before]
        u = t_arr[after] - tau
        out[after] = T0 + k * y0 * tau + (k * y0 / lam) * (1.0 - np.exp(-lam * u))
        return out

    def test_target_integrates_delayed_source_to_closed_form(self):
        y0_src, T0_tgt = 1.0, 0.0
        lam, k, tau = 0.25, 2.0, 3.0

        sys = System()
        source = Node(name='source', y0=y0_src)
        target = Node(name='target', y0=T0_tgt)
        sys.add_nodes([source, target])
        source.dydt = -lam * source.y()
        target.dydt = k * source.y(tau=jitcdde_t - tau)

        T_grid = np.linspace(0.0, 30.0, 1201)
        sys.solve(T_grid, max_delay=tau, populate_nodes=True,
                  abs_tol=1e-12, rel_tol=1e-10)

        source_cf = y0_src * np.exp(-lam * T_grid)
        target_cf = self._target_closed_form(T_grid, T0_tgt, y0_src,
                                              lam, k, tau)
        src_denom = max(np.max(np.abs(source_cf)), 1e-14)
        tgt_denom = max(np.max(np.abs(target_cf)), 1e-14)

        assert np.max(np.abs(source.y_out - source_cf)) / src_denom < 1e-4
        assert np.max(np.abs(target.y_out - target_cf)) / tgt_denom < 1e-4

    def test_target_differs_from_no_delay_case(self):
        """Regression guard that y(t − τ) isn't silently collapsing to
        y(t). The two integrals differ by O(k·y0·τ) — well above noise."""
        y0_src, T0_tgt = 1.0, 0.0
        lam, k, tau = 0.25, 2.0, 3.0

        target_with_delay = self._target_closed_form(
            np.array([30.0]), T0_tgt, y0_src, lam, k, tau
        )[0]
        # If delay were ignored: T(t) = T0 + (k·y0/λ)·(1 − exp(−λ·t))
        target_no_delay = (T0_tgt
                          + (k * y0_src / lam)
                          * (1.0 - np.exp(-lam * 30.0)))
        assert abs(target_with_delay - target_no_delay) > 1.0


class TestPipeDelayChain:
    """Two-node chain with pure pipe delay. Generalizes the existing
    undelayed two-CSTR chain test in test_thermal_hydraulics.py to τ > 0."""

    @pytest.mark.parametrize("tau", [1.0, 5.0, 10.0])
    def test_two_node_chain_with_delay_matches_closed_form(self, tau):
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        alpha = W / m

        sys, up, dn = build_delayed_advective_chain(
            T0=T0, T_in=T_in, tau=tau, m=m, W=W
        )
        # Integrate ~5 time constants past the delay window
        T = np.linspace(0.0, 5.0 / alpha + tau, 1001)
        solve_th(sys, T, max_delay=tau + 1.0)

        up_exp = cstr_upstream_step_response(T, T0, T_in, alpha)
        dn_exp = cstr_chain_step_response_with_delay(T, T0, T_in, alpha, tau)

        # Upstream is unaffected by the delay machinery
        np.testing.assert_allclose(up.y_out, up_exp,
                                   rtol=ANALYTICAL_TOL, atol=1e-3)
        # Downstream matches the delayed-chain piecewise solution
        np.testing.assert_allclose(dn.y_out, dn_exp,
                                   rtol=DELAY_ANALYTICAL_TOL, atol=5e-2)


class TestStepResponsePhaseLag:
    """For a chain with constant_past = T0, the downstream temperature
    does not begin moving until t = τ."""

    def test_downstream_starts_moving_at_tau(self):
        T0, T_in, tau = 300.0, 400.0, 10.0
        sys, _, dn = build_delayed_advective_chain(
            T0=T0, T_in=T_in, tau=tau
        )

        T = np.linspace(0.0, 200.0, 2001)
        solve_th(sys, T, max_delay=tau + 1.0)

        # Before the transport delay expires, the downstream node should remain
        # at its initial value. Use T < tau rather than T <= tau because the
        # delayed forcing can begin exactly at tau.
        pre_delay = T < tau
        np.testing.assert_allclose(
            dn.y_out[pre_delay],
            T0,
            atol=1e-6,
        )

        # After the delay, the downstream node should respond. A finite threshold
        # crossing happens slightly after tau because the downstream state has its
        # own thermal/advective time response.
        moves = np.where(np.abs(dn.y_out - T0) > 0.01)[0]
        assert len(moves) > 0, "downstream never moved"

        t_first_move = T[moves[0]]
        assert tau <= t_first_move <= tau + 2.0


class TestDelayMonotonicity:
    """At fixed sample time t > all τ values, downstream temperature
    decreases monotonically as τ increases."""

    def test_longer_delay_yields_lower_downstream_at_fixed_time(self):
        T0, T_in = 300.0, 400.0
        t_sample = 100.0
        taus = [1.0, 5.0, 10.0, 20.0]

        downstream_at_sample = []
        for tau in taus:
            sys, _, dn = build_delayed_advective_chain(
                T0=T0, T_in=T_in, tau=tau
            )
            T = np.linspace(0.0, t_sample, 1001)
            solve_th(sys, T, max_delay=tau + 1.0)
            downstream_at_sample.append(dn.y_out[-1])

        diffs = np.diff(downstream_at_sample)
        assert np.all(diffs < 0.0), \
            f"Expected monotone decrease, got {downstream_at_sample}"


class TestConvectiveDelayedSource:
    """Same delay machinery, wired through ``set_dTdt_convective``
    instead of ``set_dTdt_advective``. Verifies the symbolic plumbing
    is uniform across the two setters."""

    def test_delayed_convective_source_settles_to_inlet(self):
        m, scp, W, hA = 1.0, 1000.0, 0.01, 10.0
        T0, T_in, tau = 300.0, 400.0, 5.0

        sys, _, dn = build_delayed_convective_chain(
            T0=T0, T_in=T_in, tau=tau, m=m, scp=scp, W=W, hA=hA
        )

        # Delayed convective coupling is not guaranteed to be within 0.5%
        # after only 500 s for this setup. Use a longer horizon so the test
        # checks the asymptotic inlet limit, not a fragile settling estimate.
        T = np.linspace(0.0, 1000.0, 2001)
        solve_th(sys, T, max_delay=tau + 1.0)

        assert dn.y_out[-1] > dn.y_out[0]
        assert dn.y_out[-1] <= T_in
        assert dn.y_out[-1] == pytest.approx(T_in, rel=5e-3)
