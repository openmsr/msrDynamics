"""
TH delay tests: basic
=====================

Trivial-case and qualitative checks for delayed-source TH dynamics:
zero delay must reproduce the undelayed chain, a constant upstream
source must converge regardless of delay, and during the pre-delay
window the downstream node must hold at its IC (because the delayed
upstream call returns the constant past).
"""
import numpy as np
import pytest
from conftest import (ANALYTICAL_TOL,
                      build_delayed_advective_chain,
                      build_undelayed_advective_chain,
                      solve_th)

pytestmark = [pytest.mark.th, pytest.mark.delay, pytest.mark.basic]


class TestZeroDelay:
    """τ = 0 must reproduce the undelayed two-CSTR chain to high accuracy."""

    def test_zero_delay_matches_undelayed_chain(self):
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        t_res = m / W

        # System A: undelayed builder (uses up.y() directly)
        sysA, _, dnA = build_undelayed_advective_chain(
            T0=T0, T_in=T_in, m=m, W=W
        )
        # System B: delayed builder with τ = 0
        sysB, _, dnB = build_delayed_advective_chain(
            T0=T0, T_in=T_in, tau=0.0, m=m, W=W
        )

        T = np.linspace(0.0, 5 * t_res, 501)
        solve_th(sysA, T, max_delay=1e-3, tight=False)
        solve_th(sysB, T, max_delay=10.0, tight=False)

        # Same trajectory within numerical noise of two different
        # integrate_blindly vs integrate() schedules.
        np.testing.assert_allclose(dnA.y_out, dnB.y_out, rtol=5e-3, atol=5e-2)


class TestConstantSourceWithDelay:
    """If the *delayed* source is held constant, the delay does not
    affect the steady state — only the trajectory."""

    def test_constant_upstream_downstream_settles_to_inlet(self):
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0

        # Upstream born at T_in and held there (advective source = T_in,
        # IC = T_in ⇒ derivative is identically zero), so the delayed call
        # always returns T_in for any τ.
        sys, up, dn = build_delayed_advective_chain(
            T0=T0, T_in=T_in, tau=5.0, m=m, W=W, T0_up=T_in
        )

        T = np.linspace(0.0, 10 * (m / W), 501)
        solve_th(sys, T, max_delay=10.0)

        np.testing.assert_allclose(up.y_out, T_in, rtol=ANALYTICAL_TOL)
        assert dn.y_out[-1] == pytest.approx(T_in, rel=1e-3)


class TestHistoryWindow:
    """Before t = τ the delayed call y(t−τ) returns the constant past
    (= upstream.y0). With downstream.y0 = upstream.y0, the downstream
    rate during 0 ≤ t < τ is (y0 − y0)·W/m = 0 → downstream holds at y0."""

    def test_downstream_uses_constant_past_in_pre_delay_window(self):
        T0, T_in, tau = 300.0, 400.0, 10.0
        sys, _, dn = build_delayed_advective_chain(
            T0=T0, T_in=T_in, tau=tau
        )

        T = np.linspace(0.0, 200.0, 401)
        solve_th(sys, T, max_delay=tau + 1.0, tight=False)

        # Sample with a small interior buffer to keep clear of numerical
        # boundary effects right at t = τ.
        mask_pre = T < (tau - 0.5)
        assert np.max(np.abs(dn.y_out[mask_pre] - T0)) < 1e-2
