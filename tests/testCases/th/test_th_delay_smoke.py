"""
TH delay tests: smoke
=====================

API plumbing for delayed advective/convective sources of the form
``node.y(tau=jitcdde.t - τ)``. No physics — just confirm the solver
runs, returns finite output of the right shape, and preserves y0.
"""
import numpy as np
import pytest
from conftest import (build_delayed_advective_chain,
                      build_delayed_convective_chain,
                      solve_th)

pytestmark = [pytest.mark.th, pytest.mark.delay, pytest.mark.smoke]


class TestDelayedSourceAPI:
    """Sanity: delayed sources are accepted by both setters."""

    def test_delayed_source_accepted_by_advective_setter(self):
        sys, _, dn = build_delayed_advective_chain(
            T0=300.0, T_in=400.0, tau=5.0
        )
        T = np.linspace(0.0, 50.0, 51)
        solve_th(sys, T, max_delay=10.0, tight=False)
        assert np.all(np.isfinite(dn.y_out))

    def test_delayed_source_accepted_by_convective_setter(self):
        sys, _, dn = build_delayed_convective_chain(
            T0=300.0, T_in=400.0, tau=5.0
        )
        T = np.linspace(0.0, 200.0, 51)
        solve_th(sys, T, max_delay=10.0, tight=False)
        assert np.all(np.isfinite(dn.y_out))


class TestSolveSmoke:
    """Basic invariants under a delayed-source chain."""

    def test_solve_with_delay_finite(self):
        sys, _, dn = build_delayed_advective_chain(
            T0=300.0, T_in=400.0, tau=5.0
        )
        T = np.linspace(0.0, 500.0, 51)
        solve_th(sys, T, max_delay=10.0, tight=False)
        assert np.all(np.isfinite(dn.y_out))
        assert not np.any(np.isnan(dn.y_out))

    def test_output_length_matches_grid(self):
        sys, _, dn = build_delayed_advective_chain(
            T0=300.0, T_in=400.0, tau=5.0
        )
        T = np.linspace(0.0, 500.0, 137)  # odd prime
        solve_th(sys, T, max_delay=10.0, tight=False)
        assert len(dn.y_out) == 137

    def test_initial_value_preserved(self):
        """First output sample equals y0 (delayed source must not move it)."""
        sys, up, dn = build_delayed_advective_chain(
            T0=355.0, T_in=400.0, tau=5.0
        )
        T = np.linspace(0.0, 500.0, 51)
        solve_th(sys, T, max_delay=10.0, tight=False)
        assert up.y_out[0] == pytest.approx(355.0, rel=1e-6)
        assert dn.y_out[0] == pytest.approx(355.0, rel=1e-6)
