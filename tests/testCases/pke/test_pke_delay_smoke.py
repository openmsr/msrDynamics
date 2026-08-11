"""
PKE delay tests: smoke
======================

API guards and minimal-solve sanity for ``set_dcdt(flow=True, ...)``.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import (BETA, LAMBDA, LAM, N0, MAX_DELAY_FLOW,
                      build_circulating_one_group_system, solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.delay, pytest.mark.smoke]


class TestSetDcdtFlowAPI:
    """API-level guards on ``set_dcdt`` with flow kwargs."""

    def test_set_dcdt_flow_before_add_raises(self):
        """Pre-system guard must trip even when flow kwargs are supplied."""
        c = Node(name='c', y0=1.0)
        with pytest.raises(ValueError):
            c.set_dcdt(n=None, beta=BETA, Lambda=LAMBDA, lam=LAM,
                       flow=True, t_c=10.0, t_l=20.0)

    def test_flow_kwargs_ignored_when_flow_false(self):
        """flow=False with non-zero t_c/t_l must not raise or alter behavior."""
        sys = System()
        c = Node(name='c', y0=50.0)
        sys.add_nodes([c])
        # passing flow=False with t_c/t_l kwargs — should be ignored
        c.set_dcdt(n=1.0, beta=BETA, Lambda=LAMBDA, lam=LAM,
                   flow=False, t_c=10.0, t_l=20.0)
        T = np.linspace(0.0, 5.0, 11)
        solve_pke(sys, T, max_delay=1e-3, tight=False)
        assert np.all(np.isfinite(c.y_out))


class TestSolveSmoke:
    """Solver completes cleanly with circulating-fuel dynamics enabled."""

    def test_solve_with_flow_returns_finite(self):
        sys, n, C, rho = build_circulating_one_group_system(
            rho0=0.0, t_c=10.0, t_l=20.0
        )
        T = np.linspace(0.0, 5.0, 21)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW, tight=False)
        for node in (n, C, rho):
            assert np.all(np.isfinite(node.y_out))
            assert not np.any(np.isnan(node.y_out))

    def test_force_steady_state_solves(self):
        """force_steady_state=True replaces y(t-t_l) with y(t)·exp(−λ·t_l)."""
        sys, n, C, _ = build_circulating_one_group_system(
            rho0=0.0, t_c=10.0, t_l=20.0, force_steady_state=True
        )
        T = np.linspace(0.0, 5.0, 21)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW, tight=False)
        assert np.all(np.isfinite(n.y_out))
        assert np.all(np.isfinite(C.y_out))


class TestOutputShape:
    """Output shape and initial-value preservation under flow=True."""

    def test_output_length_matches_grid(self):
        sys, n, *_ = build_circulating_one_group_system(
            rho0=0.0, t_c=10.0, t_l=20.0
        )
        T = np.linspace(0.0, 5.0, 137)  # odd prime
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW, tight=False)
        assert len(n.y_out) == 137

    def test_initial_value_preserved_with_flow(self):
        """First output sample equals y0 even with delayed dynamics."""
        n0, C0_arb = N0, 50.0
        sys, n, C, _ = build_circulating_one_group_system(
            rho0=0.0, t_c=10.0, t_l=20.0, n0=n0, C0=C0_arb
        )
        T = np.linspace(0.0, 5.0, 21)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW, tight=False)
        assert n.y_out[0] == pytest.approx(n0, abs=1e-10)
        assert C.y_out[0] == pytest.approx(C0_arb, abs=1e-10)
