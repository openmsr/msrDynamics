"""
PKE tests: basic
================

Minimal-mechanism qualitative checks: rho=0 equilibrium holds, off-equilibrium
ICs drive n in the right direction, and the inhour spectrum produces the
expected directional trends.
"""
import numpy as np
import pytest
from conftest import (BETA, LAMBDA, LAM, N0, ANALYTICAL_TOL,
                          build_one_group_system, solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.basic]


class TestCriticalSteadyState:
    """rho=0 equilibrium and off-equilibrium IC behavior."""

    def test_critical_steady_state_one_group(self):
        """n and C stay constant at rho=0 with equilibrium precursors."""
        n0 = N0
        C0 = BETA * n0 / (LAMBDA * LAM)
        sys, n, C, rho = build_one_group_system(rho0=0.0, n0=n0)
        T = np.linspace(0.0, 10.0, 201)
        solve_pke(sys, T)
        n_err = np.max(np.abs(n.y_out - n0)) / n0
        C_err = np.max(np.abs(C.y_out - C0)) / C0
        assert n_err < ANALYTICAL_TOL
        assert C_err < ANALYTICAL_TOL

    @pytest.mark.parametrize("C_factor,direction", [
        (0.5, "below"),
        (2.0, "above"),
    ])
    def test_wrong_precursor_ic_produces_transient(self, C_factor, direction):
        """off-equilibrium C0 drives n with the correct sign."""
        n0 = N0
        C_eq = BETA * n0 / (LAMBDA * LAM)
        sys, n, *_ = build_one_group_system(
            rho0=0.0, n0=n0, C0=C_factor * C_eq
        )
        T = np.linspace(0.0, 5.0, 101)
        solve_pke(sys, T, tight=False)
        if direction == "below":
            assert np.all(n.y_out[1:] < n0)
        else:
            assert np.all(n.y_out[1:] > n0)


class TestTrendSanity:
    """qualitative trends across the inhour spectrum."""

    def test_delayed_supercritical_grows(self):
        sys, n, *_ = build_one_group_system(rho0=0.001)
        T = np.linspace(0.0, 10.0, 201)
        solve_pke(sys, T, tight=False)
        assert n.y_out[-1] > N0

    def test_larger_positive_rho_grows_faster(self):
        sys_a, n_a, *_ = build_one_group_system(rho0=0.001)
        sys_b, n_b, *_ = build_one_group_system(rho0=0.004)
        T = np.linspace(0.0, 10.0, 201)
        solve_pke(sys_a, T, tight=False)
        solve_pke(sys_b, T, tight=False)
        assert n_b.y_out[-1] > n_a.y_out[-1]

    def test_subcritical_decays(self):
        sys, n, *_ = build_one_group_system(rho0=-0.001)
        T = np.linspace(0.0, 10.0, 201)
        solve_pke(sys, T, tight=False)
        assert n.y_out[-1] < N0

    def test_more_negative_rho_decays_faster(self):
        sys_a, n_a, *_ = build_one_group_system(rho0=-0.001)
        sys_b, n_b, *_ = build_one_group_system(rho0=-0.005)
        T = np.linspace(0.0, 10.0, 201)
        solve_pke(sys_a, T, tight=False)
        solve_pke(sys_b, T, tight=False)
        assert n_b.y_out[-1] < n_a.y_out[-1]
