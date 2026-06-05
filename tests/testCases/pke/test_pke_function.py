"""
PKE tests: function
===================

Function-level closed-form comparisons and parameter sweeps for the
PKE setters: one-group PKE vs the analytical eigendecomposition,
prompt-jump amplitude, isolated precursor build-up/decay, reactivity
linear ramps, and the generic source-decay equation in ``set_dndt_decay``.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import (BETA, LAMBDA, LAM, N0, ANALYTICAL_TOL,
                          one_group_pke_closed_form,
                          precursor_closed_form,
                          reactivity_ramp_closed_form,
                          dndt_decay_closed_form,
                          build_one_group_system, solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.function]


class TestOneGroupClosedForm:
    """Tests 7-9: msrDynamics vs analytical solution across inhour spectrum."""

    @pytest.mark.parametrize("rho0", [0.0, 0.003, -0.003])
    def test_one_group_pke_matches_closed_form(self, rho0):
        n0 = N0
        C0 = BETA * n0 / (LAMBDA * LAM)
        sys, n, C, rho = build_one_group_system(rho0=rho0, n0=n0)
        T = np.linspace(0.0, 20.0, 1001)
        solve_pke(sys, T)
        n_cf, C_cf = one_group_pke_closed_form(
            T, n0, C0, rho0, BETA, LAMBDA, LAM
        )
        n_err = np.max(np.abs(n.y_out - n_cf) / np.abs(n_cf))
        C_err = np.max(np.abs(C.y_out - C_cf) / np.abs(C_cf))
        assert n_err < ANALYTICAL_TOL
        assert C_err < ANALYTICAL_TOL


class TestPromptJump:
    """Test 11: textbook prompt-jump amplitude after fast eigenmode decays."""

    def test_prompt_jump_ratio_matches_approximation(self):
        rho0 = 0.003
        sys, n, *_ = build_one_group_system(rho0=rho0)
        T = np.linspace(0.0, 2.0, 401)
        solve_pke(sys, T)

        # Sample after prompt mode decays (~4 tau_p) but before slow growth.
        t_sample = 0.5
        idx = np.argmin(np.abs(T - t_sample))
        ratio_measured  = n.y_out[idx] / N0
        ratio_predicted = BETA / (BETA - rho0)
        rel_err = abs(ratio_measured - ratio_predicted) / ratio_predicted
        # Approximation carries ~2% inherent error
        assert rel_err < 5e-2


class TestIsolatedPrecursor:
    """Tests 13-14: set_dcdt with constant neutron source."""

    def test_precursor_buildup_from_zero(self):
        n_const, C0 = 1.0, 0.0
        sys = System()
        C = Node(name='C', y0=C0)
        sys.add_nodes([C])
        C.set_dcdt(n=n_const, beta=BETA, Lambda=LAMBDA, lam=LAM)
        T = np.linspace(0.0, 50.0, 501)
        solve_pke(sys, T)
        C_cf = precursor_closed_form(T, C0, n_const, BETA, LAMBDA, LAM)
        C_ss = BETA * n_const / (LAMBDA * LAM)
        err = np.max(np.abs(C.y_out - C_cf)) / C_ss
        assert err < ANALYTICAL_TOL

    def test_precursor_decay_with_zero_source(self):
        n_const, C0 = 0.0, 162.5
        sys = System()
        C = Node(name='C', y0=C0)
        sys.add_nodes([C])
        C.set_dcdt(n=n_const, beta=BETA, Lambda=LAMBDA, lam=LAM)
        T = np.linspace(0.0, 50.0, 501)
        solve_pke(sys, T)
        C_cf = precursor_closed_form(T, C0, n_const, BETA, LAMBDA, LAM)
        err = np.max(np.abs(C.y_out - C_cf)) / C0
        assert err < ANALYTICAL_TOL


class TestSetdrdt:
    """Tests 15-17: linear reactivity ramps."""

    def test_single_source_linear_ramp(self):
        rho0, source, coeff = 0.0, 1.0, 1e-4
        sys = System()
        rho = Node(name='rho', y0=rho0)
        sys.add_nodes([rho])
        rho.set_drdt(sources=[source], coeffs=[coeff])
        T = np.linspace(0.0, 100.0, 501)
        solve_pke(sys, T)
        rho_cf = reactivity_ramp_closed_form(T, rho0, [source], [coeff])
        err = np.max(np.abs(rho.y_out - rho_cf)) / max(abs(rho_cf[-1]), 1e-12)
        assert err < ANALYTICAL_TOL

    def test_multiple_sources_sum(self):
        rho0    = 0.0
        sources = [1.0,   2.0,   0.5]
        coeffs  = [1e-4, -3e-5,  4e-5]
        sys = System()
        rho = Node(name='rho', y0=rho0)
        sys.add_nodes([rho])
        rho.set_drdt(sources=sources, coeffs=coeffs)
        T = np.linspace(0.0, 100.0, 501)
        solve_pke(sys, T)
        rho_cf = reactivity_ramp_closed_form(T, rho0, sources, coeffs)
        err = np.max(np.abs(rho.y_out - rho_cf)) / max(abs(rho_cf[-1]), 1e-12)
        assert err < ANALYTICAL_TOL

    def test_opposite_sources_cancel(self):
        rho0 = 5e-3
        s, a = 2.0, 1e-3
        sources = [s, s]
        coeffs  = [a, -a]
        sys = System()
        rho = Node(name='rho', y0=rho0)
        sys.add_nodes([rho])
        rho.set_drdt(sources=sources, coeffs=coeffs)
        T = np.linspace(0.0, 100.0, 501)
        solve_pke(sys, T)
        rho_cf = reactivity_ramp_closed_form(T, rho0, sources, coeffs)
        err = np.max(np.abs(rho.y_out - rho_cf)) / abs(rho0)
        assert err < ANALYTICAL_TOL


class TestSetdndtDecay:
    """Tests 18-19: generic source-decay equation
    dn_d/dt = (n/n0)*rel_yield - lam*n_d
    """

    def test_dndt_decay_buildup_from_zero(self):
        n0, rel_yield, lam = 1.0, 0.01, 0.1
        nd0, n_const = 0.0, 1.0
        sys = System()
        nd = Node(name='n_d', y0=nd0)
        sys.add_nodes([nd])
        nd.set_dndt_decay(n=n_const, n0=n0, rel_yield=rel_yield, lam=lam)
        T = np.linspace(0.0, 50.0, 501)
        solve_pke(sys, T)
        nd_cf = dndt_decay_closed_form(T, nd0, n_const, n0, rel_yield, lam)
        nd_eq = rel_yield / lam
        err = np.max(np.abs(nd.y_out - nd_cf)) / nd_eq
        assert err < ANALYTICAL_TOL

    def test_dndt_decay_pure_decay(self):
        n0, rel_yield, lam = 1.0, 0.01, 0.1
        nd0, n_const = 1.0, 0.0
        sys = System()
        nd = Node(name='n_d', y0=nd0)
        sys.add_nodes([nd])
        nd.set_dndt_decay(n=n_const, n0=n0, rel_yield=rel_yield, lam=lam)
        T = np.linspace(0.0, 50.0, 501)
        solve_pke(sys, T)
        nd_cf = dndt_decay_closed_form(T, nd0, n_const, n0, rel_yield, lam)
        err = np.max(np.abs(nd.y_out - nd_cf)) / nd0
        assert err < ANALYTICAL_TOL
