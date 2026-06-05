"""
PKE delay tests: basic
======================

Trivial-case limits (the circulating model should degenerate to static
PKE in obvious limits) and qualitative directional checks (longer loop
time depletes more precursors, etc.).
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import (BETA, LAMBDA, LAM, N0,
                      DELAY_ANALYTICAL_TOL, MAX_DELAY_FLOW,
                      circulating_C_ss, circulating_rho_crit,
                      build_circulating_one_group_system, solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.delay, pytest.mark.basic]


class TestTrivialLimits:
    """flow=True with degenerate parameters reproduces the static case."""

    def test_flow_false_identical_to_omitting_flow_kwargs(self):
        """Two equivalent constructions of the no-flow case must agree."""
        # System A: explicit flow=False
        sysA = System()
        nA, CA = Node(name='n', y0=N0), Node(name='C', y0=BETA*N0/(LAMBDA*LAM))
        rhoA = Node(name='rho', y0=0.0)
        sysA.add_nodes([nA, CA, rhoA])
        nA.set_dndt(r=rhoA.y(), beta_eff=BETA, Lambda=LAMBDA,
                    lam=[LAM], C=[CA.y()])
        CA.set_dcdt(n=nA.y(), beta=BETA, Lambda=LAMBDA, lam=LAM, flow=False)

        # System B: flow kwarg omitted entirely
        sysB = System()
        nB, CB = Node(name='n', y0=N0), Node(name='C', y0=BETA*N0/(LAMBDA*LAM))
        rhoB = Node(name='rho', y0=0.0)
        sysB.add_nodes([nB, CB, rhoB])
        nB.set_dndt(r=rhoB.y(), beta_eff=BETA, Lambda=LAMBDA,
                    lam=[LAM], C=[CB.y()])
        CB.set_dcdt(n=nB.y(), beta=BETA, Lambda=LAMBDA, lam=LAM)

        T = np.linspace(0.0, 5.0, 51)
        solve_pke(sysA, T, max_delay=1e-3)
        solve_pke(sysB, T, max_delay=1e-3)
        np.testing.assert_allclose(nA.y_out, nB.y_out, rtol=1e-8)
        np.testing.assert_allclose(CA.y_out, CB.y_out, rtol=1e-8)

    def test_short_loop_limit_recovers_static_C_ss(self):
        """As t_l → 0, circulating C_ss → β·n0/(Λ·λ)."""
        C_static = BETA * N0 / (LAMBDA * LAM)
        C_tiny_tl = circulating_C_ss(N0, BETA, LAMBDA, LAM,
                                     t_c=10.0, t_l=1e-6)
        assert abs(C_tiny_tl - C_static) / C_static < 1e-5

    def test_no_loss_limit_recovers_static_C_ss(self):
        """As t_c → ∞, the C/t_c terms vanish and C_ss → static C_ss."""
        C_static = BETA * N0 / (LAMBDA * LAM)
        C_huge_tc = circulating_C_ss(N0, BETA, LAMBDA, LAM,
                                     t_c=1e9, t_l=20.0)
        assert abs(C_huge_tc - C_static) / C_static < 1e-7


class TestCirculatingSteadyState:
    """At circulating critical rho and circulating C_ss, n and C hold constant."""

    @pytest.mark.parametrize("t_c,t_l", [
        (8.0, 17.0),
        (5.0, 10.0),
        (20.0, 30.0),
    ])
    def test_circulating_critical_steady_state_holds(self, t_c, t_l):
        """Initialize at circulating critical steady state."""
        rho_crit = circulating_rho_crit(BETA, LAM, t_c, t_l)

        sys, n, C, _ = build_circulating_one_group_system(
            rho0=rho_crit, t_c=t_c, t_l=t_l
        )

        C_ss = C.y0
        T = np.linspace(0.0, 50.0, 251)

        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)

        n_err = np.max(np.abs(n.y_out - N0)) / N0
        C_err = np.max(np.abs(C.y_out - C_ss)) / C_ss

        assert n_err < DELAY_ANALYTICAL_TOL
        assert C_err < DELAY_ANALYTICAL_TOL


    def test_circulating_C_ss_with_zero_rho_is_subcritical(self):
        """At circulating C_ss but rho=0, neutron density decays."""
        sys, n, C, _ = build_circulating_one_group_system(
            rho0=0.0, t_c=8.0, t_l=17.0
        )

        T = np.linspace(0.0, 20.0, 201)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)

        assert n.y_out[-1] < N0    


class TestQualitativeFlowEffects:
    """Directional checks on the closed-form C_ss across parameter sweeps."""

    def test_longer_loop_time_reduces_C_ss(self):
        """t_l ↑  ⇒  more precursor decay in the loop  ⇒  C_ss ↓."""
        C_short = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c=10.0, t_l=5.0)
        C_long  = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c=10.0, t_l=50.0)
        assert C_long < C_short

    def test_shorter_core_residence_reduces_C_ss(self):
        """t_c ↓  ⇒  faster flushing  ⇒  C_ss ↓."""
        C_long_tc  = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c=50.0, t_l=20.0)
        C_short_tc = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c=5.0,  t_l=20.0)
        assert C_short_tc < C_long_tc

    def test_circulating_C_ss_below_static_C_ss(self):
        """For any finite t_l, t_c > 0, circulating C_ss < static C_ss."""
        C_static = BETA * N0 / (LAMBDA * LAM)
        for t_c, t_l in [(8.0, 17.0), (10.0, 20.0), (5.0, 30.0)]:
            C_circ = circulating_C_ss(N0, BETA, LAMBDA, LAM, t_c, t_l)
            assert C_circ < C_static
