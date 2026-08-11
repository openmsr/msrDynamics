"""
PKE delay tests: compounded
===========================

Multi-mechanism, multi-node circulating-PKE configurations:
  * fully-coupled n/C/rho one-group circulating PKE,
  * six-group MSRE-style circulating PKE,
  * circulating vs static differentiation under a step in rho.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import (BETA, LAMBDA, LAM, N0,
                      DELAY_ANALYTICAL_TOL, MAX_DELAY_FLOW,
                      circulating_C_ss,
                      circulating_rho_crit,
                      circulating_rho_crit_multi,
                      build_one_group_system,
                      build_circulating_one_group_system,
                      solve_pke)

pytestmark = [pytest.mark.pke, pytest.mark.delay, pytest.mark.compounded]


class TestCoupledOneGroupCirculating:
    @pytest.mark.parametrize("t_c,t_l", [
        (8.0,  17.0),
        (10.0, 20.0),
    ])
    def test_critical_steady_state_circulating(self, t_c, t_l):
        """At circulating critical rho and C_ss, n and C hold."""
        rho_crit = circulating_rho_crit(BETA, LAM, t_c, t_l)

        sys, n, C, _ = build_circulating_one_group_system(
            rho0=rho_crit, t_c=t_c, t_l=t_l
        )

        C_ss = C.y0
        T = np.linspace(0.0, 100.0, 501)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)

        assert np.max(np.abs(n.y_out - N0)) / N0 < DELAY_ANALYTICAL_TOL
        assert np.max(np.abs(C.y_out - C_ss)) / C_ss < DELAY_ANALYTICAL_TOL


class TestSixGroupCirculatingMSRE:
    """Six-group circulating PKE with MSRE-style parameters."""

    # MSRE-style delayed-group data (approximate; for unit-test use only)
    BETAS = [0.000223, 0.001457, 0.001307, 0.002628, 0.000766, 0.000280]
    LAMS  = [0.0124,   0.0305,   0.111,    0.301,    1.14,     3.01]

    def test_six_group_circulating_critical_steady_state(self):
        """Each group held at circulating C_ss with circulating critical rho."""
        Lam = LAMBDA
        beta_total = sum(self.BETAS)
        t_c, t_l = 8.0, 17.0
        n0 = N0

        Cs_ss = [
            circulating_C_ss(n0, b, Lam, l, t_c, t_l)
            for b, l in zip(self.BETAS, self.LAMS)
        ]

        rho_crit = circulating_rho_crit_multi(
            betas=self.BETAS,
            lams=self.LAMS,
            t_c=t_c,
            t_l=t_l,
        )

        sys = System()
        n = Node(name='n', y0=n0)
        Cs = [Node(name=f'C{i+1}', y0=Cs_ss[i]) for i in range(6)]
        rho = Node(name='rho', y0=rho_crit)

        sys.add_nodes([n] + Cs + [rho])

        n.set_dndt(
            r=rho.y(),
            beta_eff=beta_total,
            Lambda=Lam,
            lam=self.LAMS,
            C=[c.y() for c in Cs],
        )

        for i, c in enumerate(Cs):
            c.set_dcdt(
                n=n.y(),
                beta=self.BETAS[i],
                Lambda=Lam,
                lam=self.LAMS[i],
                flow=True,
                t_c=t_c,
                t_l=t_l,
                max_delay=MAX_DELAY_FLOW,
            )

        T = np.linspace(0.0, 100.0, 501)
        solve_pke(sys, T, max_delay=MAX_DELAY_FLOW)

        assert np.max(np.abs(n.y_out - n0)) / n0 < DELAY_ANALYTICAL_TOL

        for c, c_ss in zip(Cs, Cs_ss):
            assert np.max(np.abs(c.y_out - c_ss)) / c_ss < DELAY_ANALYTICAL_TOL


class TestCirculatingVsStatic:
    def test_circulating_response_below_static_at_t_horizon(self):
        """For same absolute rho, circulating fuel is less reactive than static."""
        rho0 = 0.001
        t_c, t_l = 8.0, 17.0

        sysS, nS, *_ = build_one_group_system(rho0=rho0)

        sysC, nC, *_ = build_circulating_one_group_system(
            rho0=rho0, t_c=t_c, t_l=t_l
        )

        T = np.linspace(0.0, 5.0, 101)

        solve_pke(sysS, T, max_delay=1e-3, tight=False)
        solve_pke(sysC, T, max_delay=MAX_DELAY_FLOW, tight=False)

        assert nS.y_out[-1] > nC.y_out[-1]
