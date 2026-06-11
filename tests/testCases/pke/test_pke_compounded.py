"""
PKE tests: compounded
=====================

Multi-mechanism, multi-node PKE configurations: two-group critical steady
state exercising list-summation in ``set_dndt`` across precursor groups.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import LAMBDA, N0, ANALYTICAL_TOL, solve_pke

pytestmark = [pytest.mark.pke, pytest.mark.compounded]


class TestTwoGroup:
    """list-summation in set_dndt at multi-group steady state."""

    def test_two_group_critical_steady_state(self):
        Lam = LAMBDA
        beta1, lam1 = 0.0020, 0.05
        beta2, lam2 = 0.0045, 0.50
        beta_total  = beta1 + beta2
        n0 = N0
        C1_0 = beta1 * n0 / (Lam * lam1)
        C2_0 = beta2 * n0 / (Lam * lam2)

        sys = System()
        n   = Node(name='n',   y0=n0)
        C1  = Node(name='C1',  y0=C1_0)
        C2  = Node(name='C2',  y0=C2_0)
        rho = Node(name='rho', y0=0.0)
        sys.add_nodes([n, C1, C2, rho])
        n.set_dndt(r=rho.y(), beta_eff=beta_total, Lambda=Lam,
                   lam=[lam1, lam2], C=[C1.y(), C2.y()])
        C1.set_dcdt(n=n.y(), beta=beta1, Lambda=Lam, lam=lam1)
        C2.set_dcdt(n=n.y(), beta=beta2, Lambda=Lam, lam=lam2)

        T = np.linspace(0.0, 20.0, 401)
        solve_pke(sys, T)

        assert np.max(np.abs(n.y_out  - n0))   / n0   < ANALYTICAL_TOL
        assert np.max(np.abs(C1.y_out - C1_0)) / C1_0 < ANALYTICAL_TOL
        assert np.max(np.abs(C2.y_out - C2_0)) / C2_0 < ANALYTICAL_TOL
