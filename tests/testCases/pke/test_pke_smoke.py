"""
PKE tests: smoke
================

API guards, construction sanity, and minimal-solve invariants for the
point-kinetics setters (``set_dndt``, ``set_dcdt``, ``set_drdt``,
``set_dndt_decay``). No physics — just confirm the contract holds.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import BETA, LAMBDA, LAM, N0, build_one_group_system, solve_pke

pytestmark = [pytest.mark.pke, pytest.mark.smoke]


class TestSmoke:
    """Tests 1-4: API guards, construction sanity, output preservation."""

    # ---- Test 1: pre-system setter guards ---------------------------------

    def test_set_dndt_before_add_raises(self):
        n = Node(name='n', y0=1.0)
        with pytest.raises(ValueError):
            n.set_dndt(r=None, beta_eff=BETA, Lambda=LAMBDA,
                       lam=[LAM], C=[None])

    def test_set_dcdt_before_add_raises(self):
        c = Node(name='c', y0=1.0)
        with pytest.raises(ValueError):
            c.set_dcdt(n=None, beta=BETA, Lambda=LAMBDA, lam=LAM)

    def test_set_drdt_before_add_raises(self):
        r = Node(name='rho', y0=0.0)
        with pytest.raises(ValueError):
            r.set_drdt(sources=[0.0], coeffs=[1.0])

    def test_set_dndt_decay_before_add_raises(self):
        d = Node(name='n_d', y0=0.0)
        with pytest.raises(ValueError):
            d.set_dndt_decay(n=None, n0=1.0, rel_yield=0.01, lam=0.1)

    # ---- Test 2: cross-domain mixing guards -------------------------------

    def test_dndt_blocks_dTdt_convective(self):
        sys = System()
        n = Node(name='n', y0=1.0)
        sys.add_nodes([n])
        n.set_dndt(r=0.0, beta_eff=BETA, Lambda=LAMBDA, lam=[LAM], C=[0.0])
        with pytest.raises(ValueError):
            n.set_dTdt_convective(source=[0.0], hA=[1.0])

    def test_dTdt_blocks_set_dndt(self):
        sys = System()
        n = Node(name='n', y0=300.0, m=1.0, scp=1.0)
        sys.add_nodes([n])
        n.set_dTdt_convective(source=[400.0], hA=[1.0])
        with pytest.raises(ValueError):
            n.set_dndt(r=0.0, beta_eff=BETA, Lambda=LAMBDA, lam=[LAM], C=[0.0])

    def test_dTdt_blocks_set_dcdt(self):
        sys = System()
        n = Node(name='n', y0=300.0, m=1.0, scp=1.0)
        sys.add_nodes([n])
        n.set_dTdt_convective(source=[400.0], hA=[1.0])
        with pytest.raises(ValueError):
            n.set_dcdt(n=0.0, beta=BETA, Lambda=LAMBDA, lam=LAM)

    def test_dTdt_blocks_set_drdt(self):
        sys = System()
        n = Node(name='n', y0=300.0, m=1.0, scp=1.0)
        sys.add_nodes([n])
        n.set_dTdt_convective(source=[400.0], hA=[1.0])
        with pytest.raises(ValueError):
            n.set_drdt(sources=[0.0], coeffs=[1.0])

    # ---- Test 3: minimal solve sanity -------------------------------------

    def test_minimal_solve_finite(self):
        sys, n, C, rho = build_one_group_system(rho0=0.0, C0=100.0)
        T = np.linspace(0.0, 1.0, 21)
        solve_pke(sys, T, tight=False)
        sol = np.column_stack([n.y_out, C.y_out, rho.y_out])
        assert sol.shape == (len(T), 3)
        assert not np.any(np.isnan(sol))
        assert not np.any(np.isinf(sol))

    # ---- Test 4: output shape & initial-value preservation ----------------

    def test_output_shape_and_initial_values(self):
        n0, C0, rho0, nd0 = 1.0, 50.0, 0.0, 0.0
        sys = System()
        n   = Node(name='n',   y0=n0)
        C   = Node(name='C',   y0=C0)
        rho = Node(name='rho', y0=rho0)
        nd  = Node(name='n_d', y0=nd0)
        sys.add_nodes([n, C, rho, nd])
        n.set_dndt(r=rho.y(), beta_eff=BETA, Lambda=LAMBDA,
                   lam=[LAM], C=[C.y()])
        C.set_dcdt(n=n.y(), beta=BETA, Lambda=LAMBDA, lam=LAM)
        rho.set_drdt(sources=[0.0], coeffs=[1.0])
        nd.set_dndt_decay(n=n.y(), n0=n0, rel_yield=0.01, lam=0.1)

        T = np.linspace(0.0, 1.0, 11)
        solve_pke(sys, T, tight=False)

        for node, y0 in [(n, n0), (C, C0), (rho, rho0), (nd, nd0)]:
            assert node.y_out.size > 0
            assert len(node.y_out) == len(T)
            assert np.isclose(node.y_out[0], y0, atol=1e-10)
