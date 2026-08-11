"""
TH tests: compounded
====================

Multi-mechanism, multi-node thermal-hydraulic configurations:
  * mixed setters on one node (advective + convective + internal),
  * node-to-node advective and source coupling,
  * closed convective networks with energy conservation and equilibration checks.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import solve_th

pytestmark = [pytest.mark.th, pytest.mark.compounded]


class TestCoupledDynamics:
    """Multi-node coupling and additive-contribution composition."""

    # ------------------------------------------------------------------
    # Pairwise mixed dynamics on one node
    # ------------------------------------------------------------------

    def test_advective_plus_convective_on_same_node(self):
        """Advection + convection on one node sum to a first-order ODE."""
        m, scp = 1.0, 1000.0
        W, hA = 0.01, 10.0
        T0, T_in, T_c = 300.0, 400.0, 500.0

        node = Node(m=m, scp=scp, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)
        node.set_dTdt_convective(source=[T_c], hA=[hA])

        r_adv = W / m
        r_conv = hA / (m * scp)
        r_eff = r_adv + r_conv
        T_ss = (r_adv * T_in + r_conv * T_c) / r_eff

        t_space = np.linspace(0, 10 / r_eff, 1000)
        solve_th(system, t_space)

        expected = T_ss + (T0 - T_ss) * np.exp(-r_eff * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_internal_plus_convective_on_same_node(self):
        """Internal heat + convection: heated rod with surface cooling."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        s_val, k_val = 1.0, 1000.0
        hA, T_c = 10.0, 300.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[s_val], k=[k_val])
        node.set_dTdt_convective(source=[T_c], hA=[hA])

        t_const = m * scp / hA
        T_ss = T_c + (k_val * s_val) / hA

        t_space = np.linspace(0, 10 * t_const, 1000)
        solve_th(system, t_space)

        expected = T_ss + (T0 - T_ss) * np.exp(-t_space / t_const)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_advective_plus_internal_on_same_node(self):
        """Advection + internal: constant heat source shifts equilibrium above ``T_in``."""
        m, scp = 1.0, 1000.0
        W, T_in = 0.01, 400.0
        s_val, k_val = 1.0, 1000.0
        T0 = 300.0

        node = Node(m=m, scp=scp, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)
        node.set_dTdt_internal(source=[s_val], k=[k_val])

        r_adv = W / m
        r_int = (k_val * s_val) / (m * scp)
        T_ss = T_in + r_int / r_adv
        t_const = m / W

        t_space = np.linspace(0, 10 * t_const, 1000)
        solve_th(system, t_space)

        expected = T_ss + (T0 - T_ss) * np.exp(-r_adv * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_advective_plus_convective_plus_internal_on_same_node(self):
        """All three additive contributions on one node sum correctly."""
        m, scp = 1.0, 1000.0
        W = 0.01
        hA, T_c = 10.0, 200.0
        T_in = 400.0
        s_val, k_val = 1.0, 500.0
        T0 = 350.0

        node = Node(m=m, scp=scp, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)
        node.set_dTdt_convective(source=[T_c], hA=[hA])
        node.set_dTdt_internal(source=[s_val], k=[k_val])

        r_adv = W / m
        r_conv = hA / (m * scp)
        r_int = (k_val * s_val) / (m * scp)
        r_eff = r_adv + r_conv
        T_ss = (r_adv * T_in + r_conv * T_c + r_int) / r_eff

        t_space = np.linspace(0, 10 / r_eff, 1000)
        solve_th(system, t_space)

        expected = T_ss + (T0 - T_ss) * np.exp(-r_eff * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    # ------------------------------------------------------------------
    # Node-to-node advective coupling
    # ------------------------------------------------------------------

    def test_advective_two_node_chain(self):
        """Two CSTRs in series: ``T2`` follows ``T1`` with the proper time lag.

        With ``α = W/m``::
            T1(t) = T_in + (T0 − T_in)·exp(−αt)
            T2(t) = T_in + (T0 − T_in)·exp(−αt)·(1 + αt)
        """
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        t_res = m / W

        node1 = Node(m=m, scp=1000.0, W=W, y0=T0, name="n1")
        node2 = Node(m=m, scp=1000.0, W=W, y0=T0, name="n2")
        system = System()
        system.add_nodes([node1, node2])
        node1.set_dTdt_advective(source=T_in)
        node2.set_dTdt_advective(source=node1.y())

        t_space = np.linspace(0, 5 * t_res, 1000)
        solve_th(system, t_space)

        a = W / m
        T1_ana = T_in + (T0 - T_in) * np.exp(-a * t_space)
        T2_ana = T_in + (T0 - T_in) * np.exp(-a * t_space) * (1 + a * t_space)

        np.testing.assert_allclose(node1.y_out, T1_ana, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(node2.y_out, T2_ana, rtol=1e-4, atol=1e-6)

    def test_internal_source_coupled_to_other_node(self):
        """Internal heat driven by another node's state variable (fission-heat pattern).

        Source node has pure-decay dynamic ``Node.dydt = -λ·node.y()`` so::
            y_B(t) = y_B(0)·exp(-λt)
            T_T(t) = T0 + (k·y_B(0))/(m·scp·λ) · (1 − exp(-λt))
        """
        m_T, scp_T, T0 = 1.0, 1000.0, 300.0
        k_val = 1000.0
        y_B_0 = 1.0
        lam = 0.01

        node_B = Node(m=1.0, scp=1.0, y0=y_B_0, name="n_B")
        node_T = Node(m=m_T, scp=scp_T, y0=T0, name="n_T")
        system = System()
        system.add_nodes([node_B, node_T])

        node_B.dydt = -lam * node_B.y()
        node_T.set_dTdt_internal(source=[node_B.y()], k=[k_val])

        t_space = np.linspace(0, 5 / lam, 1000)
        solve_th(system, t_space)

        y_B_ana = y_B_0 * np.exp(-lam * t_space)
        T_T_ana = T0 + (k_val * y_B_0) / (m_T * scp_T * lam) * (
            1 - np.exp(-lam * t_space)
        )

        np.testing.assert_allclose(node_B.y_out, y_B_ana, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(node_T.y_out, T_T_ana, rtol=1e-4, atol=1e-6)

    # ------------------------------------------------------------------
    # Closed convective networks (2-node)
    # ------------------------------------------------------------------

    def test_two_node_closed_system_conserves_energy(self):
        """Closed two-node convective system conserves total enthalpy."""
        m1, scp1, T1_0 = 1.0, 1000.0, 400.0
        m2, scp2, T2_0 = 2.0, 1000.0, 300.0
        hA = 0.05

        node1 = Node(m=m1, scp=scp1, y0=T1_0, name="hot")
        node2 = Node(m=m2, scp=scp2, y0=T2_0, name="cold")
        system = System()
        system.add_nodes([node1, node2])
        node1.set_dTdt_convective(source=[node2.y()], hA=[hA])
        node2.set_dTdt_convective(source=[node1.y()], hA=[hA])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        E0 = m1 * scp1 * T1_0 + m2 * scp2 * T2_0
        E_final = m1 * scp1 * node1.y_out[-1] + m2 * scp2 * node2.y_out[-1]
        assert E_final == pytest.approx(E0, rel=1e-5)

    def test_two_node_closed_system_reaches_capacity_weighted_average(self):
        """Equilibrium is the heat-capacity-weighted average ``Σ C_i·T_i / Σ C_i``."""
        m1, scp1, T1_0 = 1.0, 1000.0, 400.0
        m2, scp2, T2_0 = 2.0, 1000.0, 300.0
        hA = 0.05
        C1, C2 = m1 * scp1, m2 * scp2
        T_final = (C1 * T1_0 + C2 * T2_0) / (C1 + C2)

        node1 = Node(m=m1, scp=scp1, y0=T1_0, name="hot")
        node2 = Node(m=m2, scp=scp2, y0=T2_0, name="cold")
        system = System()
        system.add_nodes([node1, node2])
        node1.set_dTdt_convective(source=[node2.y()], hA=[hA])
        node2.set_dTdt_convective(source=[node1.y()], hA=[hA])

        t_space = np.linspace(0, 300_000, 1000)
        solve_th(system, t_space)

        assert node1.y_out[-1] == pytest.approx(T_final, rel=1e-3)
        assert node2.y_out[-1] == pytest.approx(T_final, rel=1e-3)

    def test_two_equal_nodes_meet_at_arithmetic_mean(self):
        """Symmetric capacities: capacity-weighted mean reduces to arithmetic mean."""
        m, scp, hA = 1.0, 1000.0, 0.1
        T1_0, T2_0 = 400.0, 300.0
        T_mean = (T1_0 + T2_0) / 2.0

        node1 = Node(m=m, scp=scp, y0=T1_0, name="n1")
        node2 = Node(m=m, scp=scp, y0=T2_0, name="n2")
        system = System()
        system.add_nodes([node1, node2])
        node1.set_dTdt_convective(source=[node2.y()], hA=[hA])
        node2.set_dTdt_convective(source=[node1.y()], hA=[hA])

        t_space = np.linspace(0, 100_000, 1000)
        solve_th(system, t_space)

        assert node1.y_out[-1] == pytest.approx(T_mean, rel=1e-3)
        assert node2.y_out[-1] == pytest.approx(T_mean, rel=1e-3)

    def test_hot_node_cools_and_cold_node_warms(self):
        """Directional sanity: hot node cools, cold node warms."""
        m, scp, hA = 1.0, 1000.0, 0.05
        T1_0, T2_0 = 500.0, 300.0

        node1 = Node(m=m, scp=scp, y0=T1_0, name="hot")
        node2 = Node(m=m, scp=scp, y0=T2_0, name="cold")
        system = System()
        system.add_nodes([node1, node2])
        node1.set_dTdt_convective(source=[node2.y()], hA=[hA])
        node2.set_dTdt_convective(source=[node1.y()], hA=[hA])

        t_space = np.linspace(0, 50_000, 500)
        solve_th(system, t_space)

        assert node1.y_out[-1] < T1_0
        assert node2.y_out[-1] > T2_0

    # ------------------------------------------------------------------
    # Closed convective networks (3-node chain)
    # ------------------------------------------------------------------

    def test_three_node_chain_conserves_energy(self):
        """Chain ``1↔2↔3``: total enthalpy conserved."""
        m, scp = 1.0, 1000.0
        T0s = [500.0, 350.0, 300.0]
        hA = 0.05

        n1 = Node(m=m, scp=scp, y0=T0s[0], name="n1")
        n2 = Node(m=m, scp=scp, y0=T0s[1], name="n2")
        n3 = Node(m=m, scp=scp, y0=T0s[2], name="n3")
        system = System()
        system.add_nodes([n1, n2, n3])
        n1.set_dTdt_convective(source=[n2.y()], hA=[hA])
        n2.set_dTdt_convective(source=[n1.y(), n3.y()], hA=[hA, hA])
        n3.set_dTdt_convective(source=[n2.y()], hA=[hA])

        t_space = np.linspace(0, 500_000, 1000)
        solve_th(system, t_space)

        E0 = m * scp * sum(T0s)
        E_final = m * scp * (n1.y_out[-1] + n2.y_out[-1] + n3.y_out[-1])
        assert E_final == pytest.approx(E0, rel=1e-4)

    def test_three_node_chain_reaches_arithmetic_mean(self):
        """Chain ``1↔2↔3`` with equal capacities equilibrates at the arithmetic mean.

        Pins down where the system settles. If node 2's multi-source list is
        silently truncated, energy is conserved but the equilibrium is wrong.
        """
        m, scp = 1.0, 1000.0
        T0s = [500.0, 350.0, 300.0]
        hA = 0.05
        T_eq = sum(T0s) / 3.0

        n1 = Node(m=m, scp=scp, y0=T0s[0], name="n1")
        n2 = Node(m=m, scp=scp, y0=T0s[1], name="n2")
        n3 = Node(m=m, scp=scp, y0=T0s[2], name="n3")
        system = System()
        system.add_nodes([n1, n2, n3])
        n1.set_dTdt_convective(source=[n2.y()], hA=[hA])
        n2.set_dTdt_convective(source=[n1.y(), n3.y()], hA=[hA, hA])
        n3.set_dTdt_convective(source=[n2.y()], hA=[hA])

        t_space = np.linspace(0, 500_000, 1000)
        solve_th(system, t_space)

        assert n1.y_out[-1] == pytest.approx(T_eq, rel=1e-3)
        assert n2.y_out[-1] == pytest.approx(T_eq, rel=1e-3)
        assert n3.y_out[-1] == pytest.approx(T_eq, rel=1e-3)
