"""
TH tests: basic
===============

Endpoint convergence checks and trivial-case behavior:
  * single- and multi-source convection settle at the correct asymptote,
  * advection settles at the inlet temperature,
  * zero-coupling, equilibrium-IC, and scale-robustness cases all leave
    the trajectory pinned at ``y0``.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import solve_th

pytestmark = [pytest.mark.th, pytest.mark.basic]


class TestSteadyStateConvergence:
    """Endpoint-only checks at long times (≥10 time constants)."""

    def test_convective_approaches_source_at_steady_state(self):
        """Single-source convective node settles within 0.1% of the source."""
        m, scp, hA = 1.0, 1000.0, 0.01
        T0, T_inf = 300.0, 450.0
        t_const = m * scp / hA
        t_space = np.linspace(0, 10 * t_const, 1000)

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[hA])

        solve_th(system, t_space)
        assert node.y_out[-1] == pytest.approx(T_inf, rel=1e-3)

    def test_convective_two_sources_steady_state_is_hA_weighted_average(self):
        """Two sources equilibrate at the hA-weighted mean."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        T1, T2 = 350.0, 500.0
        hA1, hA2 = 0.01, 0.03

        T_ss = (hA1 * T1 + hA2 * T2) / (hA1 + hA2)
        t_const = m * scp / (hA1 + hA2)
        t_space = np.linspace(0, 10 * t_const, 1000)

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T1, T2], hA=[hA1, hA2])
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T_ss, rel=1e-3)

    def test_convective_dominant_hA_source_controls_steady_state(self):
        """One source with 1000× the ``hA`` of another dominates the equilibrium."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        T_weak, T_strong = 500.0, 310.0
        hA_weak, hA_strong = 0.001, 1.0

        T_ss = (hA_weak * T_weak + hA_strong * T_strong) / (hA_weak + hA_strong)
        t_const = m * scp / (hA_weak + hA_strong)
        t_space = np.linspace(0, 10 * t_const, 1000)

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(
            source=[T_weak, T_strong], hA=[hA_weak, hA_strong]
        )
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T_ss, rel=1e-3)
        # Sanity: dominant source should pull the equilibrium close.
        assert abs(node.y_out[-1] - T_strong) < 0.5

    def test_advective_approaches_inlet_at_steady_state(self):
        """Single-source advective node settles within 0.1% of the inlet temperature."""
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 450.0
        t_res = m / W
        t_space = np.linspace(0, 10 * t_res, 1000)

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T_in, rel=1e-3)


class TestEdgeCases:
    """Zero-coupling, equilibrium-IC, and scale-robustness cases."""

    # ------------------------------------------------------------------
    # Convection
    # ------------------------------------------------------------------

    def test_convective_zero_hA_keeps_temperature_constant(self):
        """``hA = 0`` collapses the convective rate to zero; ``T`` holds at ``T0``."""
        T0 = 300.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[500.0], hA=[0.0])

        t_space = np.linspace(0, 100_000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-10)

    def test_convective_source_equal_to_T0_stays_constant(self):
        """Starting at equilibrium (``source = T0``) holds ``T`` at ``T0``."""
        T0 = 350.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T0], hA=[0.05])

        t_space = np.linspace(0, 10_000, 300)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-8)

    def test_convective_very_large_mass_barely_moves(self):
        """Scale-robustness: ``t/t_const = 10⁻⁸`` produces no motion."""
        T0, T_inf = 300.0, 400.0
        node = Node(m=1e6, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[0.01])

        t_space = np.linspace(0, 1000, 200)
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T0, abs=1e-3)

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------

    def test_advective_zero_W_keeps_temperature_constant(self):
        """``W = 0`` (no flow) collapses the advective rate to zero."""
        T0 = 300.0
        node = Node(m=1.0, scp=1000.0, W=0.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=500.0)

        t_space = np.linspace(0, 1000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-10)

    def test_advective_source_equal_to_T0_stays_constant(self):
        """Advective node starting at equilibrium (``source = T0``) holds ``T``."""
        T0 = 350.0
        node = Node(m=1.0, scp=1000.0, W=0.05, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T0)

        t_space = np.linspace(0, 1000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-8)

    def test_advective_very_large_mass_barely_moves(self):
        """Scale-robustness for advection: ``t/t_res = 10⁻⁸`` produces no motion."""
        T0, T_in = 300.0, 400.0
        node = Node(m=1e6, scp=1000.0, W=1e-5, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 1000, 200)
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T0, abs=1e-3)

    # ------------------------------------------------------------------
    # Internal heat generation
    # ------------------------------------------------------------------

    def test_internal_zero_source_keeps_temperature_constant(self):
        """``source = 0`` collapses the internal rate to zero (with non-zero ``k``)."""
        T0 = 350.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[0.0], k=[1000.0])

        t_space = np.linspace(0, 1000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-10)

    def test_internal_zero_k_keeps_temperature_constant(self):
        """``k = 0`` collapses the internal rate to zero (with non-zero ``source``)."""
        T0 = 350.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[1.0], k=[0.0])

        t_space = np.linspace(0, 1000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-10)
