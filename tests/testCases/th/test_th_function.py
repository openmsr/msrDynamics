"""
TH tests: function
==================

Function-level closed-form comparisons, structural invariants, parameter
sweeps, and overwrite semantics:

  * full-trajectory matches to closed-form solutions (convection, advection, internal),
  * monotonicity and boundedness across both heating and cooling directions,
  * parameter sweeps confirming m/W/hA/scp are wired into the right denominators,
  * overwrite contract: repeated ``set_dTdt_*`` calls REPLACE, not accumulate.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import solve_th

pytestmark = [pytest.mark.th, pytest.mark.function]


# ===========================================================================
# Analytical comparisons
# ===========================================================================

class TestAnalyticalComparisons:
    """Full-trajectory matches against closed-form analytical solutions."""

    # ------------------------------------------------------------------
    # Single-source convection
    # ------------------------------------------------------------------

    def test_convective_heating_matches_analytical(self):
        """Convective heating reproduces ``T_inf + (T0-T_inf)·exp(-rate·t)``."""
        m, scp, hA = 1.0, 1000.0, 0.01
        T0, T_inf = 300.0, 400.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[hA])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        rate = hA / (m * scp)
        expected = T_inf + (T0 - T_inf) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_convective_cooling_matches_analytical(self):
        """Convective cooling reproduces the same analytical form."""
        m, scp, hA = 1.0, 1000.0, 0.01
        T0, T_inf = 500.0, 300.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[hA])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        rate = hA / (m * scp)
        expected = T_inf + (T0 - T_inf) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    # ------------------------------------------------------------------
    # Multi-source convection
    # ------------------------------------------------------------------

    def test_convective_two_sources_match_weighted_analytical(self):
        """Two convective sources sum into a first-order ODE driven toward ``T_ss``."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        T1, T2 = 350.0, 500.0
        hA1, hA2 = 0.01, 0.03

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T1, T2], hA=[hA1, hA2])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        T_ss = (hA1 * T1 + hA2 * T2) / (hA1 + hA2)
        rate = (hA1 + hA2) / (m * scp)
        expected = T_ss + (T0 - T_ss) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_convective_three_equal_sources_converge_to_arithmetic_mean(self):
        """Three sources with equal ``hA`` produce arithmetic-mean equilibrium."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        temps = [350.0, 400.0, 450.0]
        hA_list = [0.01, 0.01, 0.01]

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=temps, hA=hA_list)

        T_ss = sum(t * h for t, h in zip(temps, hA_list)) / sum(hA_list)
        t_const = m * scp / sum(hA_list)
        t_space = np.linspace(0, 10 * t_const, 1000)
        solve_th(system, t_space)

        assert node.y_out[-1] == pytest.approx(T_ss, rel=1e-3)

    # ------------------------------------------------------------------
    # Single-source advection
    # ------------------------------------------------------------------

    def test_advective_heating_matches_analytical(self):
        """Advective heating reproduces ``T_in + (T0-T_in)·exp(-W·t/m)``."""
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 200, 1000)
        solve_th(system, t_space)

        rate = W / m
        expected = T_in + (T0 - T_in) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_advective_cooling_matches_analytical(self):
        """Advective cooling reproduces the same analytical form."""
        m, W = 1.0, 0.01
        T0, T_in = 500.0, 300.0

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 200, 1000)
        solve_th(system, t_space)

        rate = W / m
        expected = T_in + (T0 - T_in) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    # ------------------------------------------------------------------
    # Internal heat generation
    # ------------------------------------------------------------------

    def test_internal_constant_source_produces_linear_ramp(self):
        """Constant internal source gives ``T(t) = T0 + k·s·t/(m·scp)``."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        s_val, k_val = 1.0, 1000.0  # rate = k·s/(m·scp) = 1.0 K/s

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[s_val], k=[k_val])

        t_space = np.linspace(0, 100, 500)
        solve_th(system, t_space)

        rate = k_val * s_val / (m * scp)
        expected = T0 + rate * t_space
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_internal_two_sources_sum_correctly(self):
        """Two constant internal sources produce a ramp with combined slope."""
        m, scp, T0 = 1.0, 1000.0, 300.0
        s1, k1 = 2.0, 500.0  # contributes 1.0 K/s
        s2, k2 = 1.0, 200.0  # contributes 0.2 K/s

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[s1, s2], k=[k1, k2])

        t_space = np.linspace(0, 100, 500)
        solve_th(system, t_space)

        rate = (k1 * s1 + k2 * s2) / (m * scp)
        expected = T0 + rate * t_space
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_internal_negative_source_cools_node(self):
        """A negative internal source produces a linear ramp **downward**."""
        m, scp, T0 = 1.0, 1000.0, 400.0
        s_val, k_val = -1.0, 1000.0  # rate = -1.0 K/s

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_internal(source=[s_val], k=[k_val])

        t_space = np.linspace(0, 100, 500)
        solve_th(system, t_space)

        rate = k_val * s_val / (m * scp)
        expected = T0 + rate * t_space
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)


# ===========================================================================
# Monotonicity and boundedness
# ===========================================================================

class TestMonotonicityAndBoundedness:
    """Physical-trajectory invariants: monotonicity and no over/undershoot."""

    # ------------------------------------------------------------------
    # Convection
    # ------------------------------------------------------------------

    def test_convective_monotonic_when_heating(self):
        """Convective heating produces monotonic non-decreasing output."""
        m, scp, hA = 1.0, 1000.0, 0.01
        T0, T_inf = 300.0, 400.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[hA])

        t_space = np.linspace(0, 50_000, 500)
        solve_th(system, t_space)

        diffs = np.diff(node.y_out)
        assert np.all(diffs >= -1e-12)

    def test_convective_monotonic_when_cooling(self):
        """Convective cooling produces monotonic non-increasing output."""
        m, scp, hA = 1.0, 1000.0, 0.01
        T0, T_inf = 500.0, 300.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[hA])

        t_space = np.linspace(0, 50_000, 500)
        solve_th(system, t_space)

        diffs = np.diff(node.y_out)
        assert np.all(diffs <= 1e-12)

    def test_convective_no_overshoot_when_heating(self):
        """Convective heating never exceeds the source temperature."""
        T0, T_inf = 300.0, 400.0

        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[0.01])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        assert np.all(node.y_out <= T_inf + 1e-8)

    def test_convective_no_undershoot_when_cooling(self):
        """Convective cooling never drops below the source temperature."""
        T0, T_inf = 500.0, 300.0

        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[T_inf], hA=[0.01])

        t_space = np.linspace(0, 200_000, 1000)
        solve_th(system, t_space)

        assert np.all(node.y_out >= T_inf - 1e-8)

    # ------------------------------------------------------------------
    # Advection
    # ------------------------------------------------------------------

    def test_advective_monotonic_when_heating(self):
        """Advective heating produces monotonic non-decreasing output."""
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        t_res = m / W

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 5 * t_res, 500)
        solve_th(system, t_space)

        diffs = np.diff(node.y_out)
        assert np.all(diffs >= -1e-12)

    def test_advective_monotonic_when_cooling(self):
        """Advective cooling produces monotonic non-increasing output."""
        m, W = 1.0, 0.01
        T0, T_in = 500.0, 300.0
        t_res = m / W

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 5 * t_res, 500)
        solve_th(system, t_space)

        diffs = np.diff(node.y_out)
        assert np.all(diffs <= 1e-12)

    def test_advective_no_overshoot_when_heating(self):
        """Advective heating never exceeds the inlet temperature."""
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        t_res = m / W

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 5 * t_res, 1000)
        solve_th(system, t_space)

        assert np.all(node.y_out <= T_in + 1e-8)

    def test_advective_no_undershoot_when_cooling(self):
        """Advective cooling never drops below the inlet temperature."""
        m, W = 1.0, 0.01
        T0, T_in = 500.0, 300.0
        t_res = m / W

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=T_in)

        t_space = np.linspace(0, 5 * t_res, 1000)
        solve_th(system, t_space)

        assert np.all(node.y_out >= T_in - 1e-8)


# ===========================================================================
# Parameter scaling
# ===========================================================================

class TestParameterScaling:
    """Comparative tests where one parameter is varied at a time."""

    def test_convective_larger_mass_slows_response(self):
        """``light.y_out[-1] > heavy.y_out[-1]`` after a fixed integration time."""
        hA, T0, T_inf = 0.01, 300.0, 400.0
        t_space = np.linspace(0, 20_000, 500)

        light = Node(m=1.0, scp=1000.0, y0=T0, name="light")
        sys_L = System()
        sys_L.add_nodes([light])
        light.set_dTdt_convective(source=[T_inf], hA=[hA])
        solve_th(sys_L, t_space)

        heavy = Node(m=10.0, scp=1000.0, y0=T0, name="heavy")
        sys_H = System()
        sys_H.add_nodes([heavy])
        heavy.set_dTdt_convective(source=[T_inf], hA=[hA])
        solve_th(sys_H, t_space)

        assert light.y_out[-1] > heavy.y_out[-1]

    def test_convective_higher_hA_speeds_up_response(self):
        """``fast.y_out[-1] > slow.y_out[-1]`` after a fixed integration time."""
        m, scp, T0, T_inf = 1.0, 1000.0, 300.0, 400.0
        t_space = np.linspace(0, 10_000, 500)

        slow = Node(m=m, scp=scp, y0=T0, name="slow")
        sys_S = System()
        sys_S.add_nodes([slow])
        slow.set_dTdt_convective(source=[T_inf], hA=[0.001])
        solve_th(sys_S, t_space)

        fast = Node(m=m, scp=scp, y0=T0, name="fast")
        sys_F = System()
        sys_F.add_nodes([fast])
        fast.set_dTdt_convective(source=[T_inf], hA=[0.01])
        solve_th(sys_F, t_space)

        assert fast.y_out[-1] > slow.y_out[-1]

    def test_advective_larger_mass_slows_response(self):
        """Advective: light (``m=1``) heats faster than heavy (``m=10``)."""
        W, T0, T_in = 0.01, 300.0, 400.0
        t_space = np.linspace(0, 200.0, 500)

        light = Node(m=1.0, scp=1000.0, W=W, y0=T0, name="light")
        sys_L = System()
        sys_L.add_nodes([light])
        light.set_dTdt_advective(source=T_in)
        solve_th(sys_L, t_space)

        heavy = Node(m=10.0, scp=1000.0, W=W, y0=T0, name="heavy")
        sys_H = System()
        sys_H.add_nodes([heavy])
        heavy.set_dTdt_advective(source=T_in)
        solve_th(sys_H, t_space)

        assert light.y_out[-1] > heavy.y_out[-1]

    def test_advective_higher_W_speeds_up_response(self):
        """Advective: fast (``W=0.01``) heats faster than slow (``W=0.001``)."""
        m, T0, T_in = 1.0, 300.0, 400.0
        t_space = np.linspace(0, 200.0, 500)

        slow = Node(m=m, scp=1000.0, W=0.001, y0=T0, name="slow")
        sys_S = System()
        sys_S.add_nodes([slow])
        slow.set_dTdt_advective(source=T_in)
        solve_th(sys_S, t_space)

        fast = Node(m=m, scp=1000.0, W=0.01, y0=T0, name="fast")
        sys_F = System()
        sys_F.add_nodes([fast])
        fast.set_dTdt_advective(source=T_in)
        solve_th(sys_F, t_space)

        assert fast.y_out[-1] > slow.y_out[-1]

    def test_larger_scp_slows_convective_response(self):
        """Convective rate ``hA/(m·scp)`` — larger ``scp`` produces a slower response."""
        m, hA, T0, T_inf = 1.0, 0.05, 300.0, 400.0
        t_space = np.linspace(0, 5_000.0, 500)

        low = Node(m=m, scp=1000.0, y0=T0, name="low_cp")
        sys_L = System()
        sys_L.add_nodes([low])
        low.set_dTdt_convective(source=[T_inf], hA=[hA])
        solve_th(sys_L, t_space)

        high = Node(m=m, scp=4200.0, y0=T0, name="high_cp")
        sys_H = System()
        sys_H.add_nodes([high])
        high.set_dTdt_convective(source=[T_inf], hA=[hA])
        solve_th(sys_H, t_space)

        assert low.y_out[-1] > high.y_out[-1]

    def test_larger_scp_slows_internal_response(self):
        """Internal rate ``k·s/(m·scp)`` — larger ``scp`` slows the ramp."""
        m, T0 = 1.0, 300.0
        s_val, k = 1.0, 1000.0
        t_space = np.linspace(0, 100.0, 500)

        low = Node(m=m, scp=1000.0, y0=T0, name="low_cp")
        sys_L = System()
        sys_L.add_nodes([low])
        low.set_dTdt_internal(source=[s_val], k=[k])
        solve_th(sys_L, t_space)

        high = Node(m=m, scp=4200.0, y0=T0, name="high_cp")
        sys_H = System()
        sys_H.add_nodes([high])
        high.set_dTdt_internal(source=[s_val], k=[k])
        solve_th(sys_H, t_space)

        assert low.y_out[-1] > high.y_out[-1]

    def test_advective_response_independent_of_scp(self):
        """Advective rate contains **no** ``scp`` — trajectories must be identical."""
        m, W, T0, T_in = 1.0, 0.01, 300.0, 400.0
        t_space = np.linspace(0, 500, 1000)

        low = Node(m=m, scp=1000.0, W=W, y0=T0, name="low_cp")
        sys_L = System()
        sys_L.add_nodes([low])
        low.set_dTdt_advective(source=T_in)
        solve_th(sys_L, t_space)

        high = Node(m=m, scp=4200.0, W=W, y0=T0, name="high_cp")
        sys_H = System()
        sys_H.add_nodes([high])
        high.set_dTdt_advective(source=T_in)
        solve_th(sys_H, t_space)

        np.testing.assert_allclose(low.y_out, high.y_out, rtol=1e-10, atol=1e-10)


# ===========================================================================
# Overwrite semantics
# ===========================================================================

class TestOverwriteSemantics:
    """Repeated-setter overwrite contract: each ``set_dTdt_*`` REPLACES
    the prior contribution rather than accumulating."""

    def test_convective_setter_overwrites_on_second_call(self):
        """Second ``set_dTdt_convective`` call replaces the first contribution.

        First call: cool toward 100 K, fast (``hA=0.1``); second call:
        heat toward 400 K, slow (``hA=0.01``). Overwrite → ``T → 400``.
        Accumulation would give ``≈127 K`` — 270 K different.
        """
        m, scp, T0 = 1.0, 1000.0, 300.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])

        node.set_dTdt_convective(source=[100.0], hA=[0.1])
        node.set_dTdt_convective(source=[400.0], hA=[0.01])

        T_inf, hA_used = 400.0, 0.01
        t_const = m * scp / hA_used
        t_space = np.linspace(0, 2 * t_const, 1000)
        solve_th(system, t_space)

        rate = hA_used / (m * scp)
        expected = T_inf + (T0 - T_inf) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_advective_setter_overwrites_on_second_call(self):
        """Second ``set_dTdt_advective`` call replaces the first contribution."""
        m, W, T0 = 1.0, 0.01, 300.0

        node = Node(m=m, scp=1000.0, W=W, y0=T0, name="n")
        system = System()
        system.add_nodes([node])

        node.set_dTdt_advective(source=100.0)
        node.set_dTdt_advective(source=400.0)

        T_in_used = 400.0
        t_res = m / W
        t_space = np.linspace(0, 5 * t_res, 1000)
        solve_th(system, t_space)

        rate = W / m
        expected = T_in_used + (T0 - T_in_used) * np.exp(-rate * t_space)
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)

    def test_internal_setter_overwrites_on_second_call(self):
        """Second ``set_dTdt_internal`` call replaces the first contribution.

        First call uses ``k=-1000`` (cool at 1 K/s), second call uses
        ``k=+500`` (heat at 0.5 K/s). Overwrite → rising linear trajectory;
        accumulation → falling. Opposite directions, 100 K difference at t=100s.
        """
        m, scp, T0 = 1.0, 1000.0, 350.0

        node = Node(m=m, scp=scp, y0=T0, name="n")
        system = System()
        system.add_nodes([node])

        node.set_dTdt_internal(source=[1.0], k=[-1000.0])
        node.set_dTdt_internal(source=[1.0], k=[500.0])

        rate_used = 500.0 * 1.0 / (m * scp)
        t_space = np.linspace(0, 100, 500)
        solve_th(system, t_space)

        expected = T0 + rate_used * t_space
        np.testing.assert_allclose(node.y_out, expected, rtol=1e-4, atol=1e-6)
