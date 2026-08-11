"""
TH delay tests: compounded
==========================

Multi-mechanism, multi-node delayed-source configurations:
  * three-node advective chain with pipe delays between each pair,
  * closed primary loop with hot- and cold-leg delays, internal heat
    in the core node, and convective heat removal at the heat exchanger.

In-transit energy is not tracked by the lumped-node model, so a strict
total-enthalpy invariant is not testable here. We verify the asymptotic
steady-state energy balance instead (which is independent of the delays).
"""
import numpy as np
import pytest
from jitcdde import t as jitcdde_t
from msrDynamics import Node, System
from conftest import solve_th

pytestmark = [pytest.mark.th, pytest.mark.delay, pytest.mark.compounded]


class TestThreeNodeChainWithDelays:
    """Chain ``up → [τ₁] → mid → [τ₂] → dn``, all advective.

    Each downstream node receives the *delayed* upstream temperature
    as its advective source. At steady state, all three converge to
    the inlet T_in regardless of delay; during the transient, the
    lag accumulates along the chain.
    """

    def test_three_node_chain_settles_to_inlet(self):
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        tau1, tau2 = 3.0, 4.0

        sys = System()
        up  = Node(m=m, scp=1000.0, W=W, y0=T0, name='up')
        mid = Node(m=m, scp=1000.0, W=W, y0=T0, name='mid')
        dn  = Node(m=m, scp=1000.0, W=W, y0=T0, name='dn')
        sys.add_nodes([up, mid, dn])

        up.set_dTdt_advective(source=T_in)
        mid.set_dTdt_advective(source=up.y(tau=jitcdde_t - tau1))
        dn.set_dTdt_advective(source=mid.y(tau=jitcdde_t - tau2))

        # Each stage has time constant m/W = 100s; integrate well past 5τ
        T = np.linspace(0.0, 1500.0, 1501)
        solve_th(sys, T,
                 max_delay=max(tau1, tau2) + 1.0,
                 tight=False)

        for node in (up, mid, dn):
            assert node.y_out[-1] == pytest.approx(T_in, rel=5e-3)

    def test_cumulative_lag_along_chain(self):
        """At an intermediate snapshot the chain is strictly ordered:
        T_up > T_mid > T_dn > T0 — each stage is progressively further
        behind the inlet step."""
        m, W = 1.0, 0.01
        T0, T_in = 300.0, 400.0
        tau1, tau2 = 3.0, 4.0
        t_sample = 50.0   # well past both τ, well before SS (τ_th = 100s)

        sys = System()
        up  = Node(m=m, scp=1000.0, W=W, y0=T0, name='up')
        mid = Node(m=m, scp=1000.0, W=W, y0=T0, name='mid')
        dn  = Node(m=m, scp=1000.0, W=W, y0=T0, name='dn')
        sys.add_nodes([up, mid, dn])

        up.set_dTdt_advective(source=T_in)
        mid.set_dTdt_advective(source=up.y(tau=jitcdde_t - tau1))
        dn.set_dTdt_advective(source=mid.y(tau=jitcdde_t - tau2))

        T = np.linspace(0.0, t_sample, 501)
        solve_th(sys, T, max_delay=max(tau1, tau2) + 1.0)

        assert up.y_out[-1]  > mid.y_out[-1]
        assert mid.y_out[-1] > dn.y_out[-1]
        assert dn.y_out[-1]  > T0          # has begun moving
        assert up.y_out[-1]  < T_in        # not yet at SS


class TestClosedPrimaryLoopWithDelays:
    """Closed loop: core (internal heat) → hot pipe (τ_hot) → HX
    (convective sink to T_sec) → cold pipe (τ_cold) → core.

    Parameters and steady-state derivation:

        m       = 1.0        scp = 1000        W = 0.01
        hA      = 10.0       Q   = 1 K/s       T_sec = 300
        τ_hot   = 2.0        τ_cold = 3.0

      Core energy balance at SS:
          0 = (T_hx − T_core)·W/m + Q
          ⇒ T_core − T_hx = Q·m/W = 100

      HX energy balance at SS:
          0 = (T_core − T_hx)·W/m + hA·(T_sec − T_hx)/(m·scp)
          ⇒ (T_core − T_hx)·W·scp = hA·(T_hx − T_sec)
          ⇒ 100 · 10 = 10·(T_hx − 300)
          ⇒ T_hx = 400,  T_core = 500.

    The delays do not enter the steady-state equations.
    """

    # Shared loop builder
    @staticmethod
    def _build_loop(T0=350.0, Q=1.0, m=1.0, scp=1000.0, W=0.01, hA=10.0,
                    T_sec=300.0, tau_hot=2.0, tau_cold=3.0):
        sys = System()
        core = Node(m=m, scp=scp, W=W, y0=T0, name='core')
        hx   = Node(m=m, scp=scp, W=W, y0=T0, name='hx')
        sys.add_nodes([core, hx])

        # core: heated internally, fed by HX outlet via the cold leg (delayed)
        core.set_dTdt_advective(source=hx.y(tau=jitcdde_t - tau_cold))
        core.set_dTdt_internal(source=[1.0], k=[Q * m * scp])

        # HX: fed by core outlet via the hot leg (delayed), sinks to T_sec
        hx.set_dTdt_advective(source=core.y(tau=jitcdde_t - tau_hot))
        hx.set_dTdt_convective(source=[T_sec], hA=[hA])

        return sys, core, hx, max(tau_hot, tau_cold)

    def test_closed_loop_reaches_predicted_steady_state(self):
        T_hx_ss, T_core_ss = 400.0, 500.0
        sys, core, hx, tau_max = self._build_loop()

        # Slowest time constant in the loop is ~m·scp/hA = 100s; 50τ is safe.
        T = np.linspace(0.0, 5000.0, 5001)
        solve_th(sys, T, max_delay=tau_max + 1.0, tight=False)

        assert hx.y_out[-1]   == pytest.approx(T_hx_ss,   rel=5e-3)
        assert core.y_out[-1] == pytest.approx(T_core_ss, rel=5e-3)
        # Predicted core-HX temperature drop at SS
        assert (core.y_out[-1] - hx.y_out[-1]) == pytest.approx(100.0, rel=1e-2)

    def test_closed_loop_temperatures_bounded(self):
        """Sanity-check the trajectory: never falls below T_sec, never
        overshoots the analytical SS unboundedly."""
        sys, core, hx, tau_max = self._build_loop()

        T = np.linspace(0.0, 5000.0, 1001)
        solve_th(sys, T, max_delay=tau_max + 1.0, tight=False)

        assert np.all(core.y_out >= 300.0 - 1.0)
        assert np.all(hx.y_out   >= 300.0 - 1.0)
        # Predicted SS values are 500 and 400; allow modest transient overshoot
        assert np.max(core.y_out) < 600.0
        assert np.max(hx.y_out)   < 500.0

    def test_closed_loop_ss_independent_of_delay_magnitude(self):
        """Different delays must still reach the same SS — the delays
        only shape the transient, not the equilibrium."""
        T_hx_ss, T_core_ss = 400.0, 500.0

        for tau_hot, tau_cold in [(0.5, 0.5), (2.0, 3.0), (5.0, 5.0)]:
            sys, core, hx, tau_max = self._build_loop(
                tau_hot=tau_hot, tau_cold=tau_cold
            )
            T = np.linspace(0.0, 5000.0, 2001)
            solve_th(sys, T, max_delay=tau_max + 1.0, tight=False)
            assert hx.y_out[-1]   == pytest.approx(T_hx_ss,   rel=1e-2)
            assert core.y_out[-1] == pytest.approx(T_core_ss, rel=1e-2)
