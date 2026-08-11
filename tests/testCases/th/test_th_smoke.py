"""
TH tests: smoke
===============

Constructor and basic-solve invariants for thermal-hydraulic Nodes:
constructor arguments stored verbatim, ``y_out[0] == y0``, output length
matches input grid, ``System.add_nodes()`` registers nodes correctly,
duplicate names rejected, and a bare node holds its initial value.
"""
import numpy as np
import pytest
from msrDynamics import Node, System
from conftest import solve_th

pytestmark = [pytest.mark.th, pytest.mark.smoke]


class TestSmokeTests:
    """Constructor and basic-solve invariants."""

    # ------------------------------------------------------------------
    # Constructor argument storage
    # ------------------------------------------------------------------

    def test_node_stores_y0(self):
        """``y0`` is stored verbatim on the instance."""
        node = Node(m=1.0, scp=1000.0, y0=350.0, name="n")
        assert node.y0 == 350.0

    def test_node_stores_m(self):
        """``m`` is stored verbatim on the instance."""
        node = Node(m=5.0, scp=800.0, y0=300.0, name="n")
        assert node.m == 5.0

    def test_node_stores_scp(self):
        """``scp`` is stored verbatim on the instance."""
        node = Node(m=1.0, scp=4200.0, y0=300.0, name="n")
        assert node.scp == 4200.0

    def test_node_stores_name(self):
        """``name`` is stored verbatim on the instance."""
        node = Node(m=1.0, scp=1000.0, y0=300.0, name="myNode")
        assert node.name == "myNode"

    def test_node_stores_W(self):
        """``W`` is stored verbatim on the instance."""
        node = Node(m=1.0, scp=1000.0, W=2.5, y0=300.0, name="n")
        assert node.W == 2.5

    # ------------------------------------------------------------------
    # Basic solver output invariants
    # ------------------------------------------------------------------

    def test_convective_solved_output_starts_at_y0(self):
        """Integrator's first output sample equals ``y0`` (convective path)."""
        T0 = 355.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[310.0], hA=[0.01])

        solve_th(system, np.linspace(0, 1000, 200))

        assert node.y_out[0] == pytest.approx(T0, rel=1e-6)

    def test_advective_solved_output_starts_at_y0(self):
        """Integrator's first output sample equals ``y0`` (advective path)."""
        T0 = 355.0
        node = Node(m=1.0, scp=1000.0, W=0.01, y0=T0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_advective(source=310.0)

        solve_th(system, np.linspace(0, 1000, 200))

        assert node.y_out[0] == pytest.approx(T0, rel=1e-6)

    def test_output_length_matches_input_grid(self):
        """``len(node.y_out) == len(t_space)``. Grid size 137 is prime so
        a pass cannot be a coincidence with internal defaults of 100/128/200."""
        node = Node(m=1.0, scp=1000.0, y0=300.0, name="n")
        system = System()
        system.add_nodes([node])
        node.set_dTdt_convective(source=[310.0], hA=[0.01])

        t_space = np.linspace(0, 1000, 137)
        solve_th(system, t_space)

        assert len(node.y_out) == 137

    # ------------------------------------------------------------------
    # System-level registration
    # ------------------------------------------------------------------

    def test_add_nodes_registers_node_by_name(self):
        """``System.add_nodes()`` registers the node and sets membership flags."""
        node = Node(m=1.0, scp=1000.0, y0=300.0, name="n")
        system = System()
        system.add_nodes([node])

        assert system.nodes.get("n") is node
        assert node.in_system is True
        assert node.index is not None

    def test_duplicate_node_names_rejected(self):
        """Duplicate names raise ``AssertionError`` and the first entry is preserved."""
        n1 = Node(m=1.0, scp=1000.0, y0=300.0, name="duplicate")
        n2 = Node(m=2.0, scp=1000.0, y0=310.0, name="duplicate")
        system = System()
        system.add_nodes([n1])

        with pytest.raises(AssertionError):
            system.add_nodes([n2])

        assert system.nodes.get("duplicate") is n1

    # ------------------------------------------------------------------
    # Bare-node behaviour
    # ------------------------------------------------------------------

    def test_bare_node_holds_initial_value(self):
        """A Node with no ``set_dTdt_*`` call held at ``y0`` by the integrator."""
        T0 = 300.0
        node = Node(m=1.0, scp=1000.0, y0=T0, name="bare")
        system = System()
        system.add_nodes([node])
        # NOTE: no set_dTdt_* call. node.dydt evaluates to 0.0.

        t_space = np.linspace(0, 10_000, 500)
        solve_th(system, t_space)

        np.testing.assert_allclose(node.y_out, T0, rtol=1e-10, atol=1e-10)
