import numpy as np
import pytest

from msrDynamics import Node, System


DH_YIELD = np.array(
    [
        2.3751e-03,
        8.7763e-05,
        1.9596e-06,
    ],
    dtype=float,
)

DH_LAMBDA = np.array(
    [
        9.453e-02,
        4.420e-03,
        8.6098e-05,
    ],
    dtype=float,
)

# ANS94, 2-sigma, 100% U-235 decay-heat fraction.
ANS94_TIME = np.array(
    [
        1.0,
        1.5,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        15.0,
        20.0,
        40.0,
        60.0,
        80.0,
        100.0,
        150.0,
        200.0,
        400.0,
        600.0,
        800.0,
        1000.0,
        1500.0,
        2000.0,
        4000.0,
        6000.0,
    ],
    dtype=float,
)

ANS94_DECAY_FRACTION = np.array(
    [
        0.06896,
        0.06679,
        0.06506,
        0.06027,
        0.05715,
        0.05485,
        0.05305,
        0.04979,
        0.04753,
        0.04224,
        0.03923,
        0.03714,
        0.03558,
        0.03291,
        0.03115,
        0.02733,
        0.02520,
        0.02365,
        0.02244,
        0.02020,
        0.01862,
        0.01512,
        0.01343,
    ],
    dtype=float,
)


def build_shutdown_decay_heat_system():
    """
    Construct a three-group decay-heat model immediately after shutdown.

    The reactor is assumed to have operated at normalized full power long
    enough for each decay-heat group to reach equilibrium:

        H_i(0) = gamma_i / lambda_i

    Following shutdown:

        n(t) = 0
        n0   = 1

    Therefore each decay-heat group follows:

        dH_i/dt = -lambda_i H_i
    """

    normalized_power = Node(y0=0.0)

    decay_groups = [
        Node(y0=rel_yield / lam)
        for rel_yield, lam in zip(DH_YIELD, DH_LAMBDA)
    ]

    system = System()
    system.add_nodes([normalized_power, *decay_groups])

    # Hold normalized reactor power at zero following shutdown.
    normalized_power.dydt = 0.0

    for group, rel_yield, lam in zip(
        decay_groups,
        DH_YIELD,
        DH_LAMBDA,
    ):
        group.set_dndt_decay(
            n=normalized_power.y(),
            n0=1.0,
            rel_yield=rel_yield,
            lam=lam,
        )

    return system, normalized_power, decay_groups


@pytest.mark.pke
class TestDecayHeatGroups:
    def test_shutdown_groups_match_exact_exponential_solution(self):
        """
        Verify that each decay-heat group follows its analytical solution.

        Following shutdown:

            H_i(t) = H_i(0) exp(-lambda_i t)

        This is an implementation test of set_dndt_decay, not a test of
        the three-group approximation against the ANS standard.
        """

        system, normalized_power, decay_groups = (
            build_shutdown_decay_heat_system()
        )

        time = np.linspace(0.0, 6000.0, 601)

        system.solve(
            time,
            populate_nodes=True,
        )

        np.testing.assert_allclose(
            np.asarray(normalized_power.y_out, dtype=float),
            np.zeros_like(time),
            rtol=0.0,
            atol=1e-12,
        )

        for group, rel_yield, lam in zip(
            decay_groups,
            DH_YIELD,
            DH_LAMBDA,
        ):
            expected = (
                rel_yield / lam
            ) * np.exp(-lam * time)

            np.testing.assert_allclose(
                np.asarray(group.y_out, dtype=float),
                expected,
                rtol=1e-5,
                atol=1e-10,
            )

    def test_three_group_curve_matches_ans94_within_ten_percent(self):
        """
        Validate the summed three-group decay-heat curve against the
        ANS94 2-sigma, 100% U-235 values from 1 through 6000 seconds.

        The three-group model is a reduced-order approximation optimized
        for this shorter post-shutdown interval, so a 10% pointwise
        relative-error acceptance criterion is used.
        """

        system, _, decay_groups = build_shutdown_decay_heat_system()

        # Include t = 0 so the solver starts from the defined initial state.
        solve_time = np.concatenate(([0.0], ANS94_TIME))

        system.solve(
            solve_time,
            populate_nodes=True,
        )

        calculated_total = np.sum(
            np.asarray(
                [group.y_out for group in decay_groups],
                dtype=float,
            ),
            axis=0,
        )

        # Remove the t = 0 result before comparing with the ANS94 table.
        calculated_at_ans_times = calculated_total[1:]

        relative_error = np.abs(
            (
                calculated_at_ans_times
                - ANS94_DECAY_FRACTION
            )
            / ANS94_DECAY_FRACTION
        )

        max_error_index = int(np.argmax(relative_error))

        assert np.all(relative_error <= 0.10), (
            "Three-group decay-heat model exceeded the 10% ANS94 "
            "acceptance criterion.\n"
            f"Maximum relative error: "
            f"{relative_error[max_error_index]:.3%}\n"
            f"Time of maximum error: "
            f"{ANS94_TIME[max_error_index]:.1f} s\n"
            f"Calculated value: "
            f"{calculated_at_ans_times[max_error_index]:.8f}\n"
            f"ANS94 value: "
            f"{ANS94_DECAY_FRACTION[max_error_index]:.8f}"
        )
