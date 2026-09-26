"""Source-order pial dt and accept/reject decisions."""

from fnit.recon_all.place_surface_decision import pial_step_decision


def test_accept_without_reduction():
    assert pial_step_decision(100.0, 10.0, 80.0, 8.0, 0.5, 0) == (
        0.5, 0, False, False, False)


def test_reduce_next_dt_but_keep_decreasing_rms_step():
    assert pial_step_decision(100.0, 10.0, 90.0, 9.97, 0.5, 0) == (
        0.25, 1, True, False, False)


def test_reject_increasing_rms_and_stop_after_third_reduction():
    assert pial_step_decision(100.0, 10.0, 101.0, 10.01, 0.5, 2) == (
        0.25, 3, True, True, True)
