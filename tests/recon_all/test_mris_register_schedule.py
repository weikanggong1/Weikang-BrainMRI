"""FreeSurfer 8.2 saved registration decisions for the fixed T1."""

from fnit.recon_all.mris_register_schedule import next_smoothwm_scale


def test_smoothwm_first_scale_stops_at_native_lh_epoch_58():
    assert next_smoothwm_scale(
        "smoothwm", 0, 1024, 1, 1429704.5365436217,
        1418705.2774543008, 3.447789430618286) == ("smoothwm", 0, 256, 0)


def test_smoothwm_last_sigma_branches_on_negative_faces():
    assert next_smoothwm_scale(
        "smoothwm", 3, 0, 1, 1_000_000, 999_999, 0.1,
        negative_faces=193) == ("fold_cleanup", 3, 64, 0)
    assert next_smoothwm_scale(
        "smoothwm", 3, 0, 1, 1_000_000, 999_999, 0.1) is None


def test_fold_cleanup_zero_step_ends_current_scale():
    assert next_smoothwm_scale(
        "fold_cleanup", 3, 64, 1, 1_000_000, 999_999, 0) == (
            "fold_cleanup", 3, 16, 0)
