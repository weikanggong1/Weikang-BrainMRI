"""Pinned FreeSurfer 8.2 first-epoch float32 quadratic-fit regression."""

from fnit.recon_all.sphere_standard_line_search import _quadratic_fit_float32


def test_first_epoch_vnl_quadratic_fit_both_hemispheres():
    left_dt = [0.004935609672628332, 0.009871219345256664,
               0.014806829017884996]
    left_sse = [2625053.012899546, 2623582.211206758,
                2623155.068984963]
    assert _quadratic_fit_float32(left_dt, left_sse) == (
        21495808.0, -306176.0, 2627632.0)

    right_dt = [2.08044813469088, 4.16089626938176, 6.24134440407264]
    right_sse = [2631386.269562474, 2605009.44653865,
                 2582398.5766699687]
    assert _quadratic_fit_float32(right_dt, right_sse) == (
        436.0, -7688.0, 2661536.0)
