"""Small GCSA fixture exercises the native binary/ASCII boundary."""

import struct

import numpy as np
import pytest

from fnit.recon_all.gcsa_initial import (
    initial_label, map_initial_nodes, read_initial_atlas,
)


def test_read_initial_atlas_and_classify(tmp_path):
    atlas = tmp_path / "tiny.gcs"
    with atlas.open("wb") as stream:
        def integer(value):
            stream.write(struct.pack(">i", value))

        integer(0xABABCDCD - 0x100000000)
        for value in (1, 0, 0, 0, 15):
            integer(value)
        stream.write(b"mean_curvature\x00")
        integer(5)
        integer(0)
        for _ in range(12):
            integer(2)
            integer(20)
            for label, mean in ((101, 0.0), (202, 1.0)):
                integer(label)
                integer(10)
                stream.write(f"1 1 1\n{mean:+f}  \n1 1 1\n+0.250000  \n".encode())
        for _ in range(12):
            integer(2)
            integer(20)
            for label, prior in ((101, 0.75), (202, 0.25)):
                integer(label)
                stream.write(struct.pack(">f", prior))
                for direction in range(4):
                    integer(1 if direction in (0, 2) else 0)
                    integer(1 if direction in (0, 2) else 0)
                    if direction in (0, 2):
                        integer(101)
                        stream.write(struct.pack(">f", 0.8 if direction == 0 else 0.3))
        integer(1)
        integer(-2)
        integer(2)
        integer(9)
        stream.write(b"test.ctab")
        integer(1)
        integer(1)
        integer(5)
        stream.write(b"ROI1\x00")
        for value in (1, 2, 3, 0):
            integer(value)

    parsed = read_initial_atlas(atlas)
    assert len(parsed.classifier_nodes) == len(parsed.prior_nodes) == 12
    assert parsed.average_variance == pytest.approx(0.25)
    assert (parsed.singular_count, parsed.regularized_count) == (0, 0)
    assert parsed.color_table[1] == ("ROI1", 1, 2, 3, 0)
    assert initial_label(parsed.classifier_nodes[0], parsed.prior_nodes[0], 0.1)[0] == 101
    with_gibbs = read_initial_atlas(atlas, include_gibbs=True)
    assert with_gibbs.gibbs_neighbours[0][0][0][101] == pytest.approx(0.8)
    assert with_gibbs.gibbs_neighbours[0][0][1][101] == pytest.approx(0.3)

    with atlas.open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="trailing data"):
        read_initial_atlas(atlas)


def test_map_initial_nodes_does_not_change_input():
    sphere = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    ico = np.array([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    classifier, prior = map_initial_nodes(sphere, ico, ico)
    np.testing.assert_array_equal(prior, [0, 1])
    np.testing.assert_array_equal(classifier, [0, 1])
    np.testing.assert_array_equal(sphere, [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
