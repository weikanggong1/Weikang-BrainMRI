"""Source-level checks for the deterministic topology preflight."""

import numpy as np

from fnit.recon_all.topology_preflight_python import (
    _edge_intersects,
    _project_sphere_point,
    center_sphere,
    genetic_base_translation,
    topology_counts,
)


def test_topology_counts_detect_closed_open_and_nonmanifold_meshes():
    tetrahedron = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    closed = topology_counts(tetrahedron, 4)
    assert (closed.vertices, closed.faces, closed.edges, closed.euler) == (4, 4, 6, 2)
    assert (closed.boundary_edges, closed.nonmanifold_edges) == (0, 0)

    opened = topology_counts(tetrahedron[:-1], 4)
    assert opened.euler == 1
    assert opened.boundary_edges == 3

    repeated = topology_counts(np.repeat(tetrahedron[:1], 3, axis=0), 4)
    assert repeated.nonmanifold_edges == 3


def test_spherical_edge_crossing_excludes_shared_endpoints():
    xyz = np.array([[1, 0, 1], [-1, 0, 1], [0, 1, 1],
                    [0, -1, 1], [0, 2, 1]], np.float32)
    assert _edge_intersects(xyz, 0, 1, 2, 3)
    assert not _edge_intersects(xyz, 0, 1, 2, 4)
    assert not _edge_intersects(xyz, 0, 1, 0, 3)


def test_center_sphere_preserves_radius():
    vertices = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0],
                         [0, -2, 0], [0, 0, 2], [0, 0, -2]], np.float32)
    centered, iterations = center_sphere(vertices)
    assert 1 <= iterations <= 100
    np.testing.assert_allclose(np.linalg.norm(centered, axis=1), 100, atol=1e-5)


def test_genetic_base_surface_keeps_source_order_outside_defect():
    faces = np.array([[0, 1, 2], [2, 3, 4], [0, 3, 4]], np.int32)
    labels = np.array([0, 1, 1, 0, 0], np.int32)
    vertex, face = genetic_base_translation(labels, faces)
    np.testing.assert_array_equal(vertex, [0, 3, 4, 1, 2])
    np.testing.assert_array_equal(face, [-1, -1, 0])


def test_sphere_projection_matches_installed_freesurfer_float_bits():
    # Native 8.2 first-pass averages and projected xyz at LH v2887/RH v2198.
    cases = (
        ((0x4237A10D, 0xC2A3FE91, 0xC208C2DF),
         (0x4237A10F, 0xC2A3FE93, 0xC208C2E1)),
        ((0xC16C42FB, 0xC2B18401, 0xC22E7EA1),
         (0xC16C45F2, 0xC2B1863B, 0xC22E80D2)),
    )
    for averaged, expected in cases:
        xyz = np.array(averaged, np.uint32).view(np.float32)
        actual = np.array(_project_sphere_point(*xyz), np.float32)
        np.testing.assert_array_equal(actual.view(np.uint32), expected)
