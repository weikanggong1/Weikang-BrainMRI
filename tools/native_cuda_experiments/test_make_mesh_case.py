import struct
import tempfile
import unittest
from pathlib import Path

from make_mesh_case import make_case, ordered_neighbors


class MeshCaseTest(unittest.TestCase):
    def test_original_face_order_and_zero_iterations(self):
        faces = [(0, 1, 2), (0, 2, 3)]
        self.assertEqual(ordered_neighbors(4, faces), [[2, 1, 3], [0, 2], [1, 0, 3], [2, 0]])
        with tempfile.TemporaryDirectory() as directory:
            surface = Path(directory) / "lh.inflated"
            output = Path(directory) / "case.bin"
            surface.write_bytes(
                (16777214).to_bytes(3, "big") + b"created by test\n\n"
                + struct.pack(">ii", 4, 2)
                + struct.pack(">12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, -1, 0, 0)
                + b"".join(struct.pack(">iii", *face) for face in faces)
            )
            make_case(surface, output, 0)
            data = output.read_bytes()
            self.assertEqual(data[:8], b"FSGRAD1\0")
            self.assertEqual(struct.unpack_from("<IIII", data, 8), (4, 10, 0, 4))
            self.assertEqual(struct.unpack_from("<5I", data, 40), (0, 3, 5, 8, 10))

    def test_invalid_face_and_iteration(self):
        with self.assertRaises(ValueError):
            ordered_neighbors(3, [(0, 1, 3)])
        with self.assertRaises(ValueError):
            make_case(Path("unused"), Path("unused"), -1)


if __name__ == "__main__":
    unittest.main()
