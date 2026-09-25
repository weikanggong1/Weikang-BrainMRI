"""Convert and write the fixed FreeSurfer inverse-GCAM displacement warp."""

from __future__ import annotations

import gzip
import math
import struct
from pathlib import Path

import numpy as np

from .ca_register_inverse import _freesurfer_vox2ras


def inverse_coordinates_to_displacement_ras(
    inverse_coordinates: np.ndarray,
    atlas_vox2ras: np.ndarray,
    source_vox2ras: np.ndarray,
) -> np.ndarray:
    """Match FreeSurfer's float32 ABS_CRS to DISP_RAS NIfTI conversion."""
    fields = np.asarray(inverse_coordinates, dtype=np.float32)
    atlas = np.asarray(atlas_vox2ras, dtype=np.float32)
    source = np.asarray(source_vox2ras, dtype=np.float32)
    width, height, depth, channels = fields.shape
    if channels != 3 or atlas.shape != (4, 4) or source.shape != (4, 4):
        raise ValueError("expected three inverse coordinates and two 4x4 geometries")
    output = np.empty_like(fields)
    x = np.arange(width, dtype=np.float64)[:, None]
    y = np.arange(height, dtype=np.float64)[None, :]
    for z in range(depth):
        for axis in range(3):
            image_ras = np.zeros((width, height), dtype=np.float64)
            for column in range(3):
                image_ras += float(atlas[axis, column]) * fields[:, :, z, column].astype(np.float64)
            image_ras = np.float32(image_ras + float(atlas[axis, 3]))
            target_ras = np.float32(
                float(source[axis, 0]) * x
                + float(source[axis, 1]) * y
                + float(source[axis, 2]) * z
                + float(source[axis, 3])
            )
            output[:, :, z, axis] = np.float32(
                image_ras.astype(np.float64) - target_ras.astype(np.float64)
            )
    return output


def _geometry_parts(payload: bytes, shearless: bool) -> tuple[bytes, bytes]:
    fixed_size = 76 if shearless else 88
    first_length = struct.unpack_from(">i", payload, fixed_size)[0]
    split = fixed_size + 4 + first_length
    second_length = struct.unpack_from(">i", payload, split + fixed_size)[0]
    if split + fixed_size + 4 + second_length != len(payload):
        raise ValueError("invalid FreeSurfer warp geometry tag")
    return payload[:split], payload[split:]


def _native_quaternion(source: tuple[int | float, ...]) -> tuple[np.float32, ...]:
    """Use the float32 arithmetic of mriToNiftiQform for this geometry."""
    f = np.float32
    r11, r21, r31, r12, r22, r32, r13, r23, r33 = map(f, source[7:16])
    determinant = f(
        f(r11 * f(r22 * r33 - r32 * r23))
        - f(r12 * f(r21 * r33 - r31 * r23))
        + f(r13 * f(r21 * r32 - r31 * r22))
    )
    if determinant >= 0:
        raise ValueError("fixed inverse warp expects negative orientation determinant")
    r13, r23, r33 = -r13, -r23, -r33
    yd = f(f(f(1) + r22) - f(r11 + r33))
    if yd <= 1:
        raise ValueError("fixed inverse warp expects the y quaternion branch")
    c = f(f(0.5) * f(math.sqrt(float(yd))))
    b = f(f(f(0.25) * f(r12 + r21)) / c)
    d = f(f(f(0.25) * f(r23 + r32)) / c)
    a = f(f(f(0.25) * f(r13 - r31)) / c)
    if a < 0:
        return -b, -c, -d
    return b, c, d


def _inverse_nifti_header_and_extension(input_warp: Path) -> tuple[bytes, bytes, tuple[int, int, int]]:
    """Swap the native FreeSurfer geometry tags and expand zero inverse labels."""
    with gzip.open(input_warp, "rb") as stream:
        header = bytearray(stream.read(348))
        flag = stream.read(4)
        offset = int(struct.unpack_from("<f", header, 108)[0])
        extension = stream.read(offset - 352)
    if flag != b"\x01\x00\x00\x00" or len(extension) < 12:
        raise ValueError("expected FreeSurfer NIfTI extension")
    size, code = struct.unpack_from("<ii", extension)
    if size != len(extension) or code != 14 or extension[8:12] != b">\x00\x03\x01":
        raise ValueError("unsupported fixed FreeSurfer warp extension")

    cursor = 12
    tags = []
    source = None
    while cursor + 12 <= len(extension):
        tag, length = struct.unpack_from(">iq", extension, cursor)
        end = cursor + 12 + length
        if length < 0 or end > len(extension):
            raise ValueError("invalid FreeSurfer warp tag length")
        data = extension[cursor + 12 : end]
        if tag in (10, 15):
            first, second = _geometry_parts(data, shearless=(tag == 10))
            if tag == 15:
                source = struct.unpack_from(">4i18f", first)
            data = second + first
        elif tag == 12:
            if source is None:
                raise ValueError("warp labels precede source geometry")
            data = bytes(4 * math.prod(source[1:4]))
        tags.append(struct.pack(">iq", tag, len(data)) + data)
        cursor = end
        if tag == -1:
            break
    if source is None or tuple(source[1:4]) != (256, 256, 256):
        raise ValueError("unexpected source grid for fixed inverse warp")
    payload = extension[8:12] + b"".join(tags)
    extension_size = (8 + len(payload) + 15) & ~15
    output_extension = struct.pack("<ii", extension_size, 14) + payload
    output_extension += bytes(extension_size - len(output_extension))

    shape = tuple(source[1:4])
    struct.pack_into("<3h", header, 42, *shape)
    struct.pack_into("<f", header, 76, -1.0)
    struct.pack_into("<f", header, 108, float(352 + extension_size))
    struct.pack_into("<3f", header, 256, *_native_quaternion(source))
    vox2ras = _freesurfer_vox2ras(source)
    struct.pack_into("<3f", header, 268, *vox2ras[:3, 3])
    struct.pack_into("<12f", header, 280, *vox2ras[:3, :].flat)
    return bytes(header), output_extension, shape


def write_inverse_warp_nifti(
    input_warp: Path, displacement_ras: np.ndarray, output_path: Path
) -> None:
    """Write the fixed ``mri_ca_register -invert-and-save`` NIfTI result."""
    header, extension, shape = _inverse_nifti_header_and_extension(input_warp)
    values = np.asarray(displacement_ras, dtype=np.float32)
    if values.shape != (*shape, 3):
        raise ValueError(f"expected inverse field {(*shape, 3)}, got {values.shape}")
    with output_path.open("wb+") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as stream:
            stream.write(header)
            stream.write(b"\x01\x00\x00\x00")
            stream.write(extension)
            for frame in range(3):
                for z in range(shape[2]):
                    stream.write(values[:, :, z, frame].tobytes(order="F"))
        raw.seek(9)
        raw.write(b"\x03")  # zlib's Unix OS byte in the pinned native gzip stream
