"""Command line interface for the supported PyTorch FLIRT path."""

import argparse
import sys

from .standalone import run_flirt


def build_parser(prog="fs-torch-flirt"):
    parser = argparse.ArgumentParser(
        prog=prog,
        description=(
            "Run the source-derived PyTorch implementation of FLIRT's supported "
            "default 12-DOF correlation-ratio registration path. Other FLIRT "
            "optimizers, costs, and degrees of freedom are rejected."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("-in", "--in", dest="input", required=True,
                        help="input/moving 3D image")
    parser.add_argument("-ref", "--ref", dest="reference", required=True,
                        help="reference/fixed image defining the output grid")
    parser.add_argument(
        "-out", "--out", dest="output",
        help="warped input on the reference grid",
    )
    parser.add_argument(
        "-omat", "--omat",
        help="input-to-reference 4x4 matrix in FSL scaled-mm coordinates",
    )
    parser.add_argument(
        "-init", "--init",
        help="initial input-to-reference FSL scaled-mm matrix",
    )
    parser.add_argument("-dof", type=int, choices=(12,), default=12,
                        help="affine degrees of freedom; only 12 is implemented")
    parser.add_argument(
        "-cost", choices=("corratio",), default="corratio",
        help="cost function; only corratio is implemented",
    )
    parser.add_argument(
        "--device", default=None,
        help="PyTorch device, for example cpu, cuda, or cuda:1; default: CUDA when available",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace existing output files; outputs are otherwise protected",
    )
    return parser


def main(argv=None, *, prog="fs-torch-flirt"):
    parser = build_parser(prog=prog)
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        run_flirt(
            args.input,
            args.reference,
            output=args.output,
            omat=args.omat,
            init=args.init,
            dof=args.dof,
            cost=args.cost,
            device=args.device,
            overwrite=args.overwrite,
        )
    except (FileExistsError, FileNotFoundError, NotImplementedError, TypeError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
