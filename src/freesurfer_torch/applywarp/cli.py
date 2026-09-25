"""Dedicated ``fs-torch-applywarp`` entry point."""

import sys


def main(argv=None):
    from ..cli import main as package_main

    arguments = sys.argv[1:] if argv is None else list(argv)
    return package_main(["applywarp", *arguments])


if __name__ == "__main__":
    main()
