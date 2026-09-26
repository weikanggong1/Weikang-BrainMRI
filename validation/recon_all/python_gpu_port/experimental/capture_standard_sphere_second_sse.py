"""Capture a bounded number of native mris_sphere line searches with SSE terms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("inflated", type=Path)
    parser.add_argument("smoothwm", type=Path)
    parser.add_argument("license", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--decisions", type=int, default=2)
    args = parser.parse_args()
    surf = args.output / "surf"
    surf.mkdir(parents=True, exist_ok=True)
    hemisphere = args.inflated.name.split(".")[0]
    shutil.copyfile(args.inflated, surf / f"{hemisphere}.inflated")
    shutil.copyfile(args.smoothwm, surf / f"{hemisphere}.smoothwm")
    environment = os.environ.copy()
    environment.update(FREESURFER_HOME=str(args.bundle), FS_LICENSE=str(args.license),
                       DIAG="0x10000040", DIAG_VERBOSE="1", FREESURFER_logSSE="1")
    command = ["stdbuf", "-oL", "-eL", str(args.bundle / "bin/mris_sphere"),
               "-threads", "4", "-seed", "1234", f"surf/{hemisphere}.inflated",
               f"{hemisphere}.sphere"]
    started = time.monotonic()
    decisions = 0
    with (args.output / "native_logsse.log").open("w") as log:
        process = subprocess.Popen(command, cwd=args.output, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True)
        try:
            for line in process.stdout:
                log.write(line)
                if line.startswith("sses:"):
                    decisions += 1
                    log.flush()
                    if decisions == args.decisions:
                        break
                if time.monotonic() - started > 90:
                    break
        finally:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)
    (args.output / "capture_summary.txt").write_text(
        f"line_search_decisions={decisions}\n"
        f"wall_seconds={time.monotonic() - started:.6f}\n")
    if decisions != args.decisions:
        raise RuntimeError("native run ended before the requested line search")


if __name__ == "__main__":
    main()
