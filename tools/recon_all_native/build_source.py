#!/usr/bin/env python3
"""Preflight and explicitly run the unmodified FS8.2 CMake build.

Does not install dependencies, change source files or copy installed binaries.
Only --execute starts configuration/compilation. The default performs preflight.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess


PIN = "d932c45b7941662ea380a05efef580568b98d41a"
DEFAULT_TARGETS = ["mri_convert", "mri_normalize", "mri_segment", "mri_fill",
                   "mri_tessellate", "mris_fix_topology", "mris_make_surfaces",
                   "mris_place_surface", "mris_sphere", "mris_register",
                   "mris_ca_label", "mris_thickness", "mris_anatomical_stats",
                   "AntsN4BiasFieldCorrectionFs"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--cmake", default="cmake")
    parser.add_argument("--target", action="append")
    parser.add_argument("--cmake-option", action="append", default=[])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    targets = args.target or DEFAULT_TARGETS
    directories = ["cmake", "packages", "distribution", "utils", "include",
                   "vtkutils", "fem_elastic", "itkutils", "itkio", "resurf"]
    missing_source = [name for name in directories if not (source / name).is_dir()]
    for target in targets:
        directory = "mris_make_surfaces" if target == "mris_place_surface" else target
        if not (source / directory / "CMakeLists.txt").is_file():
            missing_source.append(directory + "/CMakeLists.txt")
    git = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                         capture_output=True, text=True)
    commit = git.stdout.strip() if git.returncode == 0 else None
    compiler = {name: shutil.which(name) for name in (args.cmake, "gcc", "g++", "gfortran", "make")}
    build = work / "build"
    prefix = work / "install"
    configure = [args.cmake, "-S", str(source), "-B", str(build),
                 f"-DCMAKE_INSTALL_PREFIX={prefix}", "-DCMAKE_BUILD_TYPE=Release",
                 "-DMINIMAL=ON", "-DBUILD_GUIS=OFF", "-DBUILD_ATTIC=OFF",
                 "-DQATOOLS_MODULE=OFF", "-DINFANT_MODULE=OFF", "-DBUILD_TESTING=OFF",
                 "-DDISTRIBUTE_FSPYTHON=OFF", "-DINSTALL_PYTHON_DEPENDENCIES=OFF",
                 "-DDISABLE_LINEPROF=ON", *args.cmake_option]
    compile_command = [args.cmake, "--build", str(build), "--parallel", str(args.jobs),
                       "--target", *targets]
    report = {"source_commit": commit, "expected_commit": PIN,
              "missing_source": sorted(set(missing_source)), "tools": compiler,
              "configure_command": configure, "compile_command": compile_command,
              "source_built": False, "standalone_verified": False,
              "required_external_build_packages": [
                  "ITK development files including VNL; version must match reference build",
                  "zlib and OpenSSL development files",
                  "BLAS/LAPACK, Fortran runtime and compiler required by top-level project",
                  "Other packages selected by the actual official CMake configuration",
              ],
              "notes": [
                  "MINIMAL does not eliminate the utils static library dependency graph.",
                  "No full-suite install target is run; requested executables remain in the build tree.",
                  "Successful compilation does not establish redistributability, numerical equivalence or runtime closure.",
              ]}
    if args.execute:
        if commit != PIN or missing_source or any(value is None for value in compiler.values()):
            report["blocked"] = "Pinned complete source and all basic build tools are required"
        else:
            for name, command in (("configure", configure), ("compile", compile_command)):
                with (work / f"{name}.log").open("w") as output:
                    result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
                report[name + "_exit_code"] = result.returncode
                if result.returncode:
                    report["blocked"] = f"Inspect {work / (name + '.log')}"
                    break
            else:
                report["source_built"] = True
    (work / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.execute and not report["source_built"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
