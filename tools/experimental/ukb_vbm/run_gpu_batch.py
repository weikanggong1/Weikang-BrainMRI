"""Run experimental GPU GM registration on selected private T1 cases.

FAST-derived GM and optional raw-T1 GPU GM can be registered to the UKB
template; raw-T1 GPU GM can also be registered to a local HCP-derived
comparison template. Output names follow evaluate.py. gpu_register.py is an
alternative to FNIRT.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


OUTPUTS = {
    "warped_gm": "T1_GM_to_template_GM.nii.gz",
    "jacobian": "T1_GM_JAC_nl.nii.gz",
    "modulated_gm": "T1_GM_to_template_GM_mod.nii.gz",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subjects-root", type=Path, required=True)
    parser.add_argument("--ukb-template", type=Path,
                        help="required for gpu_ukb and gpu_raw_ukb")
    parser.add_argument("--hcp-template", type=Path,
                        help="also run gpu_raw_hcp; use --arms to select only that arm")
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", choices=("gpu_ukb", "gpu_raw_ukb", "gpu_raw_hcp"),
                        help="explicit arm selection; supersedes the legacy raw-GM flags")
    parser.add_argument("--include-gpu-raw", action="store_true")
    parser.add_argument("--only-gpu-raw", action="store_true",
                        help="register raw-T1 GPU GM without requiring FAST outputs")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--affine-steps", type=int, default=50)
    parser.add_argument("--deform-steps", type=int, default=40)
    parser.add_argument("--smoothness", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.hcp_template is not None and not args.hcp_template.is_file():
        parser.error(f"local HCP-derived template does not exist: {args.hcp_template}")
    if args.arms and (args.include_gpu_raw or args.only_gpu_raw):
        parser.error("--arms cannot be combined with --include-gpu-raw or --only-gpu-raw")
    if args.arms:
        selected = args.arms
    else:
        selected = [] if args.only_gpu_raw else ["gpu_ukb"]
        if args.include_gpu_raw or args.only_gpu_raw:
            selected.append("gpu_raw_ukb")
        if args.hcp_template is not None:
            selected.append("gpu_raw_hcp")
    if len(set(selected)) != len(selected):
        parser.error("duplicate arms")
    if any(arm != "gpu_raw_hcp" for arm in selected) and (
            args.ukb_template is None or not args.ukb_template.is_file()):
        parser.error(f"UKB template does not exist: {args.ukb_template}")
    if "gpu_raw_hcp" in selected and args.hcp_template is None:
        parser.error("gpu_raw_hcp requires --hcp-template")
    if len(set(args.cases)) != len(args.cases):
        parser.error("--cases contains duplicate IDs")
    known_cases = {item["case_id"] for item in json.loads(args.manifest.read_text())["cases"]}
    missing_cases = sorted(set(args.cases) - known_cases)
    if missing_cases:
        parser.error(f"case IDs absent from manifest: {', '.join(missing_cases)}")

    plans = []
    for case_id in args.cases:
        subject = args.subjects_root / case_id
        t1 = subject / "T1"
        for arm in selected:
            gm = (t1 / "T1_fast" / "T1_brain_pve_1.nii.gz" if arm == "gpu_ukb"
                  else t1 / "T1_gpu_raw" / "GM_prob.nii.gz")
            template = args.hcp_template if arm == "gpu_raw_hcp" else args.ukb_template
            if not gm.is_file():
                raise FileNotFoundError(f"{case_id}/{arm}: missing input GM {gm}")
            output = t1 / "T1_vbm" / arm
            existing = [output / name for name in OUTPUTS.values() if (output / name).exists()]
            if existing and not args.overwrite:
                raise FileExistsError(f"{case_id}/{arm}: output exists: {existing[0]}")
            plans.append((case_id, arm, gm, template, output))

    script = Path(__file__).with_name("gpu_register.py")
    for case_id, arm, gm, template, output in plans:
        output.mkdir(parents=True, exist_ok=True)
        prefix = output / "_gpu_register"
        command = [sys.executable, str(script), "--moving", str(gm),
                   "--fixed", str(template), "--output-prefix", str(prefix),
                   "--device", args.device, "--affine-steps", str(args.affine_steps),
                   "--deform-steps", str(args.deform_steps),
                   "--smoothness", str(args.smoothness)]
        started = time.perf_counter()
        with (output / "gpu_register.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        registration_sec = time.perf_counter() - started
        report_source = Path(str(prefix) + "_report.json")
        report = json.loads(report_source.read_text())
        for field, filename in OUTPUTS.items():
            source = Path(str(prefix) + f"_{field}.nii.gz")
            os.replace(source, output / filename)
        report["outputs"] = {field: str(output / filename)
                             for field, filename in OUTPUTS.items()}
        (output / "gpu_register.report.private.json").write_text(
            json.dumps(report, indent=2) + "\n")
        report_source.unlink()
        total_sec = time.perf_counter() - started
        timing = {
            "case_id": case_id,
            "arm": arm,
            "device": args.device,
            "registration_sec": registration_sec,
            "modulation_sec": None,
            "total_sec": total_sec,
            "timing_scope": "registration_sec includes registration, Jacobian, modulation and NIfTI writing",
            "gpu_register_internal_sec": report["seconds_including_io"],
            "nonpositive_jacobian_voxels": report["nonpositive_jacobian_voxels"],
        }
        if arm.startswith("gpu_raw_"):
            subject_timing = args.subjects_root / case_id / "gpu_gm_timing.private.json"
            if subject_timing.is_file():
                raw_seconds = json.loads(subject_timing.read_text()).get("gpu_gm_raw_seconds")
                timing["upstream_gm_sec"] = raw_seconds
                if isinstance(raw_seconds, (int, float)):
                    timing["total_from_raw_sec"] = raw_seconds + total_sec
        (output / "timings.private.json").write_text(json.dumps(timing, indent=2) + "\n")
        print(f"{case_id} {arm} registration_sec={registration_sec:.3f} "
              f"total_sec={total_sec:.3f}", flush=True)


if __name__ == "__main__":
    main()
