"""Run the public UKB v1.5 T1-to-VBM steps on a private T1 manifest.

The original UKB gradient-unwarp step is skipped with coeff=none because the
non-UKB scans do not have the scanner gradient-coefficient file. This script
stops after FAST and VBM; FIRST, SIENAX, defacing, and unrelated QC are omitted.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def call(name, cmd, cwd, log_dir, timings):
    started = time.monotonic()
    with (log_dir / f"{name}.log").open("w") as log:
        subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)
    timings[name] = time.monotonic() - started


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--assets", required=True)
    parser.add_argument("--hcp-template",
                        help="local HCP-derived comparison GM template")
    parser.add_argument("--arms", nargs="+", choices=("ukb", "hcp"),
                        default=("ukb", "hcp"),
                        help="template registrations to run after FAST (default: both)")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--header-transform", action="store_true",
                        help="derive cropped-to-original transform from headers when xyztrans fails")
    parser.add_argument("--resume-after-standard-roi", action="store_true")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms contains duplicates")
    if not args.preprocess_only and "hcp" in args.arms and not args.hcp_template:
        parser.error("--hcp-template is required when --arms includes hcp")
    cases = {item["case_id"]: item for item in json.load(open(args.manifest))["cases"]}
    raw = Path(cases[args.case]["path"]).expanduser().resolve()
    if not raw.is_file():
        parser.error(f"input T1 does not exist: {raw}")
    subject = Path(args.output_root) / args.case
    t1 = subject / "T1"
    logs = subject / "logs"
    fast = t1 / "T1_fast"
    assets = Path(args.assets)
    for path in (t1, logs, fast):
        path.mkdir(parents=True, exist_ok=True)
    timings_path = subject / "timings.private.json"
    timings = json.load(timings_path.open()) if timings_path.exists() else {}
    fsl = Path(os.environ["FSLDIR"])
    bin_dir = fsl / "bin"

    def exe(name):
        return str(bin_dir / name)

    if not args.register_only:
        source = t1 / "T1_orig.nii.gz"
        if not source.exists():
            source.symlink_to(raw)
        ud = t1 / "T1_orig_ud.nii.gz"
        if not ud.exists():
            ud.symlink_to(raw)
        if not args.resume_after_standard_roi:
            started = time.monotonic()
            robust = subprocess.check_output([exe("robustfov"), "-i", "T1_orig_ud"],
                                             cwd=t1, text=True)
            (logs / "robustfov.log").write_text(robust)
            head_top = int(float(next(line.split()[4] for line in robust.splitlines()
                                      if line.strip() and "Final" not in line)))
            timings["robustfov"] = time.monotonic() - started
            call("crop", [exe("fslmaths"), "T1_orig_ud", "-roi", "0", "-1", "0", "-1",
                          str(head_top), "170", "0", "1", "T1_tmp"], t1, logs, timings)
            call("bet", [exe("bet"), "T1_tmp", "T1_tmp_brain", "-R"], t1, logs, timings)
            call("standard_space_roi", [exe("standard_space_roi"), "T1_tmp_brain", "T1_tmp2",
                                        "-maskNONE", "-ssref", str(fsl / "data/standard/MNI152_T1_1mm_brain"),
                                        "-altinput", "T1_orig_ud", "-d"], t1, logs, timings)
            shutil.move(t1 / "T1_tmp2.nii.gz", t1 / "T1.nii.gz")
        if args.header_transform:
            from geometry_fallback import voxel_to_fsl
            import nibabel as nib
            import numpy as np
            existing = t1 / "T1_to_T1_orig_ud.mat"
            if existing.exists():
                shutil.copyfile(existing, t1 / "T1_to_T1_orig_ud.ukb_xyztrans.mat")
            cropped, original = nib.load(t1 / "T1.nii.gz"), nib.load(ud)
            matrix = (voxel_to_fsl(original) @ np.linalg.inv(original.affine)
                      @ cropped.affine @ np.linalg.inv(voxel_to_fsl(cropped)))
            np.savetxt(existing, matrix, fmt="%.12g")
            timings["header_transform_used"] = True
        else:
            call("flirt_xyztrans", [exe("flirt"), "-in", "T1", "-ref", "T1_orig_ud",
                                   "-omat", "T1_to_T1_orig_ud.mat", "-schedule",
                                   str(fsl / "etc/flirtsch/xyztrans.sch")], t1, logs, timings)
        call("xfm_inverse", [exe("convert_xfm"), "-omat", "T1_orig_ud_to_T1.mat",
                             "-inverse", "T1_to_T1_orig_ud.mat"], t1, logs, timings)
        call("xfm_concat", [exe("convert_xfm"), "-omat", "T1_to_MNI_linear.mat",
                            "-concat", "T1_tmp2_tmp_to_std.mat", "T1_to_T1_orig_ud.mat"],
             t1, logs, timings)
        call("fnirt_t1", [exe("fnirt"), "--in=T1", f"--ref={fsl}/data/standard/MNI152_T1_1mm",
                          "--aff=T1_to_MNI_linear.mat", f"--config={assets}/bb_fnirt.cnf",
                          f"--refmask={assets}/MNI152_T1_1mm_brain_mask_dil_GD7",
                          "--cout=T1_to_MNI_warp_coef", "--fout=T1_to_MNI_warp",
                          "--jout=T1_to_MNI_warp_jac", "--iout=T1_tmp4.nii.gz",
                          "--interp=spline"], t1, logs, timings)
        call("invwarp", [exe("invwarp"), "--ref=T1", "-w", "T1_to_MNI_warp_coef",
                         "-o", "T1_to_MNI_warp_coef_inv"], t1, logs, timings)
        call("mask_to_native", [exe("applywarp"), "--rel", "--interp=trilinear",
                               f"--in={assets}/MNI152_T1_1mm_brain_mask", "--ref=T1",
                               "-w", "T1_to_MNI_warp_coef_inv", "-o", "T1_brain_mask"],
             t1, logs, timings)
        call("brain", [exe("fslmaths"), "T1", "-mul", "T1_brain_mask", "T1_brain"],
             t1, logs, timings)
        call("fast", [exe("fast"), "-b", "-o", "T1_fast/T1_brain", "T1_brain"],
             t1, logs, timings)
        if not (fast / "T1_brain_pve_1.nii.gz").exists():
            raise RuntimeError("FAST did not produce grey-matter partial volumes")
        timings_path.write_text(json.dumps(timings, indent=2))

    if not args.preprocess_only:
        gm = fast / "T1_brain_pve_1.nii.gz"
        if not gm.exists():
            raise FileNotFoundError(gm)
        templates = {"ukb": assets / "template_GM.nii.gz"}
        if args.hcp_template:
            templates["hcp"] = Path(args.hcp_template)
        for arm in args.arms:
            template = templates[arm]
            arm_dir = t1 / "T1_vbm" / arm
            arm_dir.mkdir(parents=True, exist_ok=True)
            prefix = arm_dir / "T1_GM_to_template_GM"
            jac = arm_dir / "T1_GM_JAC_nl"
            call(f"fsl_reg_{arm}", [exe("fsl_reg"), str(gm), str(template), str(prefix),
                                   "-fnirt", f"--config=GM_2_MNI152GM_2mm.cnf --jout={jac} "
                                   f"--logout={logs}/bb_vbm_{arm}_fnirt.log"],
                 t1, logs, timings)
            call(f"modulate_{arm}", [exe("fslmaths"), str(prefix), "-mul", str(jac),
                                    str(arm_dir / "T1_GM_to_template_GM_mod"), "-odt", "float"],
                 t1, logs, timings)
            timings_path.write_text(json.dumps(timings, indent=2))
    print(json.dumps({"case_id": args.case, "timings_sec": timings}, indent=2))


if __name__ == "__main__":
    main()
