# Native runtime packaging tools

These tools prepare and inspect a runtime candidate for the pinned FreeSurfer
8.2 single-T1 workflow. They use Python 3.10+ and standard-library modules.
The audit host also needs `ldd` and `readelf`.

The candidate contains **official prebuilt native programs**, not programs
rebuilt from source. Its manifest always starts with `standalone_verified=false`.
Packaging, a successful library check, or a development overlay does not prove
independent reconstruction.

## Build a candidate from an existing reference installation

Run on the machine holding the trusted FreeSurfer installation and completed
reference log. The output directory must not already exist.

```bash
python tools/recon_all_native/package_candidate.py \
  --fs-home /path/to/freesurfer-8.2 \
  --log /path/to/subject/scripts/recon-all.log \
  --output work/native_candidate \
  --tcsh /usr/bin/tcsh \
  --tcsh-notice /path/to/matching-tcsh-Copyright \
  --plan-only
```

This writes `work/native_candidate.plan.json` and prints the expected payload
size from actual file sizes, including non-system libraries and the interpreter.
It does not copy a bundle. Small generated scripts and metadata add overhead.
Remove `--plan-only` to create the candidate.

The procedure:

- Extracts command/resource candidates from the real log and includes required
  startup configuration and neural weights.
- Includes the `lib/bem` icosahedron and skull templates. The native
  `mris_ca_label` program opens `lib/bem/ic4.tri` while reading the cortical
  atlas; this constructed path was absent from the shell-level resource scan.
- Includes every program in recon-all's `allinfocmds` list. The unmodified
  driver invokes their `-all-info` entry points even when their processing
  branches are disabled; log-only extraction misses these dependencies.
- Follows literal script resource paths, including installed Bash wrappers
  pointing into `python/scripts`. Constructed paths remain review items.
- Copies native programs, scripts, resources and resolved non-system libraries.
  Hashes and original paths remain in `metadata/source_inventory.json`.
- Installs three explicit PyTorch launchers: `mri_synthstrip`, `mri_synthseg`,
  and `mri_synthmorph`. Each runs
  `"$FS_TORCH_PYTHON" -m freesurfer_torch.recon_all.gpu_tools TOOL "$@"`.
  Replacement and original hashes are recorded in `neural_replacements`.
- With `--tcsh`, bundles that interpreter and its libraries. Executable tcsh
  and csh scripts are preserved verbatim under `upstream_scripts/`, while launchers
  dispatch them through the package-owned interpreter. Configuration helpers
  preserve their official Python code and dispatch it through the package
  Python in isolated mode. They require PyYAML in that Python environment.
- Dispatches `csvprint` and the private `fspython` entry point through the
  package Python. The original scripts remain in the provenance archive.
  Imports in Python bodies called through `fspython` remain closure checks;
  changing its interpreter does not satisfy missing TensorFlow dependencies.
- Preserves the MGH software license notice and an explicitly supplied tcsh
  copyright notice. Missing third-party notices are not treated as reviewed.
- Excludes personal `.license`, `license.txt`, the file named by `FS_LICENSE`,
  and aliases resolving to those files. No runtime key is embedded or bypassed.

Use repeated `--extra-file RELATIVE_PATH` for a reviewed missing file, or
`--include-tree RELATIVE_DIRECTORY` for a reviewed resource/Python subtree.
These paths must stay inside the reference installation. Do not copy its full
private Python environment and assume the dependency problem is solved: its
interpreter, imports, shared libraries, resources and licenses need verification.

## Check the candidate without the original installation

```bash
python tools/recon_all_native/preflight_bundle.py \
  --bundle work/native_candidate \
  --package-python /path/to/installed-package/python \
  --output work/native_candidate/metadata/preflight.json
```

The preflight verifies staged hashes and ELF resolution under a constrained
environment with the bundle's library path. It also checks required resources,
script interpreters/dispatchers, Python import availability, literal command
references and literal resource references. Missing dependencies cause exit 2.
Constructed paths are listed for an execution trace; shell branch reachability
is not inferred, so a missing optional branch requires explicit review.

`script_resource_checks_passed` is a static check, not a complete proof.
`standalone_verified` remains false even if all preflight checks pass.

The scanner recognizes the FreeSurfer ASCII label format using its header,
declared point count and numeric rows. For upstream script help appended after
`BEGINHELP`, it requires an unconditional preceding exit and the `cat $0 | awk`
help reader before excluding the prose. Executed files under `upstream_scripts/`
are still scanned; only explicitly archived, replaced implementations are skipped.

The candidate targets Linux. The exact Darwin guard in the installed
`sources.csh` can establish that `SetUpFreeSurfer.csh` and `FreeSurferEnv.csh`
are inactive on Linux, provided the other candidate scripts do not source them.
Their hashes and that source evidence remain in the manifest. Preflight rejects
another operating system; this classification is not a portable-runtime claim.

### Repair the first candidate without recopying native programs

For candidates made before the `csh`, `csvprint`, and helper closure fixes:

```bash
python tools/recon_all_native/repair_candidate.py \
  --bundle work/native_candidate \
  --fs-home /path/to/freesurfer-8.2 \
  --package-python /path/to/installed-package/python
```

This verifies existing hashes and preserves a manifest snapshot, audits and adds
`fs_temp_file`, `fs_temp_dir`, `fs-check-os`, `reconbatchjobs`, and missing `allinfocmds` programs,
then updates interpreter
dispatch and reruns script checks. ELF helpers include their audited libraries;
existing library hashes and SONAME aliases must agree before the merge. It only
accepts an unverified candidate whose build-stamp hash matches the source
installation. Add `--plan-only` first to inspect the extra payload size.
Run the regular preflight again afterward. Unresolved imports and conditional
command/resource references continue to fail the static check.
The [conditional branch notes](conditional_branches.md) document source evidence
for selected optional processing paths and the limits of using that evidence.

### Restrict the candidate to the reviewed single-T1 route

For the exact FS8.2 build reviewed here, both package and repair accept
`--single-t1-reference-config` pointing to the **resolved** reference subject's
`scripts/recon-config.yaml`. The source config, V8 expert options, configuration
resolver, upstream scripts and resolved reference YAML must match pinned SHA256
values. A different configuration fails this audit.

```bash
python tools/recon_all_native/repair_candidate.py \
  --bundle work/native_candidate \
  --fs-home /path/to/freesurfer-8.2 \
  --package-python /path/to/installed-package/python \
  --single-t1-reference-config /path/to/reference-subject/scripts/recon-config.yaml
```

This writes `runtime_profile` and `inactive_command_audit` into the manifest.
Profile `fs820-single-t1-v8-all-v1` requires Linux, one T1, a fresh unedited
subject in an empty subjects directory, a clean environment, and exactly
`-all -parallel -openmp 4 -itkthreads 1`. The package runner must enforce this
profile and reject additional flags. Calling `bundle/bin/recon-all` with
arbitrary options is outside the audited interface.

`single_t1_scope.py` records each of 16 reviewed `(script, command)` pairs,
source commit/line evidence, applicable conditions and relevant resolved
configuration values. Preflight checks that the audit itself and every pinned
file are unchanged before excluding those pairs. The same command in another
script remains a dependency, and unconditional `allinfocmds` queries remain
required. The copied resolved YAML is an inventoried evidence file; it does not
replace the runtime configuration resolver.

The two GUI entry points, `tkmeditfv` and `tkregisterfv`, become explicit
rejections with exit status 64 in this profile. Their original scripts and
dispatchers remain archived and hashed. The manifest records that the reviewed
pipeline only prints these commands as optional QC instructions. If execution
does invoke a GUI entry, the run fails visibly. The absent upstream `isargflag`
is therefore not invented or fetched from an external installation.

Rerun preflight after attaching the profile. Passing this scoped static audit
still leaves `standalone_verified=false`; a complete isolated reconstruction
and scientific comparison are required.

The remaining auxiliary SCLimbic programs and their models must be included or
replaced with independently validated package implementations. The three Synth
launchers alone do not replace every neural stage in the reference workflow.

`package_candidate.py --replace-entowm` additionally replaces `mri_entowm_seg`
with `python -m freesurfer_torch.recon_all.sclimbic`, passing the bundled EntoWM
model/ctab directory and selected GPU. Subject arguments pass through unchanged.
The original script is preserved and both hashes are recorded. This opt-in is
disabled by default; enable it only after validating the CLI in the intended
pipeline. It does not replace MCA/dura or venous-sinus segmentation.

`--replace-mcadura` and `--replace-vsinus` are separate opt-ins supported by both
the packager and repair tool. They invoke the corresponding `aux_seg` CLI with
the bundle root as its asset directory, retaining upstream scripts and hashes.
Required resources are the two model H5 files, both hemispheres of the MCA/dura
MNI152 prior and the venous-sinus MNI152 prior; each enters `required_resources`.
The original TensorFlow SCLimbic scripts are classified as inactive only after
all three auxiliary callers have replacements and no active bundle script
references the generic helper. The manifest records that evidence. Any remaining
caller keeps the original Python imports in the closure check.

## Individual audit and source-build tools

- `extract_reference.py`: offline command/resource extraction. FSTIME rows are
  distinguished from command-like log mentions; nested timings must not be
  added twice when reporting total wall time.
- `audit_bundle.py`: read-only executable/resource hashes and ELF inventory;
  optional `--stage` creates an unverified copy. `--include-scripts` is explicit.
  `--trace` accepts available absolute-path file/exec traces. Relative paths
  require cwd reconstruction and are not resolved automatically.
- `build_source.py`: checks pinned commit
  `d932c45b7941662ea380a05efef580568b98d41a`, source completeness and basic tools;
  records the exact official CMake commands. Only `--execute` configures/builds.
  It does not install dependencies or alter official source. `MINIMAL=ON`
  does not remove the large `utils`/ITK/image-library build dependency graph.

```bash
python tools/recon_all_native/build_source.py \
  --source /path/to/complete/freesurfer-source --work work/native_source_build
```

No complete source-built reconstruction runtime has been validated by these
tools. The source path needs the full required checkout, compatible CMake,
compiler/Fortran tools, ITK and image-library development dependencies.

## Release gate

After producing the numerical comparator JSON, check the documented additional
aggregate criteria without rereading the images:

```bash
python tools/recon_all_native/check_comparison_gate.py work/comparison.numeric.json \
  --output work/comparison.aggregate-gates.json
```

This requires the comparator itself to pass, white/pial mean displacement
<=0.01 mm and P99 <=0.05 mm, thickness MAE <=0.005 mm, annotation agreement
>=0.999 for both hemispheres and all three atlases, and foreground macro Dice
>=0.999 for aseg/aparc+aseg. Background label 0 is excluded from the macro mean.
Missing, nonfinite or out-of-range metrics fail. The output records each check
and the input JSON's SHA256; exit 0 means all 19 checks pass, exit 1 means a
failure. These remain provisional development criteria, not frozen validation
limits or a standalone-runtime certificate.

Run the complete T1-to-surface/statistics pipeline in a supported clean Linux
environment where the reference neuroimaging installation is unavailable.
Record subprocess/file accesses and demonstrate that all non-system code and
resources come from the package or declared cache. Check the native runtime
authorization mechanism without distributing personal keys. Compare anatomy
and final regional metrics with the pinned reference. Review all copied code,
library and resource notices before redistribution. Only a separately recorded
successful independent run can justify updating the verification state.

### Record and promote the tested candidate

Before starting a fresh clean run, freeze the exact installed package code,
Python/dependency versions, CUDA/cuDNN versions and candidate manifest:

```bash
python tools/recon_all_native/promote_bundle.py freeze-code \
  --bundle work/native_candidate --package-python .venv/bin/python \
  --output work/validation/code_manifest.json
```

The snapshot file must be new. Do not change the package or candidate after
this snapshot. A run made before the snapshot cannot certify the snapshot's
code. Preserve the original `env -i ... strace -f -e trace=file,process ...`
launch command in a text file and keep the raw trace. A shell-history recovery
is accepted as an operator-supplied record, not described as a digital signature.

After the complete run and both comparison stages, check all linked evidence:

```bash
python tools/recon_all_native/promote_bundle.py \
  --bundle work/native_candidate \
  --preflight work/native_candidate/metadata/preflight.json \
  --run work/subjects/sub01.recon-all.run.json \
  --comparison work/comparison.numeric.json \
  --aggregate work/comparison.aggregate-gates.json \
  --trace work/validation/trace.log \
  --trace-command-file work/validation/command.txt \
  --trace-cwd "$PWD" --package-python .venv/bin/python \
  --input examples/data/sub-01_T1w.nii.gz --license-file /path/to/personal/license.txt \
  --code-manifest work/validation/code_manifest.json --check-only
```

The tool verifies source/configuration pins and bundle hashes, reruns static
preflight, requires the same input/subject/configuration and successful run,
checks all 52 comparator entries at the declared numerical tolerances, and
recomputes all 19 aggregate gates. It rejects a stale aggregate report, changed
outputs, package code or software versions, and a trace from another subject.

The trace audit checks successful `open/openat/openat2/creat/execve/execveat`
path arguments for the source FreeSurfer installation and visible
FreeSurfer/FSL/TensorFlow path components, with only the exact supplied personal
license path excepted. It pairs unfinished/resumed records and rejects unmatched
or truncated records. It does **not** reconstruct arbitrary relative cwd/dirfd
targets or establish closure of every other external resource. This boundary is
recorded in the manifest; it must not be presented as a universal path audit.

Remove `--check-only` to atomically set `standalone_verified=true` after every
gate passes. The original manifest and fresh preflight are retained in
`metadata/`; `verification` records report/trace/launch/snapshot hashes, frozen
software versions and output hashes. Failure exits 2 without changing the
manifest. Promotion does not set `redistribution_review_complete`, approve
binary redistribution, or establish bitwise numerical identity.
