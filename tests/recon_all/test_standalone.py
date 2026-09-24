"""Release gate and environment isolation for the standalone entry point."""

import hashlib
import json

import pytest

from freesurfer_torch.recon_all import standalone


REQUIRED = (
    "bin/recon-all", "bin/mri_synthstrip", "bin/mri_synthseg", "bin/mri_synthmorph",
    "build-stamp.txt", "models/synthseg_2.0.h5",
    "models/synthseg_segmentation_labels_2.0.npy",
    "models/synthseg_segmentation_names_2.0.npy",
    "models/synthseg_topological_classes_2.0.npy",
    "models/synthmorph.affine.2.h5", "models/synthmorph.deform.3.h5",
    "models/synthstrip.1.pt", "average/RB_all_withskull_2020_01_02.gca",
)


def bundle_fixture(tmp_path):
    bundle = tmp_path / "bundle"
    config_name = "etc/scoped-reference-recon-config.yaml"
    config = bundle / config_name
    config.parent.mkdir(parents=True)
    config.write_text("UseSynthSeg: true\n")
    config_sha = hashlib.sha256(config.read_bytes()).hexdigest()
    rows = [{"path": config_name, "sha256": config_sha}]
    for name in REQUIRED:
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        rows.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (bundle / "manifest.json").write_text(json.dumps({
        "standalone_verified": True, "files": rows,
        "runtime_profile": standalone.RUNTIME_PROFILE,
        "inactive_command_audit": {
            "profile_id": standalone.RUNTIME_PROFILE["id"],
            "resolved_config": {"path": config_name, "sha256": config_sha},
        },
    }))
    return bundle


def test_manifest_rejects_truthy_string_and_missing_inventory(tmp_path):
    bundle = bundle_fixture(tmp_path)
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["standalone_verified"] = "true"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="not passed"):
        standalone._check_bundle(bundle, development=False)
    manifest["standalone_verified"] = True
    manifest["files"].pop()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not cover"):
        standalone._check_bundle(bundle, development=False)


def test_bundle_rejects_untracked_file_and_external_link(tmp_path):
    bundle = bundle_fixture(tmp_path)
    stray = bundle / "bin" / "stray"
    stray.write_text("x")
    with pytest.raises(ValueError, match="Untracked"):
        standalone._check_bundle(bundle, development=False)
    stray.unlink()
    outside = tmp_path / "outside"
    outside.write_text("x")
    (bundle / "libexec").mkdir()
    (bundle / "libexec" / "tcsh").symlink_to(outside)
    with pytest.raises(ValueError, match="untracked symlink"):
        standalone._check_bundle(bundle, development=False)
    (bundle / "libexec" / "tcsh").unlink()
    (bundle / "metadata").mkdir()
    (bundle / "metadata" / "probe.sh").write_text("echo unsafe")
    (bundle / "bin" / "which").symlink_to("../metadata/probe.sh")
    with pytest.raises(ValueError, match="untracked symlink"):
        standalone._check_bundle(bundle, development=False)


def test_runner_uses_bundle_environment_and_fresh_subjects_dir(tmp_path, monkeypatch):
    bundle = bundle_fixture(tmp_path)
    t1 = tmp_path / "t1.nii.gz"
    t1.write_bytes(b"t1")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    subjects = tmp_path / "subjects"
    captured = {}

    def fake_run(command, *, env, stdout, stderr, check):
        captured.update(env)
        config = subjects / "sub01/scripts/recon-config.yaml"
        config.parent.mkdir(parents=True)
        config.write_bytes((bundle / "etc/scoped-reference-recon-config.yaml").read_bytes())
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(standalone.subprocess, "run", fake_run)
    monkeypatch.setattr(standalone, "REQUIRED_OUTPUTS", ())
    monkeypatch.setenv("FREESURFER", "/external/freesurfer")
    monkeypatch.setenv("PYTHONPATH", "/external/python")
    monkeypatch.setenv("LD_PRELOAD", "/external/library.so")
    report = standalone.run_recon_all(t1, "sub01", subjects, bundle, license_file=license_file)
    assert report["return_code"] == 0
    assert report["effective_config_matches_profile"]
    assert captured["FREESURFER"] == captured["FREESURFER_HOME"] == str(bundle)
    assert "PYTHONPATH" not in captured
    assert "LD_PRELOAD" not in captured
    with pytest.raises(FileExistsError, match="fresh, empty"):
        standalone.run_recon_all(t1, "sub02", subjects, bundle, license_file=license_file)


def test_profile_rejects_unreviewed_thread_count_and_config(tmp_path, monkeypatch):
    bundle = bundle_fixture(tmp_path)
    t1 = tmp_path / "t1.nii.gz"
    t1.write_bytes(b"t1")
    license_file = tmp_path / "license.txt"
    license_file.write_text("license")
    subjects = tmp_path / "subjects"
    with pytest.raises(ValueError, match="exactly 4 threads"):
        standalone.run_recon_all(t1, "sub01", subjects, bundle, threads=8,
                                 license_file=license_file)

    def wrong_config(command, *, env, stdout, stderr, check):
        config = subjects / "sub01/scripts/recon-config.yaml"
        config.parent.mkdir(parents=True)
        config.write_text("UseSynthSeg: false\n")
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(standalone.subprocess, "run", wrong_config)
    monkeypatch.setattr(standalone, "REQUIRED_OUTPUTS", ())
    with pytest.raises(RuntimeError, match="reconstruction failed"):
        standalone.run_recon_all(t1, "sub01", subjects, bundle,
                                 license_file=license_file)
    report = json.loads((subjects / "sub01.recon-all.run.json").read_text())
    assert not report["effective_config_matches_profile"]
