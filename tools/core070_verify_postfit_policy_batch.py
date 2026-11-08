"""Verify the retained "postfit-policy" manifest-area batch (15 executable
cases + 2 negative controls; 8 needs-new-Julia-surface deferrals + 1
zero-case spec-defect row are recorded in the contract but never executed).

Two independent checks, mirroring tools/core070_verify_data_batch.py and
tools/core070_verify_masks_known.py:

  1. verify_contract() -- structural checks on the frozen contract itself
     (bucket counts sum to the manifest's 24 rows, no case id appears in more
     than one bucket, source pins are well-formed). Needs no retained run.
  2. verify_state() / check_report() -- structural + hash checks against a
     retained run of tools/core070_postfit_policy_batch.R (receipt.json,
     julia-results.json, results.tsv, diagnostics.log).
  3. --self-test -- mutates a *synthetic* valid report (built from the frozen
     contract, no retained run required) in several independent ways and
     asserts every mutation is rejected by check_report(). Runs without R,
     Julia, or the frozen library. Per the 2026-09-01 vacuous-pass incident,
     --self-test NEVER substitutes for the real --state check below: passing
     --self-test alone still exits nonzero if --state does not exist.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys as _sys  # noqa: E402
_sys.path.insert(0, str(ROOT / "tools"))
from parity_oracle import R_REF_PINS, SELECTED_PIN  # noqa: E402  (GLLVM_PARITY_PIN; P0 default; unknown pin exits)
import core070_source_pin_check  # noqa: E402  (PR #569 review finding 2: library pin + version)
import core070_postcoef_fit_health  # noqa: E402
from core070_json_semantics import json_value_equal  # noqa: E402
CONTRACT_PATH = ROOT / ("docs/dev-log/core070/true-parity-latest/postfit-policy-batch-contract-p1.json"
                        if SELECTED_PIN == "P1" else "docs/dev-log/core070/postfit-policy-batch-contract.json")
REFERENCE_COMMIT = R_REF_PINS[SELECTED_PIN] if SELECTED_PIN == "P1" else "b4d5fee64def88bc768dda1f1f77c29b295edd86"
DEFAULT_STATE = ROOT / ".unlazy/core070-aghq/postfit-policy-batch-01"

MANIFEST_ROW_COUNT = 24


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


POSTCOEF_EXPECTED_LENGTHS = [0, 5, 6]  # contract expected_lengths, in core070_postcoef_fit_health.CASES order


def check_postcoef_three_fit_evidence(evidence):
    need(isinstance(evidence, dict), "three-fit evidence must be an object")
    need(core070_postcoef_fit_health.classify(evidence.get("fit_health")) == "FIT_HEALTH_PASS",
         "POST-COEF fit health is failed or incomplete under the signed rule")
    need(evidence.get("selected_accessor") == "real", "evidence used a mutated accessor")
    expected = [0, 5, 6]
    for key in ("r_lengths", "julia_lengths", "r_design_columns"):
        values = evidence.get(key)
        need(isinstance(values, list) and all(type(n) is int for n in values) and
             values == expected, f"{key} differs from the three exact lengths")
    values = evidence.get("r_values")
    need(isinstance(values, list) and len(values) == 3 and
         all(isinstance(item, list) and len(item) == length and
             all(type(value) in (int, float) and math.isfinite(value) for value in item)
             for item, length in zip(values, expected)),
         "R coefficient values do not match the recorded lengths")
    for key in ("r_names", "julia_names"):
        names = evidence.get(key)
        need(isinstance(names, list) and len(names) == 3 and
             all(item is None or (isinstance(item, list) and len(item) == length and
                 all(isinstance(name, str) for name in item))
                 for item, length in zip(names, expected)),
             f"{key} were not recorded consistently for all fits")
    controls = evidence.get("control_lengths")
    need(isinstance(controls, dict) and set(controls) == {"always_empty", "spurious_coefficient"},
         "three-fit accessor controls are incomplete")
    for key, expected_lengths in (("always_empty", [0, 0, 0]),
                                  ("spurious_coefficient", [1, 6, 7])):
        values = controls[key]
        need(isinstance(values, list) and all(type(n) is int for n in values) and
             values == expected_lengths, f"{key} control did not produce its expected lengths")


def check_postcoef_authenticated_inputs(evidence):
    """Compare each fit's recorded lambda_constraint_map with the authenticated case spec.

    Authenticated source: the contract's CORE070-POSTFIT-COEF-EMPTY-NATIVE
    ``expected_lengths`` [0, 5, 6] (coefficient lengths of the three designs, already
    required of ``julia_lengths`` and ``r_design_columns`` by
    check_postcoef_three_fit_evidence). ``StatsAPI.coef(fit::GllvmFit)`` returns a copy of
    ``fit.pars.β`` and the Julia writer asserts that this is the free-coefficient block, so
    each map's ``q_free`` must equal the case's expected length. No authenticated spec
    exists for ``p``, ``K`` or the map entries: those remain record fields.
    """
    need(isinstance(evidence, dict) and isinstance(evidence.get("fit_health"), dict),
         "three-fit evidence has no fit health")
    fits = evidence["fit_health"].get("Julia")
    need(isinstance(fits, list) and len(fits) == len(core070_postcoef_fit_health.CASES) and
         all(isinstance(fit, dict) for fit in fits), "Julia fit health records are malformed")
    by_case = {fit.get("case"): fit for fit in fits
               if type(fit.get("case")) is str}
    need(set(by_case) == set(core070_postcoef_fit_health.CASES), "Julia fit health case set is malformed")
    for case, length in zip(core070_postcoef_fit_health.CASES, POSTCOEF_EXPECTED_LENGTHS):
        recorded = by_case[case].get("lambda_constraint_map")
        need(isinstance(recorded, dict) and type(recorded.get("q_free")) is int and
             recorded["q_free"] == length,
             f"lambda_constraint_map of {case} differs from the authenticated coefficient length")


def postcoef_pass_trace(evidence):
    """Authenticated-input comparison, then the radius status a certified pass must carry."""
    check_postcoef_authenticated_inputs(evidence)
    return core070_postcoef_fit_health.pass_radius_status(evidence["fit_health"])


def check_postcoef_r_oracle_binding(evidence, oracle):
    need(isinstance(oracle, dict), "R oracle must be an object")
    pair = oracle.get("postcoef_three_fit")
    need(isinstance(pair, dict), "R oracle has no three-fit data")
    fits = pair.get("fits")
    need(isinstance(fits, list) and len(fits) == 3 and
         all(isinstance(fit, dict) for fit in fits), "R oracle must retain three fits")
    need(all(type(fit.get("length")) is int and type(fit.get("fixed_columns")) is int and
             isinstance(fit.get("values"), list) and
             all(type(value) in (int, float) and math.isfinite(value) for value in fit["values"])
             for fit in fits), "R oracle contains malformed numeric fields")
    for report_key, fit_key in (("r_lengths", "length"), ("r_design_columns", "fixed_columns"),
                                ("r_values", "values"), ("r_names", "names")):
        need(all(fit_key in fit for fit in fits), f"R oracle is missing {fit_key}")
        need(report_key in evidence and evidence[report_key] == [fit[fit_key] for fit in fits],
             f"{report_key} does not match the retained R oracle")
    need(all("fit_health" in fit for fit in fits) and
         isinstance(evidence.get("fit_health"), dict) and
         json_value_equal(evidence["fit_health"].get("R"), [fit["fit_health"] for fit in fits]),
         "R fit health does not match the retained R oracle")


def check_postcoef_control_outcome(real, report, kind, exit_code, real_identity, control_identity, contract):
    check_postcoef_three_fit_evidence(real)
    expected = {"always_empty": [0, 0, 0], "spurious_coefficient": [1, 6, 7]}
    need(kind in expected and isinstance(report, dict), "unknown or malformed accessor control")
    need(report.get("status") == "FAIL" and report.get("all_checks") is False,
         "accessor control report is incomplete or did not fail")
    need(report.get("area") == "postfit-policy" and
         report.get("scope") == "CORE070_POSTFIT_POLICY_BATCH" and
         type(report.get("case_count")) is int and report["case_count"] == len(contract["cases"]) and
         type(report.get("negative_control_count")) is int and
         report["negative_control_count"] == len(contract["negative_controls"]),
         "accessor control report metadata differs from the contract")
    need(report.get("all_positive_pass") is False and
         report.get("negative_controls_behaved_as_expected") is True,
         "accessor control batch summary is inconsistent")
    cases = report.get("cases")
    need(isinstance(cases, dict), "accessor control case reports are missing")
    need(set(cases) == {case["case_id"] for case in contract["cases"]},
         "accessor control case set is incomplete")
    need(all(isinstance(value, dict) and value.get("pass") is True
             for key, value in cases.items() if key != "CORE070-POSTFIT-COEF-EMPTY-NATIVE"),
         "an unrelated batch case failed in the accessor control")
    negatives = report.get("negative_controls")
    need(isinstance(negatives, dict) and
         set(negatives) == {control["control_id"] for control in contract["negative_controls"]} and
         all(isinstance(value, dict) and value.get("behaved") is True for value in negatives.values()),
         "another batch negative control is missing or failed")
    case = cases.get("CORE070-POSTFIT-COEF-EMPTY-NATIVE")
    need(isinstance(case, dict) and case.get("pass") is False,
         "POST-COEF case did not report failure")
    control = case.get("three_fit")
    need(isinstance(control, dict), "accessor control three-fit results are missing")
    need(type(exit_code) is int and exit_code == 1,
         "accessor control did not produce a POST-COEF failure with exit 1")
    identity_fields = {"source_commit", "source_tree", "r_oracle_sha256", "julia_runner_sha256"}
    need(isinstance(real_identity, dict) and isinstance(control_identity, dict) and
         set(real_identity) == identity_fields and set(control_identity) == identity_fields and
         real_identity == control_identity, "accessor controls do not share capture identity")
    for key, value in real_identity.items():
        sizes = (40, 64) if key in ("source_commit", "source_tree") else (64,)
        need(isinstance(value, str) and len(value) in sizes and
             all(c in "0123456789abcdef" for c in value), f"invalid capture identity {key}")
    need(control.get("selected_accessor") == kind, "accessor control did not select its mutant")
    values = control.get("julia_lengths")
    need(isinstance(values, list) and all(type(value) is int for value in values) and
         values == expected[kind], "accessor control returned unexpected lengths")
    normalised = dict(control, selected_accessor="real", julia_lengths=real["julia_lengths"])
    check_postcoef_three_fit_evidence(normalised)
    for field in ("r_lengths", "r_design_columns", "r_values", "r_names"):
        need(field in control and control[field] == real[field],
             f"accessor control changed the R oracle field {field}")


def check_postcoef_capture_artifacts(state, repo, identity):
    need(isinstance(identity, dict), "capture identity is missing")
    for key, sizes in (("source_commit", (40, 64)), ("source_tree", (40, 64)),
                       ("r_oracle_sha256", (64,)), ("julia_runner_sha256", (64,))):
        value = identity.get(key)
        need(isinstance(value, str) and len(value) in sizes and
             all(c in "0123456789abcdef" for c in value), f"invalid capture identity {key}")
    state = Path(state).resolve()
    for rel, key in (("r-oracle.json", "r_oracle_sha256"),
                     ("runner-snapshots/core070_postfit_policy_batch.jl", "julia_runner_sha256")):
        path = (state / rel).resolve()
        need(state in path.parents and path.is_file(), f"capture artifact is missing: {rel}")
        need(sha(path) == identity[key], f"capture artifact hash mismatch: {rel}")
    try:
        tree = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify",
                               identity["source_commit"] + ":src"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("capture source commit or src tree is unavailable") from exc
    need(tree == identity["source_tree"], "capture source tree differs from its commit")


def check_postcoef_committed_files(repo, state, commit, contract_rel):
    repo, state = Path(repo).resolve(), Path(state).resolve()
    need(isinstance(commit, str) and len(commit) in (40, 64) and
         all(c in "0123456789abcdef" for c in commit), "capture commit is malformed")
    relative = Path(contract_rel)
    need(not relative.is_absolute() and ".." not in relative.parts, "contract path is unsafe")
    entries = [("tools/core070_postfit_policy_batch.R", state / "runner-snapshots/core070_postfit_policy_batch.R", state),
               ("tools/core070_postfit_policy_batch.jl", state / "runner-snapshots/core070_postfit_policy_batch.jl", state),
               ("tools/core070_verify_postfit_policy_batch.py", repo / "tools/core070_verify_postfit_policy_batch.py", repo),
               (relative.as_posix(), repo / relative, repo)]
    for rel in ("Project.toml", "tools/parity_oracle.py", "tools/core070_source_pin_check.py",
                "tools/core070_source_pin.R", "tools/core070_oracle_pins.toml",
                "tools/core070_postcoef_depot_check.py", "tools/core070_json_semantics.py", "tools/core070_postcoef_fit_health.py",
                "tools/core070_postcoef_fit_health.R", "tools/core070_postcoef_fit_health.jl"):
        entries.append((rel, repo / rel, repo))
    for git_path, local_path, allowed in entries:
        local_path = local_path.resolve()
        need(allowed in local_path.parents and local_path.is_file(), f"captured file missing: {git_path}")
        try:
            blob = subprocess.run(["git", "-C", str(repo), "show", commit + ":" + git_path],
                                  check=True, capture_output=True).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f"captured commit lacks {git_path}") from exc
        need(hashlib.sha256(blob).hexdigest() == sha(local_path),
             f"captured file differs from the committed version: {git_path}")


def check_postcoef_retained_outputs(state, receipt):
    need(isinstance(receipt, dict), "capture receipt is missing")
    need(isinstance(receipt.get("postcoef_capture_identity"), dict), "capture identity is missing")
    state = Path(state).resolve()

    def checked(rel, expected):
        path = (state / rel).resolve()
        need(state in path.parents and path.is_file(), f"retained output is missing: {rel}")
        need(isinstance(expected, str) and sha(path) == expected, f"retained output hash mismatch: {rel}")
        return path

    def report_at(path):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("retained Julia report is malformed") from exc
        need(isinstance(report, dict) and report.get("postcoef_capture_identity") ==
             receipt.get("postcoef_capture_identity"), "retained report capture identity differs")
        return report

    checked("runner-snapshots/core070_postfit_policy_batch.R", receipt.get("r_runner_sha256"))
    real = report_at(checked("julia-results.json", receipt.get("julia_results_sha256")))
    checked("julia-stdout.log", receipt.get("julia_stdout_sha256"))
    checked("julia-stderr.log", receipt.get("julia_stderr_sha256"))
    try:
        library = Path(receipt["frozen_library"]).resolve()
        dll = Path(receipt["r_library_path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("R library capture path is missing") from exc
    need(library in dll.parents and dll.is_file() and sha(dll) == receipt.get("r_library_sha256"),
         "loaded R library does not match the retained capture")
    controls = receipt.get("postcoef_controls")
    need(isinstance(controls, dict) and set(controls) == {"always_empty", "spurious_coefficient"},
         "retained accessor controls are incomplete")
    reports = {}
    for mode, control in controls.items():
        need(isinstance(control, dict) and control.get("mode") == mode and
             type(control.get("exit_code")) is int and control["exit_code"] == 1,
             f"invalid retained accessor control {mode}")
        prefix = "julia-control-" + mode
        reports[mode] = report_at(checked(prefix + ".json", control.get("report_sha256")))
        checked(prefix + ".stdout.log", control.get("stdout_sha256"))
        checked(prefix + ".stderr.log", control.get("stderr_sha256"))
        exit_file = checked(prefix + ".exit", control.get("exit_sha256"))
        need(exit_file.read_text().strip() == "1", f"control {mode} exit record is inconsistent")
    return real, reports


def check_postcoef_runtime(repo, state, receipt, report, controls):
    import core070_postcoef_depot_check
    import tomllib
    repo, state = Path(repo).resolve(), Path(state).resolve()
    runtime = receipt.get("postcoef_runtime")
    keys = {"executable_path", "executable_sha256", "project_sha256", "manifest_sha256",
            "preferences_sha256", "load_path", "depot_path", "preloads", "julia_version",
            "depot_inventory", "depot_checker_sha256", "python_executable_path", "python_executable_sha256",
            "stdlib_root", "depot_check"}
    need(isinstance(runtime, dict) and set(runtime) == keys, "Julia runtime record is incomplete")
    need(runtime["load_path"] == "@:@stdlib" and isinstance(runtime["depot_path"], str) and
         runtime["depot_path"] and isinstance(runtime["julia_version"], str) and
         runtime["julia_version"].startswith("1.10."), "Julia runtime environment is not the reference setup")
    need(report.get("postcoef_runtime") == runtime and isinstance(controls, dict) and
         set(controls) == {"always_empty", "spurious_coefficient"} and
         all(isinstance(control, dict) and control.get("postcoef_runtime") == runtime
             for control in controls.values()), "Julia runtime differs across captures")

    def matches(path, expected, label):
        need(path.is_file() and isinstance(expected, str) and sha(path) == expected, f"{label} changed")

    try:
        launcher = Path(receipt["julia_launcher_path"]).resolve()
        executable = Path(runtime["executable_path"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("Julia runtime path is missing") from exc
    need(launcher == executable and receipt.get("julia_launcher_sha256") == runtime["executable_sha256"],
         "Julia launcher does not match the reported executable")
    stdlib = runtime["stdlib_root"]
    need(isinstance(stdlib, str) and Path(stdlib).is_absolute(), "Julia stdlib path is missing")
    expected_stdlib = (executable.parent.parent / "share/julia/stdlib/v1.10").resolve()
    need(expected_stdlib.is_dir() and Path(stdlib).resolve() == expected_stdlib,
         "Julia stdlib path does not match the executable installation")
    matches(launcher, receipt.get("julia_launcher_sha256"), "Julia launcher")
    matches(state / "runner-snapshots/julia-launcher", receipt.get("julia_launcher_sha256"), "Julia launcher snapshot")
    matches(executable, runtime["executable_sha256"], "Julia executable")
    matches(repo / "Project.toml", runtime["project_sha256"], "Julia project")
    need(not (repo / "Manifest-v1.10.toml").exists(), "alternate Julia manifest is present")
    matches(repo / "Manifest.toml", runtime["manifest_sha256"], "Julia manifest")
    matches(state / "julia-Manifest.toml", runtime["manifest_sha256"], "Julia manifest snapshot")
    matches(repo / "tools/core070_postcoef_depot_check.py", runtime["depot_checker_sha256"], "Julia depot checker")
    python = runtime["python_executable_path"]
    need(isinstance(python, str) and Path(python).is_absolute(), "Python executable path is missing")
    matches(Path(python), runtime["python_executable_sha256"], "Python executable")
    try:
        manifest = tomllib.loads((state / "julia-Manifest.toml").read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("captured Julia manifest is malformed") from exc
    depot_check = core070_postcoef_depot_check.verify_manifest_sources(manifest, runtime["depot_inventory"], stdlib)
    need(depot_check["checked"] > 0, "Julia manifest has no verified package trees")
    try:
        recorded_check = json.loads(runtime["depot_check"])
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded Julia depot check is malformed") from exc
    need(recorded_check == core070_postcoef_depot_check.source_summary(depot_check),
         "Julia depot contents or coverage exclusions changed since capture")
    preferences = repo / "LocalPreferences.toml"
    if runtime["preferences_sha256"] == "absent":
        need(not preferences.exists() and not (state / "julia-LocalPreferences.toml").exists(),
             "Julia preferences appeared after capture")
    else:
        matches(preferences, runtime["preferences_sha256"], "Julia preferences")
        matches(state / "julia-LocalPreferences.toml", runtime["preferences_sha256"], "Julia preferences snapshot")
    need(isinstance(runtime["preloads"], list), "Julia preloads are malformed")
    for preload in runtime["preloads"]:
        need(isinstance(preload, dict) and set(preload) == {"path", "sha256"} and
             isinstance(preload["path"], str) and Path(preload["path"]).is_absolute(),
             "Julia preload entry is malformed")
        matches(Path(preload["path"]), preload["sha256"], "Julia preload")


def check_postcoef_r_runtime(receipt, build):
    need(isinstance(receipt, dict) and isinstance(build, dict), "R runtime or build is missing")
    version = receipt.get("r_version")
    full_version = receipt.get("r_full_version")
    need(isinstance(version, str) and version and isinstance(full_version, str) and
         full_version.startswith(version + " -- ") and full_version == build.get("r_version") and
         isinstance(receipt.get("r_platform"), str) and receipt["r_platform"],
         "R runtime version or platform does not match the oracle build")
    path = receipt.get("r_executable_path")
    need(isinstance(path, str) and Path(path).is_absolute(), "R executable path is missing")
    executable = Path(path)
    need(executable.is_file() and sha(executable) == receipt.get("r_executable_sha256"),
         "R executable changed since capture")


def check_postcoef_oracle_build(repo, state, receipt):
    repo, state = Path(repo).resolve(), Path(state).resolve()
    identity = receipt.get("postcoef_capture_identity")
    need(isinstance(identity, dict), "oracle capture identity is missing")
    commit = identity.get("source_commit")
    need(isinstance(commit, str) and len(commit) in (40, 64) and
         all(c in "0123456789abcdef" for c in commit), "oracle capture commit is malformed")
    builder = (repo / "tools/core070_build_oracle.py").resolve()
    need(repo in builder.parents and builder.is_file() and sha(builder) == receipt.get("oracle_builder_sha256"),
         "oracle verifier differs from the capture")
    try:
        committed = subprocess.run(["git", "-C", str(repo), "show",
                                    commit + ":tools/core070_build_oracle.py"],
                                   check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("captured oracle verifier is unavailable in Git") from exc
    need(hashlib.sha256(committed).hexdigest() == receipt.get("oracle_builder_sha256"),
         "oracle verifier does not match the captured source commit")
    try:
        library = Path(receipt["frozen_library"]).resolve()
    except (KeyError, TypeError) as exc:
        raise ValueError("oracle library path is missing") from exc
    need(library.name == "library" and library.is_dir(), "oracle is not a current-format build")
    build = library.parent
    receipt_path = build / "build.json"
    need(receipt_path.is_file() and sha(receipt_path) == receipt.get("oracle_build_sha256"),
         "oracle build receipt differs from capture")
    try:
        build_metadata = json.loads(receipt_path.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("oracle build receipt is malformed") from exc
    check_postcoef_r_runtime(receipt, build_metadata)
    for when in ("preload", "before", "after"):
        path = (state / f"oracle-verify-{when}.log").resolve()
        need(state in path.parents and path.is_file() and
             sha(path) == receipt.get(f"oracle_verify_{when}_sha256") and
             "CORE070_ORACLE_VERIFY_PASS" in path.read_text().splitlines(),
             f"oracle {when} verification is missing or changed")
    try:
        run = subprocess.run([_sys.executable, str(builder), "verify", "--destination", str(build)],
                             cwd=repo, env=dict(os.environ, GLLVM_PARITY_PIN="P1"),
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("oracle build verification could not complete") from exc
    need(run.returncode == 0 and "CORE070_ORACLE_VERIFY_PASS" in run.stdout.splitlines(),
         "oracle build no longer verifies")


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


# ---------------------------------------------------------------------------
# 1. Contract-only structural checks (no retained run required).
# ---------------------------------------------------------------------------
def verify_contract(contract=None):
    c = contract or load_contract()
    if SELECTED_PIN == "P1":
        p0 = ROOT / "docs/dev-log/core070/postfit-policy-batch-contract.json"
        need(c.get("p0_contract_sha256") == sha(p0), "P1 P0 ancestry hash differs")
        target = [item for item in c.get("cases", []) if item.get("case_id") ==
                  "CORE070-POSTFIT-COEF-EMPTY-NATIVE"]
        need(len(target) == 1, "P1 POST-COEF case is missing or duplicated")
        expected = {"implementation_status": "pending_runner_and_receipt_verification",
                    "expected_lengths": [0, 5, 6], "length_tolerance": 0,
                    "fit_designs": ["no_fixed_effects", "trait_intercepts", "trait_intercepts_and_shared_x"],
                    "accessor_controls": ["always_empty", "spurious_coefficient"],
                    "name_comparison": "record_separately"}
        comparison = target[0].get("discriminating_comparison")
        need(isinstance(comparison, dict) and comparison == expected and
             type(comparison.get("length_tolerance")) is int and
             all(type(value) is int for value in comparison.get("expected_lengths", [])),
             "P1 POST-COEF comparison differs from the approved contract")
    need(c["status"] == "FROZEN_POSTFIT_POLICY_BATCH_CONTRACT", "wrong contract status")
    need(c["area"] == "postfit-policy", "wrong area")
    need(c["reference_commit"] == REFERENCE_COMMIT, "wrong reference commit")
    need(c["manifest_row_count"] == MANIFEST_ROW_COUNT, "manifest row count drift")

    executable_ids = {x["case_id"] for x in c["cases"]}
    needs_surface_ids = {x["source_id"] for x in c["needs_new_julia_surface"]}
    spec_defect_ids = {x["source_id"] for x in c["spec_defect_notes"]}

    need(len(executable_ids) == len(c["cases"]) == c["expected_case_count"] == 15,
         "executable case id set has duplicates or wrong count")
    need(len(needs_surface_ids) == len(c["needs_new_julia_surface"]) ==
         c["needs_new_julia_surface_count"] == 8,
         "needs-new-surface id set has duplicates or wrong count")
    need(len(spec_defect_ids) == len(c["spec_defect_notes"]) == c["spec_defect_note_count"] == 1,
         "spec-defect note set has duplicates or wrong count")

    executable_source_ids = {x["source_id"] for x in c["cases"]}
    need(len(executable_source_ids & needs_surface_ids) == 0,
         "a source_id appears in both the executable and needs-new-surface buckets")
    need(len(executable_source_ids & spec_defect_ids) == 0,
         "a source_id appears in both the executable and spec-defect buckets")
    need(len(needs_surface_ids & spec_defect_ids) == 0,
         "a source_id appears in both the needs-new-surface and spec-defect buckets")

    total_rows = len(executable_source_ids) + len(needs_surface_ids) + len(spec_defect_ids)
    need(total_rows == MANIFEST_ROW_COUNT,
         f"bucket accounting does not sum to the manifest's {MANIFEST_ROW_COUNT} rows (got {total_rows})")

    for x in c["cases"]:
        need(x["case_id"].startswith("CORE070-POSTFIT-") and x["case_id"].endswith("-NATIVE"),
             f"malformed case_id {x['case_id']}")
        need(x["source_id"].startswith("postfit-policy/POST-"), f"malformed source_id on {x['case_id']}")
        for key in ("julia_surface", "r_call", "comparand", "check"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['case_id']} missing {key}")

    for x in c["needs_new_julia_surface"]:
        for key in ("missing_surface", "reason"):
            need(isinstance(x.get(key), str) and len(x[key]) > 0, f"{x['source_id']} missing {key}")

    need(len(c["negative_controls"]) >= 2, "contract itself must name >=2 negative controls")
    for nc in c["negative_controls"]:
        for key in ("control_id", "description", "check"):
            need(isinstance(nc.get(key), str) and len(nc[key]) > 0, f"{nc.get('control_id')} missing {key}")

    need(isinstance(c["source_pins"], dict) and len(c["source_pins"]) == 4, "wrong source_pins count")
    for path_str, digest in c["source_pins"].items():
        need(len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
             f"malformed sha256 for {path_str}")

    print("CORE070_POSTFIT_POLICY_CONTRACT_VERIFIED",
          "executable=", len(executable_ids), "needs_surface=", len(needs_surface_ids),
          "spec_defect=", len(spec_defect_ids), "total=", total_rows)
    return c


# ---------------------------------------------------------------------------
# 2. Retained-run checks.
# ---------------------------------------------------------------------------
def check_report(report, contract, contract_sha256):
    contract_cases = {c["case_id"]: c for c in contract["cases"]}
    contract_neg = {c["control_id"]: c for c in contract["negative_controls"]}

    need(report.get("status") == "PASS", "batch did not pass")
    need(report.get("area") == "postfit-policy", "wrong area")
    need(report.get("scope") == "CORE070_POSTFIT_POLICY_BATCH", "wrong scope")
    need(report.get("case_count") == len(contract_cases), "wrong case count")
    need(report.get("negative_control_count") == len(contract_neg), "wrong negative-control count")
    need(report.get("all_positive_pass") is True, "a required case failed")
    need(report.get("negative_controls_behaved_as_expected") is True, "a negative control misbehaved")
    need(report.get("all_checks") is True, "runner reported an overall failure")

    cases = report.get("cases", {})
    need(set(cases) == set(contract_cases), "case id set does not match contract")
    for cid, case in cases.items():
        need(case.get("pass") is True, f"{cid} did not pass")

    if SELECTED_PIN == "P1":
        check_postcoef_three_fit_evidence(cases["CORE070-POSTFIT-COEF-EMPTY-NATIVE"].get("three_fit"))

    negatives = report.get("negative_controls", {})
    need(set(negatives) == set(contract_neg), "negative-control id set does not match contract")
    for cid, nc in negatives.items():
        need(nc.get("behaved") is True, f"negative control {cid} did not behave as required")


def verify_state(state=DEFAULT_STATE):
    contract = load_contract()
    verify_contract(contract)
    contract_sha256 = sha(CONTRACT_PATH)

    receipt_path = state / "receipt.json"
    julia_path = state / "julia-results.json"
    raw_path = state / "results.tsv"
    diag_path = state / "diagnostics.log"
    need(receipt_path.is_file() and julia_path.is_file() and raw_path.is_file() and diag_path.is_file(),
         "missing retained R-side run")

    receipt = json.loads(receipt_path.read_text())
    need(receipt.get("status") == "PASS" and receipt.get("source_unchanged") is True,
         "receipt did not pass")
    need(receipt.get("reference_commit") == contract["reference_commit"], "wrong reference commit")
    need(receipt.get("contract_sha256") == contract_sha256, "stale contract in receipt")
    need(receipt.get("source_pins") == contract["source_pins"], "receipt source pins changed")
    need(receipt.get("case_count") == contract["expected_case_count"], "receipt case count wrong")
    need(receipt.get("negative_control_count") == len(contract["negative_controls"]),
         "receipt negative-control count wrong")
    need(sorted(receipt.get("expected_case_ids", [])) ==
         sorted(c["case_id"] for c in contract["cases"]), "receipt case id list drifted")
    need(sorted(receipt.get("negative_control_case_ids", [])) ==
         sorted(c["control_id"] for c in contract["negative_controls"]),
         "receipt negative-control id list drifted")
    need(receipt.get("julia_exit_code") == 0, "julia child exited nonzero")
    core070_source_pin_check.check_source_pin(receipt, SELECTED_PIN, need)
    need(receipt.get("julia_results_sha256") == sha(julia_path), "julia-results.json changed since receipt")
    need(receipt.get("raw_sha256") == sha(raw_path), "results.tsv changed since receipt")
    need(receipt.get("diagnostics_sha256") == sha(diag_path), "diagnostics.log changed since receipt")

    report = json.loads(julia_path.read_text())
    check_report(report, contract, contract_sha256)
    radius_status = None  # set on the P1 POST-COEF path from a certified pass
    if SELECTED_PIN == "P1":
        identity = receipt.get("postcoef_capture_identity")
        check_postcoef_capture_artifacts(state, ROOT, identity)
        check_postcoef_committed_files(ROOT, state, identity["source_commit"],
                                       CONTRACT_PATH.relative_to(ROOT))
        check_postcoef_oracle_build(ROOT, state, receipt)
        retained_report, controls = check_postcoef_retained_outputs(state, receipt)
        need(retained_report == report, "retained real report differs from verified report")
        check_postcoef_runtime(ROOT, state, receipt, report, controls)
        evidence = report["cases"]["CORE070-POSTFIT-COEF-EMPTY-NATIVE"].get("three_fit")
        check_postcoef_three_fit_evidence(evidence)
        radius_status = postcoef_pass_trace(evidence)
        need(all(core070_postcoef_fit_health.julia_prior_bound(
            fit.get("penalty_provenance"), identity.get("source_tree"), fit.get("case"))
            for fit in evidence["fit_health"]["Julia"]),
            "Julia penalty provenance differs from the authenticated source tree")
        need(all(core070_postcoef_fit_health.historical_r_prior_bound(
            fit.get("prior_provenance"), receipt.get("oracle_build_sha256"))
            for fit in evidence["fit_health"]["R"]),
            "R prior provenance differs from the authenticated oracle build")
        try:
            r_oracle = json.loads((state / "r-oracle.json").read_text())
        except (OSError, ValueError) as exc:
            raise ValueError("retained R oracle is malformed") from exc
        check_postcoef_r_oracle_binding(evidence, r_oracle)
        for mode, control in controls.items():
            control_receipt = receipt["postcoef_controls"][mode]
            check_postcoef_control_outcome(evidence, control, mode, control_receipt["exit_code"],
                                           identity, control.get("postcoef_capture_identity"), contract)

    diag_text = diag_path.read_text().strip()
    need(diag_text == "", "diagnostics.log is non-empty on an all-PASS run")

    print("CORE070_POSTFIT_POLICY_BATCH_VERIFIED", len(contract["cases"]),
          "negative_controls", len(contract["negative_controls"]))
    return {
        "status": "CORE070_POSTFIT_POLICY_BATCH_PASS",
        "case_count": contract["expected_case_count"],
        "negative_control_count": len(contract["negative_controls"]),
        "receipt_sha256": sha(receipt_path),
        "julia_results_sha256": sha(julia_path),
        "postcoef_radius_status": radius_status,
    }


# ---------------------------------------------------------------------------
# 3. Self-test: synthetic report + mutation battery (no retained run needed).
# ---------------------------------------------------------------------------
def synthetic_report(contract):
    cases = {
        c["case_id"]: {"pass": True}
        for c in contract["cases"]
    }
    if SELECTED_PIN == "P1":
        cases["CORE070-POSTFIT-COEF-EMPTY-NATIVE"]["three_fit"] = {
            "selected_accessor": "real", "r_lengths": [0, 5, 6], "julia_lengths": [0, 5, 6],
            "r_design_columns": [0, 5, 6], "r_values": [[], [0.0] * 5, [0.0] * 6],
            "r_names": [None, None, None], "julia_names": [None, None, None],
            "control_lengths": {"always_empty": [0, 0, 0], "spurious_coefficient": [1, 6, 7]},
            "fit_health": {engine: [dict(case=case, pd_hessian=True, max_abs_gradient=1e-7,
                                           parameters=[0.0, 0.0], gradient=[1e-7, 0.0], free_indices=[1, 2],
                                           optimizer_objective=3.0, recomputed_objective=3.0,
                                           objective_verified=True, penalty_configuration_recorded=True,
                                           error=None, **({'fixed_prior_state': 'none', 'ridge_tau': 'Inf',
                                                           'ridge_penalised': False} if engine == 'R'
                                                          else {'reconstruction': {'recovered_parameters': [0.0, 0.0], 'recovered_loglik': -3.0, 'parameters_match': True, 'loglik_matches': True, 'error': None}, 'penalty_state': 'none', 'objective_difference': 0.0, 'objective_roundoff_limit': 5.329070518200751e-15, 'hessian': [[1.0, 0.0], [0.0, 1.0]], 'hessian_raw': [[1.0, 0.0], [0.0, 1.0]], 'hessian_asymmetry': 0.0, 'hessian_asymmetry_limit': 1.4210854715202004e-14, 'optimizer': {'optimizer_parameters': [0.0, 0.0], 'optimizer_gradient': [1e-7, 0.0], 'optimizer_max_abs_gradient': 1e-7, 'optimizer_minimum': 3.0, 'optimizer_profile_value': 3.0, 'optimizer_profile_verified': True, 'optimizer_error': None}}))
                                    for case in core070_postcoef_fit_health.CASES]
                           for engine in core070_postcoef_fit_health.ENGINES}}
    negatives = {
        nc["control_id"]: {"behaved": True}
        for nc in contract["negative_controls"]
    }
    return {
        "status": "PASS",
        "area": "postfit-policy",
        "scope": "CORE070_POSTFIT_POLICY_BATCH",
        "case_count": len(cases),
        "negative_control_count": len(negatives),
        "all_positive_pass": True,
        "negative_controls_behaved_as_expected": True,
        "all_checks": True,
        "cases": cases,
        "negative_controls": negatives,
    }


def self_test():
    contract = load_contract()
    verify_contract(contract)  # sanity: the frozen contract itself is well-formed
    contract_sha256 = sha(CONTRACT_PATH)
    good = synthetic_report(contract)
    check_report(good, contract, contract_sha256)  # sanity: synthetic report is itself valid

    first_case_id = contract["cases"][0]["case_id"]
    first_neg_id = contract["negative_controls"][0]["control_id"]

    mutations = [
        ("drop a case", lambda r: r["cases"].pop(first_case_id)),
        ("flip status", lambda r: r.update(status="FAIL")),
        ("flip all_checks", lambda r: r.update(all_checks=False)),
        ("flip all_positive_pass", lambda r: r.update(all_positive_pass=False)),
        ("flip negative_controls_behaved_as_expected",
         lambda r: r.update(negative_controls_behaved_as_expected=False)),
        ("flip one case's pass", lambda r: r["cases"][first_case_id].__setitem__("pass", False)),
        ("flip one negative control's behaved",
         lambda r: r["negative_controls"][first_neg_id].__setitem__("behaved", False)),
        ("drop a negative control", lambda r: r["negative_controls"].pop(first_neg_id)),
        ("wrong case count", lambda r: r.update(case_count=99)),
        ("duplicate-shaped extra case",
         lambda r: r["cases"].__setitem__("CORE070-POSTFIT-BOGUS-NATIVE", {"pass": True})),
    ]
    for name, mutate in mutations:
        bad = deepcopy(good)
        mutate(bad)
        try:
            check_report(bad, contract, contract_sha256)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid postfit-policy-batch evidence: {name}")
    print("CORE070_POSTFIT_POLICY_BATCH_NEGATIVES_PASS", len(mutations))

    # Contract-level mutations: a tampered bucket accounting must also be caught.
    contract_mutations = [
        ("shrink executable cases without updating expected_case_count",
         lambda c: c["cases"].pop()),
        ("duplicate a source_id across buckets",
         lambda c: c["needs_new_julia_surface"].append(
             {**c["cases"][0], "source_id": c["cases"][0]["source_id"],
              "missing_surface": "x", "reason": "x"})),
        ("wrong manifest_row_count", lambda c: c.update(manifest_row_count=23)),
        ("only one negative control",
         lambda c: c.__setitem__("negative_controls", c["negative_controls"][:1])),
    ]
    for name, mutate in contract_mutations:
        bad = deepcopy(contract)
        mutate(bad)
        try:
            verify_contract(bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid postfit-policy contract: {name}")
    print("CORE070_POSTFIT_POLICY_CONTRACT_NEGATIVES_PASS", len(contract_mutations))

    n_pin = core070_source_pin_check.self_test(SELECTED_PIN)
    print("CORE070_POSTFIT_POLICY_SOURCE_PIN_NEGATIVES_PASS", n_pin)

    return len(mutations) + len(contract_mutations) + n_pin


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    # --self-test never substitutes for the real --state check below --
    # there is deliberately no flag that skips it (the vacuous-pass
    # incident: verifier exit 0 with only self-test output while the real
    # batch had produced nothing).
    mutation_count = self_test() if args.self_test else None

    if not args.state.exists():
        raise SystemExit(f"verify_postfit_policy_batch: state does not exist: {args.state}")
    result = verify_state(args.state)
    if result and mutation_count:
        result["self_test_mutations_rejected"] = mutation_count
    if args.output and result:
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
