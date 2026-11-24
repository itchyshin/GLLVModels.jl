"""Convert the independently frozen JSON oracle to dependency-free Julia data.
Usage: python3 export_fixture.py /absolute/path/to/frozen/oracle
No package source is imported. The original oracle remains authoritative.
"""
import hashlib
import json
import pathlib
import sys

origin = pathlib.Path(sys.argv[1])
fixture = json.loads((origin / "fixture.json").read_text())
expected = json.loads((origin / "expected.json").read_text())
manifest = (origin / "SHA256SUMS").read_bytes()
assert hashlib.sha256(manifest).hexdigest() == "437977c4f7834f9bb8815aa0f992dccb13234eaeb6f1cd83abb7dfbbbf9f3fdb"
for line in manifest.decode().splitlines():
    checksum, filename = line.split(maxsplit=1)
    path = origin / filename.lstrip("*")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checksum, path

def julia(value):
    if value is None:
        return "nothing"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        return "[" + ",".join(map(julia, value)) + "]"
    return "Dict{String,Any}(" + ",".join(julia(k) + "=>" + julia(v) for k, v in value.items()) + ")"

records = []
for point, oracle in zip(fixture["points"], expected):
    assert point["name"] == oracle["name"]
    records.append({"theta": point["theta"], **oracle})
data = {"schema": fixture["schema"], "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "traits": fixture["traits"], "response": fixture["response"],
        "kernels": fixture["kernels"], "priors": fixture["priors"],
        "known_residual_sd": fixture["known_residual_sd"], "points": records,
        "boundary_references": fixture["boundary_points"],
        "fixed_sigma_sensitivity": fixture["fixed_sigma_sensitivity"]}
path = pathlib.Path(__file__).with_name("fixture.jl")
path.write_text("# Generated from the frozen independent oracle; do not edit numbers.\nPSI_SD_ORACLE = " + julia(data) + "\n")
print(path, path.stat().st_size, "bytes")
