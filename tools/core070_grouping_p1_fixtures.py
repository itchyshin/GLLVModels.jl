#!/usr/bin/env python3
"""Generate the four literal Gaussian grouping-level fixtures for the P1 receipts.

One small long-format CSV per grouping level (unit, unit_obs, cluster, cluster2),
drawn from a documented Gaussian model with a pure-Python generator (a fixed
64-bit LCG plus Box-Muller), so the bytes do not depend on R's or Julia's RNG.
The CSVs are committed; the receipts read them, never this generator. A
manifest records each file's sha256, and both engine runners refuse a CSV
whose hash differs from the manifest.

Model for every level (two traits t, observation row i, group g(i) of the level):
    value_ti = beta_t + u_{t,g(i)} + eps_ti
    u_{t,g} ~ N(0, sd_t^2) independently per trait (diagonal, `indep`)
    eps_ti  ~ N(0, sigma_eps^2)
beta = (0.3, -0.5), sd = (0.8, 0.5), sigma_eps = 0.4.

Designs (rows are one observation x one trait; `obs_row` indexes the
observation, i.e. one column of Julia's traits x observations matrix):
  unit:     12 units x 4 replicate observations; group = unit.
  unit_obs: 12 units x 2 nested unit_obs x 2 replicates; group = unit_obs
            (labels globally unique, nested in unit).
  cluster:  6 clusters x 4 units x 2 replicates, unit labels u1..u4 recur in
            every cluster (crossed); group = cluster.
  cluster2: same shape as cluster; group = cluster2.

Usage:
  python3 tools/core070_grouping_p1_fixtures.py          # (re)write fixtures + manifest
  python3 tools/core070_grouping_p1_fixtures.py --check  # regenerate in memory, compare bytes
"""
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/dev-log/core070/true-parity-latest/receipts/grouping/fixtures"
BETA = (0.3, -0.5)
SD = (0.8, 0.5)
SIGMA_EPS = 0.4
SEEDS = {"unit": 20260928, "unit_obs": 20260929, "cluster": 20260930, "cluster2": 20261001}
HEADER = ["obs_row", "trait", "unit", "unit_obs", "cluster_id", "cluster2_id", "value"]


class LCG:
    """Knuth MMIX 64-bit LCG; uniform from the top 53 bits."""

    def __init__(self, seed):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def uniform(self):
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return ((self.state >> 11) + 0.5) / 2.0 ** 53

    def normal(self):
        u1, u2 = self.uniform(), self.uniform()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def design(level):
    """Return a list of observation dicts (without values), in row order."""
    obs = []
    if level == "unit":
        for u in range(1, 13):
            for r in range(1, 5):
                obs.append({"unit": f"u{u}", "unit_obs": f"u{u}_o{r}", "cluster_id": "c0",
                            "cluster2_id": "k0", "group": f"u{u}"})
    elif level == "unit_obs":
        for u in range(1, 13):
            for o in range(1, 3):
                for r in range(1, 3):
                    obs.append({"unit": f"u{u}", "unit_obs": f"u{u}_o{o}", "cluster_id": "c0",
                                "cluster2_id": "k0", "group": f"u{u}_o{o}"})
    elif level in ("cluster", "cluster2"):
        for c in range(1, 7):
            for u in range(1, 5):
                for r in range(1, 3):
                    cl = f"c{c}" if level == "cluster" else "c0"
                    k2 = f"k{c}" if level == "cluster2" else "k0"
                    obs.append({"unit": f"u{u}", "unit_obs": f"u{u}_{cl}{k2}_r{r}", "cluster_id": cl,
                                "cluster2_id": k2, "group": cl if level == "cluster" else k2})
    else:
        raise ValueError(level)
    return obs


def render(level):
    rng = LCG(SEEDS[level])
    obs = design(level)
    groups = sorted({o["group"] for o in obs},
                    key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)])
    effect = {(g, t): SD[t] * rng.normal() for g in groups for t in (0, 1)}
    lines = [",".join(HEADER)]
    for i, o in enumerate(obs, start=1):
        for t in (0, 1):
            v = BETA[t] + effect[(o["group"], t)] + SIGMA_EPS * rng.normal()
            lines.append(",".join([str(i), f"trait_{t + 1}", o["unit"], o["unit_obs"], o["cluster_id"],
                                   o["cluster2_id"], repr(v)]))
    return ("\n".join(lines) + "\n").encode()


def manifest(blobs):
    return {"generator": "tools/core070_grouping_p1_fixtures.py",
            "model": {"beta": BETA, "sd": SD, "sigma_eps": SIGMA_EPS, "seeds": SEEDS},
            "files": {f"{lvl}.csv": hashlib.sha256(b).hexdigest() for lvl, b in blobs.items()}}


def main():
    blobs = {lvl: render(lvl) for lvl in SEEDS}
    man = manifest(blobs)
    if "--check" in sys.argv:
        bad = [f for f in man["files"] if not (OUT / f).is_file() or (OUT / f).read_bytes() != blobs[f[:-4]]]
        on_disk = json.loads((OUT / "manifest.json").read_text())
        if bad or on_disk != json.loads(json.dumps(man)):
            raise SystemExit(f"fixture drift: {bad or 'manifest.json'}")
        print("fixtures OK:", ", ".join(f"{f} {h[:12]}" for f, h in man["files"].items()))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for lvl, b in blobs.items():
        (OUT / f"{lvl}.csv").write_bytes(b)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    print(json.dumps(man["files"], indent=2))


if __name__ == "__main__":
    main()
