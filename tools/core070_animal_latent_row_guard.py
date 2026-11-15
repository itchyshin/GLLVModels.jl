#!/usr/bin/env python3
"""Read-only guard for C2 row namespace/export/animal_latent. No fits, no writes.

Pins the row in its signed state (2026-10-05 D-319: "animal_latent: leave unbound"):
unbound, NOT-MEASURED, no executable ids, no evidence, no disposition, and none of the
neighbouring intercept-only PASS receipts cited as evidence. If the row is ever bound
(by a later signature) the guard instead requires a P1 pin, ids, and a signer for any
signed disposition. It does not bind, sign, or change any status.

Usage: core070_animal_latent_row_guard.py [REPO_ROOT] [--negative-controls]
Exit 0 = all checks pass (and, with --negative-controls, every mutation is caught).
"""
import copy, json, pathlib, re, sys

SLUG = "namespace/export/animal_latent"
P0 = {"SLOPE-ANIMAL-LAT-DEFAULT", "SLOPE-ANIMAL-LAT-NOUNIQUE", "SLOPE-ANIMAL-LAT-RANK4"}
P1_PIN = "9539352f66f2db2cc26b1c393e67212a359b60c9"
TL = "docs/dev-log/core070/true-parity-latest/"
NEIGHBOUR = "CORE070-FIT-INPUT-ANIMAL-LATENT"


def load(root):
    r = pathlib.Path(root)
    return dict(
        cm=json.loads((r / TL / "case-map-namespace.json").read_text()),
        contract=json.loads((r / TL / "namespace-1-batch-contract-p1.json").read_text()),
        sb=(r / TL / "scoreboard.md").read_text(),
        gates=(r / TL / "GATES.md").read_text(),
        plan=json.loads((r / "docs/dev-log/core070/slopes-required-case-plan.json").read_text()),
        src=(r / "src/animal_latent.jl").read_text(),
        mod=(r / "src/GLLVModels.jl").read_text(),
    )


def find(o, key, val):
    if isinstance(o, dict):
        if o.get(key) == val:
            yield o
        for v in o.values():
            yield from find(v, key, val)
    elif isinstance(o, list):
        for v in o:
            yield from find(v, key, val)


def checks(S):
    out = []
    def ok(name, cond, detail=""):
        out.append((name, bool(cond), detail))
    rows = list(find(S["cm"], "source_id", SLUG))
    ok("case-map: exactly one row", len(rows) == 1, len(rows))
    row = rows[0] if rows else {}
    bound = bool(row.get("executable_case_ids") or row.get("evidence")
                 or row.get("measured_against") or row.get("disposition"))
    line = next((l for l in S["sb"].splitlines() if "`" + SLUG + "`" in l), "")
    ok("scoreboard: one row for the slug", bool(line))
    if not bound:
        ok("unbound: scoreboard says NOT-MEASURED and not bound", "NOT-MEASURED" in line and "not bound" in line)
        ok("unbound: scoreboard cites no receipt", "no receipt cited" in line)
    else:
        ok("bound: measured_against is the P1 pin", row.get("measured_against") == P1_PIN, row.get("measured_against"))
        ok("bound: has executable ids", bool(row.get("executable_case_ids")))
        if row.get("disposition") == "DISPOSITION-SIGNED":
            ok("bound: signed disposition has signer and date", row.get("signed_by") and row.get("signed_on"))
    ok("P0 ids unchanged", set(row.get("p0_case_ids", [])) == P0, row.get("p0_case_ids"))
    ok("neighbour intercept-only receipts not cited for this row", NEIGHBOUR not in json.dumps(row))
    ok("GATES quotes the 2026-10-05 signature", "animal_latent: leave unbound" in S["gates"])
    ce = list(find(S["contract"], "source_id", SLUG))
    ok("contract: one entry, status REUSED_OR_RECLASSIFY unless a ruling changed it",
       len(ce) == 1 and (ce[0].get("status") == "REUSED_OR_RECLASSIFY" or bound), [c.get("status") for c in ce])
    ok("src header still says NOT gllvmTMB's animal_latent", "NOT gllvmTMB's" in S["src"])
    ok("src: wrapper has no slope/covariate/formula argument",
       not re.search(r"function fit_animal_latent_gllvm\([^)]*\b(x|X|covariate|formula|slope)\b", S["src"], re.S))
    ok("src: no Julia function literally named animal_latent (no same-name registration route)",
       not re.search(r"function\s+animal_latent\b|^\s*animal_latent\s*\(.*\)\s*=", S["src"] + S["mod"], re.M))
    ids = {}
    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("id"), str) and o["id"].startswith("SLOPE-ANIMAL-LAT"):
                ids[o["id"]] = o
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(S["plan"])
    ok("case plan: the three P0 ids present", set(ids) == P0, sorted(ids))
    ok("case plan: native_status UNPAID unless the row is bound",
       bound or all(v.get("native_status") == "UNPAID" for v in ids.values()))
    return out


def _dis(S):
    for r in find(S["cm"], "source_id", SLUG):
        r.update(disposition="DISPOSITION-SIGNED", measured_against=P1_PIN,
                 executable_case_ids=["X"], evidence={"r": 1})

NEG = {
 "N1 bind row to the unrelated PASS receipt": lambda S: [r.update(executable_case_ids=[NEIGHBOUR + "-NATIVE-MODEL"], evidence={"receipt": [NEIGHBOUR]}, measured_against=P1_PIN) for r in find(S["cm"], "source_id", SLUG)],
 "N2 scoreboard says EVIDENCED with no receipt": lambda S: S.update(sb=re.sub(r"(`" + SLUG + r"`[^\n]*?\| )NOT-MEASURED", r"\1EVIDENCED", S["sb"], count=1)),
 "N3 drop the 2026-10-05 signature quote": lambda S: S.update(gates=S["gates"].replace("animal_latent: leave unbound", "")),
 "N4 drop a P0 id": lambda S: [r.update(p0_case_ids=r["p0_case_ids"][:2]) for r in find(S["cm"], "source_id", SLUG)],
 "N5 wrapper gains a slope argument": lambda S: S.update(src=S["src"].replace("function fit_animal_latent_gllvm(Y::AbstractMatrix,", "function fit_animal_latent_gllvm(Y::AbstractMatrix, x::AbstractVector,", 1)),
 "N6 signed disposition without signer": _dis,
 "N7 a same-named Julia animal_latent appears": lambda S: S.update(src=S["src"] + "\nfunction animal_latent(args...) end\n"),
 "N8 contract entry silently reclassified": lambda S: [c.update(status="TIER0") for c in find(S["contract"], "source_id", SLUG)],
}


def main(argv):
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = args[0] if args else pathlib.Path(__file__).resolve().parents[1]
    S = load(root)
    res = checks(S)
    for n, c, d in res:
        print(("PASS " if c else "FAIL ") + n + (f"   [{d}]" if d != "" and not c else ""))
    rc = 0 if all(c for _, c, _ in res) else 1
    if "--negative-controls" in argv:
        print("\nnegative controls (each mutation must make a check FAIL):")
        for n, mut in NEG.items():
            T = copy.deepcopy(S); mut(T)
            caught = any(not c for _, c, _ in checks(T))
            print(("CAUGHT " if caught else "MISSED ") + n)
            rc = rc or (0 if caught else 1)
    return rc

if __name__ == "__main__":
    sys.exit(main(sys.argv))
