"""Record the maintainer's D-300 answer 9 promotion on the phylo_latent P1 receipts.

D-300 answer 9 (2026-09-27): A14/A15 promote only on a dated maintainer block. The maintainer
signed that promotion on 2026-10-05 (vault D-319: "D: [yes]"; signature source
LOOP/lanes/true-parity-latest/signed-rulings-2026-10-05.md in the lane kit). The block itself is
the section "Maintainer promotion (D-300 answer 9)" of docs/dev-log/core070/phylo-latent-p1/README.md.
This tool records that signature; it does not sign anything.

Mechanism: a text edit, so every other byte of each receipt is kept (no JSON re-serialisation
of the R or Julia numbers). In each of the six receipts the line `"qualified": false` becomes a
`"maintainer_promotion"` object (PROMOTION below) followed by `"qualified": true`. The Julia
receipts are edited first; each R receipt's `julia_receipt_sha256` is then set to the new hash
of its Julia receipt. Nothing else changes: the recorded A15 stall (converged = false,
gradient_not_converged) stays as recorded. After --apply, update the SHA-256 pins in
test/test_phylo_latent_paired_p1.jl and the README hash table.

  python3 tools/phylo_latent/promote_p1.py --apply
  python3 tools/phylo_latent/promote_p1.py --check
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "docs/dev-log/core070/phylo-latent-p1"
CASES = ["struct_phy_tree_rr", "struct_phy_dense_rr", "cov_phylo_latent_rsz"]
HEADING = "## Maintainer promotion (D-300 answer 9)"
PROMOTION = {
    "signed_by": "Shinichi Nakagawa",
    "signed_on": "2026-10-05",
    "ruling": "D-300 answer 9 (dated maintainer block), signed yes under vault D-319 on 2026-10-05",
    "block": "docs/dev-log/core070/phylo-latent-p1/README.md, section 'Maintainer promotion (D-300 answer 9)'",
    "a15_stationarity_gap": "recorded and unchanged; the promotion is despite it (D-319)",
}
QUAL_RE = re.compile(r'^(?P<ind>[ \t]*)"qualified": false(?P<comma>,?)$', re.M)


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def promote_text(text):
    hits = QUAL_RE.findall(text)
    if len(hits) != 1:
        raise SystemExit(f"expected one '\"qualified\": false' line, found {len(hits)}")
    block = json.dumps(PROMOTION, ensure_ascii=False)
    return QUAL_RE.sub(lambda m: f'{m["ind"]}"maintainer_promotion": {block},\n'
                                 f'{m["ind"]}"qualified": true{m["comma"]}', text)


def apply():
    if HEADING not in (DIR / "README.md").read_text():
        raise SystemExit(f"README.md has no '{HEADING}' section: write the dated block first")
    for case in CASES:
        jp, rp = DIR / case / "julia-receipt.json", DIR / case / "r-receipt.json"
        old_j = sha(jp)
        jp.write_text(promote_text(jp.read_text()))
        rt = rp.read_text()
        if f'"julia_receipt_sha256": "{old_j}"' not in rt:
            raise SystemExit(f"{rp}: julia_receipt_sha256 is not the hash of {jp.name} before promotion")
        rt = rt.replace(f'"julia_receipt_sha256": "{old_j}"', f'"julia_receipt_sha256": "{sha(jp)}"')
        rp.write_text(promote_text(rt))
        print(f"{case}: julia {sha(jp)} r {sha(rp)}")


def check():
    bad = []
    if HEADING not in (DIR / "README.md").read_text():
        bad.append(f"README.md lacks '{HEADING}'")
    for case in CASES:
        jp, rp = DIR / case / "julia-receipt.json", DIR / case / "r-receipt.json"
        j, r = json.loads(jp.read_text()), json.loads(rp.read_text())
        for name, d in (("julia", j), ("r", r)):
            if d.get("qualified") is not True or d.get("maintainer_promotion") != PROMOTION:
                bad.append(f"{case}/{name}: not promoted with the recorded block")
        if r.get("julia_receipt_sha256") != sha(jp):
            bad.append(f"{case}: r-receipt julia_receipt_sha256 is not the Julia receipt's hash")
    if bad:
        sys.exit("PHYLO_LATENT_P1_PROMOTION_BAD\n" + "\n".join(bad))
    print("PHYLO_LATENT_P1_PROMOTION_OK")


if __name__ == "__main__":
    {"--apply": apply, "--check": check}.get(sys.argv[1] if len(sys.argv) == 2 else "", lambda: sys.exit(__doc__))()
