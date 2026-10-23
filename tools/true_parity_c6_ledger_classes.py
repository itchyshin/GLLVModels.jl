#!/usr/bin/env python3
"""Print the tools/parity_ledger.py reverse-direction class of each name given as an argument.

One line per name, three tab-separated fields:

    name <TAB> class <TAB> alias_of

`class` is classify_ahead(name) from tools/parity_ledger.py, the written class that script prints in
its REVERSE section for a Julia export with no R twin (empty when the name has no class).
`alias_of` lists the R export names whose hand-made ALIASES entry maps to this Julia name (comma
separated, empty when none); the ledger treats such a name as the Julia twin of an R export.

This reads the class tables of parity_ledger.py only. It never reads gllvmTMB, so it runs anywhere
the repo does. tools/true_parity_c6_classify.jl calls it. Read-only.

    python3 tools/true_parity_c6_ledger_classes.py rrr_marginal_loglik welch_t fit_gllvm
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parity_ledger as pl  # noqa: E402


def alias_of(name: str):
    n = pl.norm(name)
    return sorted(r for r, j in pl.ALIASES.items() if pl.norm(j) == n)


def main(argv):
    for name in argv:
        cls = pl.classify_ahead(name) or ""
        assert "\t" not in cls and "\n" not in cls, name
        print(f"{name}\t{cls}\t{','.join(alias_of(name))}")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main(sys.argv[1:]))
