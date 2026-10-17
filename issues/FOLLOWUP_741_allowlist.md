# Follow-up issue draft (GitHub issue create returned 403)

**Intended title:** bridge_fit: per-route allow-list for unknown options keys (follow-up of #741)

**Repo:** itchyshin/GLLVModels.jl

**Labels:** code-review

**Needs:** a maintainer (or a token with `issues: write`) to open this as a GitHub issue, then paste the issue number into PR #922's body in place of this file path.

---

Follow-up of #741 / PR #922.

#922 rejects the phylo-precision diagnostic hook’s silently dropped arguments (`d != 1`, `X`, `X_lv`, `mask`, `N`, `trait_names`, `unit_names`, `ci_method != "none"`, `reml = true`, `coef_fixed`-style options), `reml = true` on routes that do not read it, and a non-`Dict` `options` container.

**Not done there:** a per-route allow-list for every unknown `options` key. Unknown keys are still dropped at the other `_bridge_get` lookups in `src/bridge.jl` (about 28 sites); only the container type is checked.

Issue #741 asks: each route either uses every supplied argument or throws `ArgumentError` naming the ignored one. For `options`, unknown keys and keys the chosen route does not read should be rejected against a per-route allow-list.

That is a larger design change because the R side (`R/julia-bridge.R`) may rely on pass-through keys. Do not treat #741 as fully covered until this allow-list lands.

## What the code would need to guarantee

- Each `bridge_fit` route has an explicit allow-list of `options` keys it reads.
- An unknown key, or a recognised key on a route that does not use it, throws `ArgumentError` naming the key and the route.
- Confirm against gllvmTMB `engine = "julia"` that pass-through keys used today are either allowed or rejected on purpose.

## Cause tag

silently-ignored-argument

Refs #741. PR #922 keeps `Refs #741` until this lands.
