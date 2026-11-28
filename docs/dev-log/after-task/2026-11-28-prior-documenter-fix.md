# Prior-table documentation build

The ready pull request's Documenter job stopped because nine source-fit and
convergence docstrings were absent from the manual. The low-level reference
now includes those entries and identifies underscore-prefixed functions as
internal. The change does not widen the supported fitting or inference scope.

An isolated local-preview build used the source archive for commit `08e407b`
and the revised reference page. It exited 0 on Julia 1.12.6; the log SHA-256
is `8a1833dc42f9793f2ca5d1d61c12d0b8761513029abf081932178218bd1c6c04`.
Documenter noted its local environment and supplied default VitePress assets.
The initial build without local-preview mode stopped because the archive lacked
Git history; that attempt is preserved separately. No site was deployed.

The earlier `Pkg.test()` result applies to unchanged production code. GitHub
checks on the revised pull-request head still need to finish.

The README now directs readers to the separate Gaussian residual-SD and
named-source unique-SD prior routes described in the prior guide.

The local-preview manual build including the README correction exited 0; its
log SHA-256 is `f30ddb47baf8af13243deb649437aea0292f97ba0fa30390a29a4a9616a73dbe`.
