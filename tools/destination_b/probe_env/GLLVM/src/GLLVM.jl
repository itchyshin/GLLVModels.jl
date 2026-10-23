# Probe-only compatibility shim for the frozen S4 recorder. Never registered,
# not part of the GLLVModels package, do not use it anywhere else.
#
# Why it exists: the package was renamed GLLVM -> GLLVModels (#423), but the
# frozen recorder (gllvmTMB PR #1283 @ 97214679c, D-220: never edited) still
# runs `using GLLVM` (tests/testthat/test-julia-phylo-rr-bridge.R lines 209,
# 273, 368, 437; R/julia-bridge.R line 305; the runner's clean-Julia probe in
# tests/testthat/run-destination-b-s4-public-phylo-dep-isolated.R) and then
# calls exactly one name, `GLLVM.bridge_fit` (R/julia-bridge.R lines 2753 and
# 3263). Maintainer decision 2026-09-24, option A. Every probe receipt made
# through this shim records it as a deviation.
module GLLVM
using GLLVModels: GLLVModels, bridge_fit
export bridge_fit
end
