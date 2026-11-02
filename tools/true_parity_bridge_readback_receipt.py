#!/usr/bin/env python3
"""Write the P1 bridge-readback receipt from test/fixtures/bridge_readback_p1.toml.

Both sides of every case were recorded live, in one R session, by
test/fixtures/gen_bridge_readback_p1.R (gllvmTMB at P1 -> JuliaCall -> GLLVModels). This tool
copies them into a comparison block, one case per compared quantity, with the tolerance asserted
for that case in test/test_bridge_readback_p1.jl (the line is read from the test, by its marker,
and copied into the receipt). It does not run R or Julia.

Usage: python3 tools/true_parity_bridge_readback_receipt.py [--check]
  --check  exit 1 if the tracked receipt differs from what this tool would write.
"""
import hashlib
import json
import tomllib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = "test/fixtures/bridge_readback_p1.toml"
GENERATOR = "test/fixtures/gen_bridge_readback_p1.R"
TEST = "test/test_bridge_readback_p1.jl"
DATA = "test/fixtures/ns_gauss_p1_data.csv"
OUT = "docs/dev-log/core070/true-parity-latest/receipts/julia-twins/namespace-numeric/bridge_readback.json"
P1 = "9539352f66f2db2cc26b1c393e67212a359b60c9"

ROWS = {
    "coef": "namespace/S3method/coef,gllvmTMB_julia",
    "fitted": "namespace/S3method/fitted,gllvmTMB_julia",
    "logLik": "namespace/S3method/logLik,gllvmTMB_julia",
    "predict": "namespace/S3method/predict,gllvmTMB_julia",
    "residuals": "namespace/S3method/residuals,gllvmTMB_julia",
    "summary": "namespace/S3method/summary,gllvmTMB_julia",
    "gjf": "namespace/export/gllvm_julia_fit",
}

# (case suffix, row, R-side key in "bridge", direct-Julia key in "julia_direct", R call, Julia call)
READBACK = [
    ("COEF-ALPHA", "coef", "coef_alpha", "alpha", "coef(fit)$alpha", "vec(mean(Y; dims = 2)) (the bridge's intercepts; StatsAPI.coef of the centred fit is empty)"),
    ("COEF-LOADINGS", "coef", "coef_loadings", "loadings", "coef(fit)$loadings (6 x 2, column-major)", "getLoadings(f; rotate = true)"),
    ("FITTED-RESPONSE", "fitted", "fitted_response", "fitted_response", "fitted(fit) (6 x 200, column-major)", "alpha .+ predict(f, Yc; type = :response)"),
    ("FITTED-LINK", "fitted", "fitted_link", "fitted_link", "fitted(fit, type = \"link\")", "alpha .+ predict(f, Yc; type = :link)"),
    ("LOGLIK-VALUE", "logLik", "logLik", "loglik", "as.numeric(logLik(fit))", "loglikelihood(f)"),
    ("LOGLIK-DF", "logLik", "logLik_df", "df", "attr(logLik(fit), \"df\")", "p + dof(f) (the p intercepts the bridge profiles out by centring, plus dof of the centred fit)"),
    ("LOGLIK-NOBS", "logLik", "logLik_nobs", "nobs", "attr(logLik(fit), \"nobs\")", "p * n"),
    ("PREDICT-LINK", "predict", "predict_link_est", "fitted_link", "predict(fit)$est (long frame, trait fastest)", "vec(alpha .+ predict(f, Yc; type = :link))"),
    ("PREDICT-RESPONSE", "predict", "predict_response_est", "fitted_response", "predict(fit, type = \"response\")$est", "vec(alpha .+ predict(f, Yc; type = :response))"),
    ("RESIDUALS-RESPONSE", "residuals", "residuals_response", "resid_response", "residuals(fit) (type = \"response\")", "Y .- (alpha .+ predict(f, Yc))"),
    ("RESIDUALS-PEARSON", "residuals", "residuals_pearson", "resid_pearson", "residuals(fit, type = \"pearson\")", "residuals(f, Yc; type = :pearson)"),
    ("SUMMARY-LOGLIK", "summary", "summary_logLik", "loglik", "summary(fit)$header$logLik", "loglikelihood(f)"),
    ("SUMMARY-AIC", "summary", "summary_AIC", "aic", "summary(fit)$header$AIC", "2k - 2 loglikelihood(f), k = p + dof(f)"),
    ("SUMMARY-BIC", "summary", "summary_BIC", "bic", "summary(fit)$header$BIC", "k log(p n) - 2 loglikelihood(f), k = p + dof(f)"),
    ("SUMMARY-DF", "summary", "summary_df", "df", "summary(fit)$header$df", "p + dof(f)"),
    ("SUMMARY-NOBS", "summary", "summary_nobs", "nobs", "summary(fit)$header$nobs", "p * n"),
    ("SUMMARY-COEF-ALPHA", "summary", "summary_coef_alpha", "alpha", "summary(fit)$coefficients$alpha", "vec(mean(Y; dims = 2))"),
    ("SUMMARY-COEF-LOADINGS", "summary", "summary_coef_loadings", "loadings", "summary(fit)$coefficients$loadings", "getLoadings(f; rotate = true)"),
    ("SUMMARY-SIGMA", "summary", "summary_Sigma", "Sigma", "summary(fit)$covariance$Sigma", "sigma_y_site(f)"),
    ("SUMMARY-CORRELATION", "summary", "summary_correlation", "correlation", "summary(fit)$covariance$correlation", "correlation(f)"),
    ("SUMMARY-COMMUNALITY", "summary", "summary_communality", "communality", "summary(fit)$covariance$communality", "communality(f)"),
]
INTEGER_KEYS = {"logLik_df", "logLik_nobs", "summary_df", "summary_nobs"}


def sha(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def vals(x):
    if isinstance(x, dict):
        return [float(v) for v in x["colmajor"]]
    if isinstance(x, list):
        return [float(v) for v in x]
    return float(x)


def marker_line(marker):
    lines = (ROOT / TEST).read_text().splitlines()
    hits = [(i + 1, s.strip()) for i, s in enumerate(lines) if f"# {marker}" in s]
    if len(hits) != 1:
        raise SystemExit(f"marker {marker} found {len(hits)} times in {TEST}")
    return hits[0]


def diff(r, j):
    if isinstance(r, list):
        assert len(r) == len(j), (len(r), len(j))
        return max(abs(a - b) for a, b in zip(r, j))
    return abs(r - j)


def tol_fields(marker, tol):
    line, text = marker_line(marker)
    return {"tolerance": tol, "tolerance_source": f"{TEST}:{line}", "tolerance_source_line": text}


def build():
    fx = tomllib.loads((ROOT / FIXTURE).read_text())
    assert fx["gllvmtmb_commit"] == P1
    assert fx["data_sha256"] == sha(DATA)
    b, j, g, t = fx["bridge"], fx["julia_direct"], fx["gllvm_julia_fit"], fx["tmb"]
    assert b["converged"] and j["converged"] and g["converged"] and t["convergence"] == 0 and t["pd_hessian"]
    fit_note = ("One live R session: gllvmTMB at P1 (" + fx["gllvmtmb_library"] + ", " + fx["r_version"] + ") drove "
                "JuliaCall " + fx["juliacall_version"] + " into Julia " + fx["julia_version"] + " loading GLLVModels from "
                + fx["gllvmodels_source"] + "; " + fx["formula"] + ", Gaussian, p = 6 traits, n = 200 units, data "
                + DATA + " (sha256 checked). The R value is the output of the R S3 method on the gllvmTMB_julia object "
                "returned by gllvmTMB(..., engine = \"julia\"); the Julia value is the native accessor in the same "
                "session on a refit made with the call bridge_fit makes for this row (alpha = row means of Y; "
                "f = fit_gaussian_gllvm(Yc; K = 2), Yc = Y .- alpha), with Y read back from the bridge object's "
                "bridge_input$y. The bridge returns a payload, not the fit object, hence the refit; two recordings "
                "reproduced every value bitwise.")
    cases = []
    for suffix, row, rk, jk, rcall, jcall in READBACK:
        r, jv = vals(b[rk]), vals(j[jk])
        c = {"case_id": f"P1-BRIDGE-READBACK-{suffix}", "source_id": ROWS[row],
             "quantity": f"{rcall} vs {jcall}",
             "r_source": f"{FIXTURE} bridge.{rk}", "julia_source": f"{FIXTURE} julia_direct.{jk}",
             "r_value": r, "julia_value": jv}
        if rk in INTEGER_KEYS:
            r, jv = int(r), int(jv)
            c.update(r_value=r, julia_value=jv, kind="integer_equality", tolerance=0.5,
                     tolerance_source=f"{TEST}:{marker_line('BRIDGE-READBACK-TOL')[0]}",
                     tolerance_source_line="integer equality (the test asserts the same pair within its 1e-12 floor)")
        else:
            c.update(tol_fields("BRIDGE-READBACK-TOL", 1e-12))
        c["max_abs_diff" if isinstance(r, list) else "abs_diff"] = diff(c["r_value"], c["julia_value"])
        c["note"] = fit_note
        cases.append(c)

    # gllvm_julia_fit export: formula-route agreement, then against the native R engine.
    L = g["loadings"]
    n, k, col = L["nrow"], L["ncol"], [float(v) for v in L["colmajor"]]
    llt = [sum(col[i + r * n] * col[jj + r * n] for r in range(k)) for jj in range(n) for i in range(n)]
    gjf_note = ("gllvm_julia_fit(Y, family = \"gaussian\", num.lv = 2) called directly in the same session (Y the "
                "6 x 200 matrix of the formula route) returned a gllvmTMB_julia object without a marshalling error, "
                "converged; it is compared with the formula route and with gllvmTMB(" + fx["formula"] + ") on the "
                "default engine = \"tmb\" (nlminb code 0, positive-definite Hessian), each side at its own optimum. "
                "Loadings are rotation dependent, so the latent covariance Lambda Lambda' is compared (R: "
                "extract_Sigma(fit, level = \"unit\")$Sigma, which is latent-only for unique = FALSE).")
    gjf = [
        ("GJF-FORMULA-ROUTE-LOGLIK", "gllvm_julia_fit(Y)$loglik vs logLik(gllvmTMB(..., engine = \"julia\"))",
         f"{FIXTURE} bridge.logLik", f"{FIXTURE} gllvm_julia_fit.loglik", vals(b["logLik"]), vals(g["loglik"]),
         tol_fields("GJF-FORMULA-ROUTE-TOL", 1e-12)),
        ("GJF-LOGLIK-TMB", "log-likelihood: engine = \"tmb\" logLik vs gllvm_julia_fit loglik",
         f"{FIXTURE} tmb.logLik", f"{FIXTURE} gllvm_julia_fit.loglik", vals(t["logLik"]), vals(g["loglik"]),
         tol_fields("GJF-LOGLIK-TOL", 1e-6)),
        ("GJF-DF-TMB", "df: attr(logLik(tmb fit), \"df\") vs gllvm_julia_fit df",
         f"{FIXTURE} tmb.df", f"{FIXTURE} gllvm_julia_fit.df", int(t["df"]), int(g["df"]),
         {"kind": "integer_equality", "tolerance": 0.5, "tolerance_source": f"{TEST}:{marker_line('GJF-DF')[0]}",
          "tolerance_source_line": marker_line("GJF-DF")[1]}),
        ("GJF-INTERCEPTS-TMB", "trait intercepts: tmb b_fix vs gllvm_julia_fit alpha (6 values)",
         f"{FIXTURE} tmb.beta", f"{FIXTURE} gllvm_julia_fit.alpha", vals(t["beta"]), vals(g["alpha"]),
         tol_fields("GJF-INTERCEPT-TOL", 3e-12)),
        ("GJF-LATENT-SIGMA-TMB", "latent covariance Lambda Lambda' (6 x 6, column-major): tmb extract_Sigma vs gllvm_julia_fit loadings L L'",
         f"{FIXTURE} tmb.Sigma_latent", f"{FIXTURE} gllvm_julia_fit.loadings (L L' formed by this tool)",
         vals(t["Sigma_latent"]), llt, tol_fields("GJF-LATENT-SIGMA-TOL", 5e-5)),
        ("GJF-SIGMA-EPS-TMB", "residual SD: exp(tmb log_sigma_eps) vs gllvm_julia_fit sigma_eps",
         f"{FIXTURE} tmb.sigma_eps", f"{FIXTURE} gllvm_julia_fit.sigma_eps", vals(t["sigma_eps"]), vals(g["sigma_eps"]),
         tol_fields("GJF-SIGMA-EPS-TOL", 1e-6)),
    ]
    for suffix, q, rs, js, r, jv, tf in gjf:
        c = {"case_id": f"P1-BRIDGE-READBACK-{suffix}", "source_id": ROWS["gjf"], "quantity": q,
             "r_source": rs, "julia_source": js, "r_value": r, "julia_value": jv}
        c.update(tf)
        c["max_abs_diff" if isinstance(r, list) else "abs_diff"] = diff(r, jv)
        c["note"] = gjf_note
        cases.append(c)
    for c in cases:
        assert c[("max_abs_diff" if "max_abs_diff" in c else "abs_diff")] <= c["tolerance"], c["case_id"]

    return {
        "schema": "true-parity-julia-twin-receipt/v1",
        "source_ids": list(ROWS.values()),
        "verdict": "PASS",
        "evidence_kind": "live_bridge_readback",
        "pin": "P1",
        "reference_commit": P1,
        "generator": "tools/true_parity_bridge_readback_receipt.py",
        "recorder": GENERATOR,
        "julia_version": fx["julia_version"],
        "juliacall_version": fx["juliacall_version"],
        "r_version": fx["r_version"],
        "source_fixtures": [{"path": p, "sha256": sha(p)} for p in (FIXTURE, DATA, GENERATOR)],
        "source_tests": [{"path": TEST, "sha256": sha(TEST)}],
        "what_this_is_not": ("Both sides were recorded once, live, by " + GENERATOR + "; this tool copies them and "
                             "runs neither R nor Julia. The readback cases compare the R bridge methods with "
                             "GLLVModels on the same inputs in the same process: they show that the R adapter "
                             "returns what Julia computes, not that Julia agrees with engine = \"tmb\" (only the "
                             "gllvm_julia_fit cases compare with the native R engine). One Gaussian, no-X, "
                             "complete-response fixture; predict is in-sample (newdata is gated on the bridge)."),
        "comparison": {"pin": "P1", "cases": cases},
    }


def main(argv):
    doc = json.dumps(build(), indent=1) + "\n"
    out = ROOT / OUT
    if "--check" in argv:
        if not out.is_file() or out.read_text() != doc:
            print(f"BRIDGE_READBACK_RECEIPT_STALE {OUT}")
            return 1
        print("BRIDGE_READBACK_RECEIPT_OK")
        return 0
    out.write_text(doc)
    print(f"BRIDGE_READBACK_RECEIPT_WRITTEN {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
