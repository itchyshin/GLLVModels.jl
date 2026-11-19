# Mock test for .convert_dispersion on row-effect fits (GLLVModels.jl #1272).
# Run from the repository root: Rscript r/test_convert_dispersion.R
.jl_value <- function(x) x
src <- readLines("r/gllvmtmb_julia.R")
s <- grep("^\\.disp_field <- function", src)
e <- grep("^# The gllvmTMB-style front door", src) - 2
eval(parse(text = src[s:e]))
row_fit <- list(dispersion = 2)  # RowEffectFit / RowRandomFit: no r, no α
stopifnot(isTRUE(all.equal(.convert_dispersion("nb2", row_fit, FALSE)$value, 0.5)))
stopifnot(isTRUE(all.equal(.convert_dispersion("gamma", row_fit, FALSE)$value, 2)))
plain <- list(r = 4, "α" = 3)    # plain fitters keep their own fields
stopifnot(isTRUE(all.equal(.convert_dispersion("nb2", plain, FALSE)$value, 0.25)))
stopifnot(isTRUE(all.equal(.convert_dispersion("gamma", plain, FALSE)$value, 3)))
cat("ok\n")
