# Which R library the second-order tools load gllvmTMB from.
#
# GLLVM_PARITY_R_LIBS set   -> that one library, which must hold gllvmTMB.
#                              A missing library or package is an error, never
#                              a silent fall back to R's default library.
# GLLVM_PARITY_R_LIBS unset -> `nothing`: R's own .libPaths() decides, as before.
#
# Pure Julia (no RCall) so the core suite can test it without R.
function second_order_r_lib(env::AbstractDict = ENV)
    lib = strip(get(env, "GLLVM_PARITY_R_LIBS", ""))
    isempty(lib) && return nothing
    isdir(joinpath(lib, "gllvmTMB")) || throw(ArgumentError(
        "GLLVM_PARITY_R_LIBS=$lib does not contain gllvmTMB; " *
        "refusing to fall back to R's default library. Point it at the " *
        "library that holds the intended gllvmTMB build, or unset it."))
    return realpath(lib)
end
