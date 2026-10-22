# P1 FAMILY-00 identity pin assessment

## Question

Can the existing df-8 numerical result be bound without changing the installed oracle pin?

## Evidence

`test/parity/parity_helpers.jl` calls
`_core070_check_installed_package_version(get(pin, "gllvmTMB_version", nothing),
package_version)` after confirming that the marker is in the package directory.
That checker accepts only a string-valued `gllvmTMB_version` field equal to the
selected P1 version, `0.7.1`.

The installed `CORE070_SOURCE_PIN.toml` contains matching commit and hash fields,
but contains no `gllvmTMB_version` field. None of its existing fields is read as a
version or substitutes for that required key. The checker therefore fails closed
with `R source-pin marker lacks gllvmTMB_version`.

Independently, the tracked P1 oracle build receipt
`docs/dev-log/core070/true-parity-latest/receipts/covariance/oracle/build.json`
also lacks `gllvmTMB_version`. `_core070_check_oracle_receipt` requires that field
for build receipts and would reject the run even if `find.package("gllvmTMB")`
were corrected to the oracle library.

## Conclusion

**NOT_PIN_CLEAN**

Binding requires a rebuilt or otherwise updated oracle evidence set that adds
`gllvmTMB_version = "0.7.1"` to both the installed marker and the tracked build
receipt. The existing marker fields do not satisfy this version check.
