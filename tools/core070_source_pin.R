# Oracle source-pin check shared by the postfit batch runners (PR #569 review
# finding 2). Same rule as the wave6 / covariance batch runners (PR #567
# review finding 6): the expected commit, tree hash, archive hash, NAMESPACE
# hash and package version come from tools/core070_oracle_pins.toml, and the
# library must carry the CORE070_SOURCE_PIN.toml marker written by
# tools/core070_build_oracle.py build. The marker is required at P1; at P0 a
# library without one still runs as before, but a marker that is present must
# match the P0 pin. The returned record goes into the batch receipt so the
# verifier can check it against the same pin table.
#
# Usage (after the library is on .libPaths() and gllvmTMB is loaded):
#   source(file.path(root, "tools/core070_source_pin.R"))
#   source_pin <- core070_source_pin(root, frozen_library, parity_pin, expected_reference)

core070_read_flat_toml <- function(path, table = NULL) {
  lines <- readLines(path, warn = FALSE)
  if (!is.null(table)) {
    heads <- grep("^\\[", lines)
    start <- match(paste0("[", table, "]"), trimws(lines))
    if (is.na(start)) stop("table [", table, "] not found in ", path)
    stop_at <- c(heads[heads > start], length(lines) + 1L)[[1L]]
    lines <- lines[seq.int(start + 1L, stop_at - 1L)]
  }
  kv <- regmatches(lines, regexec('^([A-Za-z0-9_]+) = "([^"]*)"$', lines))
  kv <- kv[lengths(kv) == 3L]
  stats::setNames(lapply(kv, `[[`, 3L), vapply(kv, `[[`, "", 2L))
}

core070_source_pin <- function(root, frozen_library, parity_pin, expected_reference) {
  oracle_pin <- core070_read_flat_toml(file.path(root, "tools/core070_oracle_pins.toml"), parity_pin)
  if (!identical(oracle_pin$reference_commit, expected_reference)) {
    stop("expected reference ", expected_reference, " != tools/core070_oracle_pins.toml [", parity_pin, "]")
  }
  marker_path <- file.path(frozen_library, "gllvmTMB", "CORE070_SOURCE_PIN.toml")
  if (!file.exists(marker_path)) {
    if (identical(parity_pin, "P1")) {
      stop("GLLVM_PARITY_PIN=P1 needs the CORE070_SOURCE_PIN.toml marker in ", file.path(frozen_library, "gllvmTMB"))
    }
    return(NULL)
  }
  marker <- core070_read_flat_toml(marker_path)
  installed_version <- as.character(utils::packageVersion("gllvmTMB"))
  for (key in c("reference_commit", "source_tree_sha256", "archive_sha256", "namespace_sha256")) {
    if (!identical(marker[[key]], oracle_pin[[key]])) {
      stop("CORE070_SOURCE_PIN.toml ", key, " does not match tools/core070_oracle_pins.toml [", parity_pin, "]")
    }
  }
  if (!identical(installed_version, oracle_pin$version)) {
    stop("installed gllvmTMB ", installed_version, " != tools/core070_oracle_pins.toml [", parity_pin,
         "] version ", oracle_pin$version)
  }
  c(marker[c("reference_commit", "source_tree_sha256", "installed_tree_sha256",
             "archive_sha256", "namespace_sha256")],
    list(version = installed_version, marker_path = normalizePath(marker_path),
         marker_sha256 = core070_sha256_file(marker_path)))
}

core070_sha256_file <- function(path) {
  command <- if (nzchar(Sys.which("sha256sum"))) "sha256sum" else "shasum"
  argv <- if (identical(command, "sha256sum")) path else c("-a", "256", path)
  line <- system2(command, argv, stdout = TRUE, stderr = TRUE)
  stopifnot(is.null(attr(line, "status")), length(line) >= 1L)
  sub("[[:space:]].*$", "", line[[1L]])
}
