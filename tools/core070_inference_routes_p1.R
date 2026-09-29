# P1 twin of tools/core070_inference_routes.R (that file stays byte-identical; the
# P0 inference-batch contract pins its hash). The only change: R/temporal.R is
# added to the parsed file list. At gllvmTMB P1 (9539352f6) confint.gllvmTMB_multi
# first calls .temporal_assert_no_iid_inference(), defined in R/temporal.R; the
# P0 file list does not define it, so every probe row errors with 'could not find
# function' (measured: 98/98 FAIL). The guard is a no-op here because the
# synthetic object has no $temporal. Cases, fixture and expectations are unchanged.
# Frozen-source control-flow probes. Intercepted endpoints throw, never fit or return intervals.
args <- commandArgs(TRUE)
stopifnot(length(args)==3L)
source_root <- args[[1]]; fixture <- args[[2]]; output <- args[[3]]
e <- new.env(parent=globalenv())
# Only function definitions are evaluated: no package initialization or top-level runtime work.
for (file in c('mspl.R','fit-multi.R','z-confint-gllvmTMB.R','temporal.R')) {
 for (expr in parse(file.path(source_root,'R',file))) {
  if (is.call(expr) && identical(expr[[1]],as.name('<-')) && length(expr)==3L &&
      is.call(expr[[3]]) && identical(expr[[3]][[1]],as.name('function'))) eval(expr,e)
 }
}
e$`%||%` <- function(x,y) if (is.null(x)) y else x
# Synthetic inventory supplies labels only. It cannot certify their estimands or fitted availability.
e$profile_targets <- function(object,...) data.frame(parm=c('beta','sigma_eps'),tmb_parameter=c('b_fix','log_sigma_eps'))
route_stop <- function(endpoint,method) stop(structure(list(message=paste(endpoint,method,sep=':'),call=NULL),class=c('route_boundary','error','condition')))
endpoint <- function(name) {force(name); function(object,parm,level,method,...) route_stop(name,method)}
original <- as.list(e, all.names=TRUE)
cases <- read.delim(fixture,stringsAsFactors=FALSE,check.names=FALSE)
stopifnot(!anyDuplicated(cases$id))
results <- vector('list',nrow(cases))
for (i in seq_len(nrow(cases))) {
 row <- cases[i,]; list2env(original,e)
 if (row$stage=='dispatch') {
  for (name in c('.confint_lambda','.confint_icc','.confint_phylo_signal','.confint_communality','.confint_rho','.confint_proportion','.confint_sigma')) assign(name,endpoint(name),e)
 }
 e$.confint_fixef_profile <- function(...) route_stop('.confint_fixef_profile','profile')
 e$.confint_profile_targets <- function(...) route_stop('.confint_profile_targets','profile')
 e$.confint_wald_targets <- function(...) route_stop('.confint_wald_targets','wald')
 e$tidy <- function(...) route_stop('tidy','wald')
 object <- list(estimator='ML',likelihood_weights=list(active=FALSE),data=data.frame(trait=factor(c('a','b','c'))),trait_col='trait',report=list(Lambda_B=matrix(1,3,1)),sd_report=list(cov.fixed=diag(2)))
 if (row$variant=='mspl') object$estimator <- 'MSPL'
 if (row$variant=='weighted') object$likelihood_weights$active <- TRUE
 if (row$variant=='no-se') object$sd_report <- NULL
 if (row$variant=='bad-se') object$sd_report$cov.fixed[,] <- NaN
 callargs <- list(object=object,parm=row$parm)
 if (row$method!='DEFAULT') callargs$method <- row$method
 messages <- character()
 actual <- withCallingHandlers(tryCatch({do.call(e$confint.gllvmTMB_multi,callargs);simpleError('UNEXPECTED_NUMERICAL_RETURN')},error=identity),message=function(m){messages <<- c(messages,conditionMessage(m));invokeRestart('muffleMessage')})
 pass <- switch(row$expected_kind,
  route=inherits(actual,'route_boundary') && identical(conditionMessage(actual),row$expected),
  class=inherits(actual,row$expected) && !inherits(actual,'route_boundary'),
  error=inherits(actual,'error') && !inherits(actual,'route_boundary') && grepl(row$expected,conditionMessage(actual),fixed=TRUE),FALSE)
 # Bootstrap fixed/direct-target fallback must remain visible, not silently become genuine bootstrap.
 if (row$stage=='dispatch' && row$parm %in% c('beta','sigma_eps') && row$method=='bootstrap') pass <- pass && any(grepl('falling back',messages,fixed=TRUE))
 results[[i]] <- data.frame(id=row$id,pass=pass,actual_class=paste(class(actual),collapse=';'),actual=gsub('[\r\n\t]+',' ',conditionMessage(actual)),messages=gsub('[\r\n\t]+',' ',paste(messages,collapse=' | ')))
}
result <- do.call(rbind,results)
write.table(result,output,sep='\t',quote=TRUE,row.names=FALSE)
cat('RUNTIME',R.version.string,'cli',as.character(packageVersion('cli')),'\n')
cat('RESULT',sum(result$pass),'PASS',sum(!result$pass),'FAIL\n')
if (!all(result$pass)) {print(result[!result$pass,]);quit(status=1L)}
cat('FROZEN_INFERENCE_ROUTE_PROBES_PASS_NOT_INTERVAL_PARITY\n')
