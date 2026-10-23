# Deterministic covariance/preparation fixtures; no claim of estimator recovery.
df<-expand.grid(site=factor(paste0('u',1:18)),trait=factor(paste0('t',1:3)))
df$species<-factor(rep(paste0('g',rep(1:6,each=3)),3),levels=paste0('g',1:6))
df$value<-sin(seq_len(nrow(df))/3)+as.integer(df$trait)/5
dfp<-df;dfp$value<-as.numeric(seq_len(nrow(df))%%4)
dfb<-df;dfb$value<-as.numeric(seq_len(nrow(df))%%2)
dfm<-data.frame(site=factor(paste0('u',1:18)),trait=factor('nom'),species=factor(paste0('g',rep(1:6,each=3))),value=factor(rep(c('a','b','c'),6)))
A<-diag(6);dimnames(A)<-list(levels(df$species),levels(df$species))
B<-0.7*A+0.3;dimnames(B)<-dimnames(A)
fixtures<-list(gaussian=df,poisson=dfp,binomial=dfb,multinomial=dfm,A=A,B=B)
control<-gllvmTMBcontrol(n_init=1L,se=FALSE)
cases<-list()
add<-function(id,call,check=NULL,error=NULL) {
 cases[[length(cases)+1L]]<<-list(id=id,call=call,expected=if(is.null(error))'PREPARED' else 'REJECTED_BEFORE_TAPE',check=check,error_class=if(is.null(error))NULL else 'error',error_contains=error)
}
add('INPUT-GAUSS-DEFAULT',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1),df,control=control)),function(x,n)identical(x$random,'z_B') && x$data$use_diag_B==1L && n(x,'theta_diag_B')==3L && n(x,'log_sigma_eps')==0L)
add('INPUT-GAUSS-LOADINGS',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1,unique=FALSE),df,control=control)),function(x,n)identical(x$random,'z_B') && x$data$use_diag_B==0L && n(x,'log_sigma_eps')==1L)
add('INPUT-GAUSS-COMMON',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1,common=TRUE),df,control=control)),function(x,n)n(x,'theta_diag_B')==1L && n(x,'log_sigma_eps')==0L)
add('INPUT-POISSON-DEFAULT',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1),dfp,family=poisson(),control=control)),function(x,n)x$data$use_diag_B==1L && n(x,'theta_diag_B')==3L && all(x$data$family_id_vec==2L))
add('INPUT-BINOMIAL-DEFAULT',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1),dfb,family=binomial(),control=control)),function(x,n)all(x$data$diag_B_skip==1L) && n(x,'theta_diag_B')==0L && all(x$data$family_id_vec==1L))
add('INPUT-ANIMAL-LATENT',quote(gllvmTMB(value~0+trait+animal_latent(species,A=A,d=1),df,cluster='species',control=control)),function(x,n)x$data$use_phylo_rr==1L && 'g_phy'%in%x$random && x$data$d_phy==1L)
add('INPUT-KERNEL-ONE',quote(gllvmTMB(value~0+trait+kernel_latent(species,K=A,d=1,name='a'),df,cluster='species',control=control)),function(x,n)x$data$use_phylo_rr==1L && x$data$n_kernel_tiers==0L)
add('INPUT-KERNEL-TWO',quote(gllvmTMB(value~0+trait+kernel_latent(species,K=A,d=1,name='a')+kernel_latent(species,K=B,d=1,name='b'),df,cluster='species',control=control)),function(x,n)x$data$n_kernel_tiers==2L && all(x$data$kernel_has_diag==0L) && 'g_kernel'%in%x$random)
add('INPUT-KERNEL-TWO-AUTO',quote(gllvmTMB(value~0+trait+kernel_latent(species,K=A,d=1,name='a',unique=TRUE)+kernel_latent(species,K=B,d=1,name='b',unique=TRUE),df,cluster='species',control=control)),function(x,n)x$data$n_kernel_tiers==2L && all(x$data$kernel_has_diag==0L) && !'g_kernel_diag'%in%x$random)
add('INPUT-KERNEL-TWO-EXPLICIT',quote(gllvmTMB(value~0+trait+kernel_latent(species,K=A,d=1,name='a')+kernel_unique(species,K=A,name='a')+kernel_latent(species,K=B,d=1,name='b'),df,cluster='species',control=control)),error='latent-only')
add('INPUT-MN-LATENT',quote(gllvmTMB(value~0+trait+latent(0+trait|site,d=1),dfm,cluster='species',family=multinomial(),control=control)),function(x,n)all(x$data$family_id_vec==16L) && all(x$data$diag_B_skip==1L) && n(x,'theta_diag_B')==0L && 'z_B'%in%x$random)
add('INPUT-MN-ANIMAL-LATENT',quote(gllvmTMB(value~0+trait+animal_latent(species,A=A,d=1),dfm,cluster='species',family=multinomial(),control=control)),function(x,n)x$data$use_phylo_rr==1L && all(x$data$family_id_vec==16L))
add('INPUT-MN-ANIMAL-UNIQUE',quote(gllvmTMB(value~0+trait+animal_latent(species,A=A,d=1,unique=TRUE),dfm,cluster='species',family=multinomial(),control=control)),error='not admitted')
add('INPUT-MN-KERNEL-TWO',quote(gllvmTMB(value~0+trait+kernel_latent(species,K=A,d=1,name='a')+kernel_latent(species,K=B,d=1,name='b'),dfm,cluster='species',family=multinomial(),control=control)),error='multiple kernel_*() terms in one fit (multi-kernel)')

# Require the intended structured-model error, not an unrelated failure.
for(i in seq_along(cases)) if(cases[[i]]$expected=="REJECTED_BEFORE_TAPE") {
 cases[[i]]$error_class<-if(grepl("^INPUT-MN-",cases[[i]]$id)) "gllvmTMB_multinomial_structured_not_admitted" else "rlang_error"
}
