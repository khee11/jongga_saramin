# Welch's approximation
# batch

###################
# packages needed #
###################
library(nlme)
library(rootSolve)
library(expm)
library(lattice)
library(dplyr)
library(ramify)
library(mvtnorm)
library(parallel)
########################
# subset data function #
########################
newdat_id = function(id_num=1,clump_num=1,data=newdat_narm){
  subset(data,subset = id==id_num & clump==clump_num)
}

########################
# Quick plot functions #
########################
plot_logdos=function(data,main_title="raw data logdos"){
  plot(x=data$logdos,y=data$resp,xlab="LogDose",ylab="response",
       main=main_title)
}

plot_logdos_mean=function(data,main_title="grouped logdos"){
  xy=sortedXyData("logdos" , "resp" , data)
  plot(x=xy$x,y=xy$y,xlab="Dose",ylab="response",
       main=main_title)
}

###################
# model functions #
###################
hill=function(x,b0,b1,b2,b3){
  denum=1+exp(  (b3-x)/b2 )
  return(b0 + b1/denum)
}
hill_vec=function(b,logdos){
  return(b[1]+b[2]/(1+exp((b[4]-logdos)/b[3])))
}

hill_lse=function(b,dat){
  with(data = dat,
       return(mean((resp-hill(logdos,b[1],b[2],b[3],b[4]))^2)))
}

#####################
# initial functions #
#####################
# use max or min of sortedXyData for b0,b1 & numerical slope for b2
hill_init=function(mCall,LHS,data){ # mCall is hill function's expression
  xy = sortedXyData(mCall[["x"]],LHS,data)
  yx = sortedXyData(LHS,mCall[["x"]],data)
  if(nrow(xy)<3){ # since model is growth model , 2 distinct point doesn't make sense
    stop("Two few distinct input")
  }
  
  b0=min(xy$y);b1=max(xy$y)-b0  # b0=max(xy$)
  b3=NLSstClosestX(xy,b0+(b1/2));b3
  
  h=abs(mean(xy$x)*1e-04)
  slope=(NLSstClosestX(yx,b3+h)-NLSstClosestX(yx,b3-h))/(2*h)
  lm1=lm(resp~logdos,data=data)
  sign_lm=sign(lm1$coefficients[2])
  
  if(sign(slope)==sign_lm){
    b2=(b1)/(4*slope + 1e-8)
  } else
  {b2=(b1)/(-4*slope + 1e-8)}
  
  param=c(b0,b1,b2,b3)
  names(param)=mCall[c("b0","b1","b2","b3")] # ensure that formal parameter names and actual ones are same
  return(param)
}

hil_try=selfStart(model=hill,initial = hill_init)

hill_init2=function(mCall,LHS,data){ # mCall is hill function's expression
  xy = sortedXyData(mCall[["x"]],LHS,data)
  yx = sortedXyData(LHS,mCall[["x"]],data)
  if(nrow(xy)<3){ # since model is growth model , 2 distinct point doesn't make sense
    stop("Two few distinct input")
  }
  b0=min(xy$y);b1=max(xy$y)-b0
  
  b3=NLSstClosestX(xy,b0+(b1/2));b3
  h=abs(mean(xy$x)*1e-04)
  slope=(NLSstClosestX(yx,b3+h)-NLSstClosestX(yx,b3-h))/(2*h)
  lm1=lm(resp~logdos,data=data)
  sign_lm=sign(lm1$coefficients[2])
  
  if(sign(slope)==sign_lm){
    b2=(b1)/(4*slope + 1e-10)
  } else
  {b2=(b1)/(4*(-1)*slope + 1e-10)}
  
  param=c(b0,b1,b2,b3)
  names(param)=mCall[c("b0","b1","b2","b3")] # ensure that formal parameter names and actual ones are same
  
  lm0=summary(lm(resp~poly(logdos,3),data=data))
  if(lm0$fstatistic[1]<qf(0.95,lm0$fstatistic[2],lm0$fstatistic[3])){
    param[1:2]=param[1:2]/10
    return(param)
  } else{
    return(param) 
  }
}
hil_try2=selfStart(model=hill,initial = hill_init2)

hill_init3=function(mCall,LHS,data){ # where slope = b2
  xy = sortedXyData(mCall[["x"]],LHS,data)
  yx = sortedXyData(LHS,mCall[["x"]],data)
  if(nrow(xy)<3){ # since model is growth model , 2 distinct point doesn't make sense
    stop("Two few distinct input")
  }
  b0=min(xy$y);b1=max(xy$y)-b0
  
  b3=NLSstClosestX(xy,b0+(b1/2));b3
  h=abs(mean(xy$x)*1e-04)
  b2=(NLSstClosestX(yx,b3+h)-NLSstClosestX(yx,b3-h))/(2*h)
  lm1=lm(resp~logdos,data=data)
  sign_lm=sign(lm1$coefficients[2])
  
  if(sign(b2)==sign_lm){
    b2=b2
  } else{
    b2=-1*b2
  }
  
  param=c(b0,b1,b2,b3)
  names(param)=mCall[c("b0","b1","b2","b3")] # ensure that formal parameter names and actual ones are same
  return(param)
}
hil_try3=selfStart(model=hill,initial = hill_init3)


poly_lm=function(p=9,data,x=seq(-22,-5,by=0.5)){
  lm0=lm(resp~poly(logdos,p),data=data)
  coef=lm0$coefficients
  return(predict(lm0,newdata = list(logdos=x)))
} # polynomial regression fitted for x

var_init=function(data,p=5){
  lm1=lm(resp~poly(logdos,p),data=data)
  return(sum(lm1$residuals^2)/ lm1$df.residual)
}
#########
# step1 #
#########
mean1=function(y,x){
  mean(y[cut(x,2,labels = c(0,1))==0])
}
mean2=function(y,x){
  mean(y[cut(x,2,labels = c(0,1))==1])
}
var1=function(y,x){
  var(y[cut(x,2,labels = c(0,1))==0])
}
var2=function(y,x){
  var(y[cut(x,2,labels = c(0,1))==1])
}

left_size=function(x){
  n1=sum(cut(x,2,labels = c(0,1))==0)
  return(n1)
}

right_size=function(x){
  n2=sum(cut(x,2,labels = c(0,1))==1)
  return(n2)
}

#########
# step2 #
#########
# check only the sign of b2
optim_nls2=function(resp,logdos,trace=F,maxit=50,method="BFGS"){
  data=data.frame(cbind(resp,logdos))
  init_hill=getInitial(resp~hil_try(logdos,b0,b1,b2,b3),data = data)
  
  result=optim(par=init_hill,fn = hill_lse,dat=data,control =list(trace=trace,maxit=50),method = "BFGS")
  return(result$par[3])
}

# return all the result
optim_nls22=function(resp,logdos,trace=F,maxit=50,method="BFGS"){
  dat=data.frame(cbind(resp,logdos))
  init_hill=getInitial(resp~hil_try(logdos,b0,b1,b2,b3),data = dat)
  
  result_lst=list(optim(par=init_hill,fn = hill_lse,dat=dat
        ,control =list(trace=trace,maxit=maxit),method = method)$par)
  return(result_lst)
}
#########
# step3 #
#########
huber_score=function(x,c=1.5){
  idx=abs(x)<c
  idx2=abs(x)>=c
  
  e=numeric(length(x))
  
  e[idx]=x[idx]/sqrt(2)
  e[idx2]=sqrt(c*(abs(x[idx2])-(c/2)))
  return(e)
}

hill_hlse=function(b,data,c=1.5){
  with(data=data
       ,return(
         mean(huber_score(resp-hill_vec(b,logdos),c = c)^2)
       )
  )
}

deriv_hill=function(b,x){
  b0=b[1] ; b1=b[2] ; b2=b[3] ; b3=b[4]
  g=1+exp((b3-x)/b2)
  matrix(
    c(rep(1,length(g))
      , 1/g, b1*g^(-2)*(g-1)*(b3-x)/(b2^2)
      , (-b1/b2)*g^(-2)*(g-1))
    ,nrow = length(g),ncol = length(b)
  )
}

psi=function(x,c=1.5){
  idx=abs(x)<c
  idx2=abs(x)>=c
  
  e=numeric(length=length(x))
  
  e[idx]=x[idx]
  e[idx2]=c*sign(x[idx2])
  
  return(e)
}

psi_hat=function(x,c=1.5){
  idx=abs(x)<c
  idx2=abs(x)>=c
  
  e=numeric(length=length(x))
  
  e[idx]=1
  e[idx2]=0
  return(e)
  
}

hill_hlse2=function(par,data,c=1.5){
  b=par[1:4]
  sig=exp(par[5]) # turn it into a real sig(sigma has been parameterized)
  n=nrow(data)
  
  with(data=data
       ,return(
         sum(huber_score(x =(resp-hill_vec(b,logdos))/sig,c = c)^2) + log(sig)
       )
  )
}

GAMMA_hat=function(data,beta,sig){
  res=with(data = data,expr = resp-hill_vec(b = beta,logdos))
  n=length(res)
  epsilon=res/sig
  
  gamma1=mean(psi(epsilon)*epsilon)
  gamma2=mean(psi_hat(epsilon))
  gamma3=mean(psi_hat(epsilon)*(epsilon^2))
  sig_psi1=mean(psi(epsilon)^2)
  sig_psi2=var(psi(epsilon)*epsilon)
  
  f_theta=gradient(f = hill_vec,x = beta,logdos=data$logdos)
  GAMMA_1n=forceSymmetric(gamma2*(sig^-2)*(t(f_theta)%*%f_theta)+diag(1e-12,4))
  GAMMA_31n=forceSymmetric(sig_psi1*(sig^-2)*(t(f_theta)%*%f_theta)+diag(1e-12,4))
  
  GAMMA_32n=sig_psi2*1*n
  GAMMA_2n= n*(2*gamma1+gamma3-1 + (1-gamma1))
  
  GAMMA_3n=bdiag(GAMMA_31n,GAMMA_32n)
  GAMMA_5n=bdiag(GAMMA_1n,GAMMA_2n)
  return( solve(GAMMA_5n/n)%*%(GAMMA_3n/n)%*%solve(GAMMA_5n/n) )
}
#########
# step4 #
#########
sqrt_mat=function(X){
  X[is.na(X)]=0
  X=X+diag(1e-12,dim(X)[1])
  eigen_X=eigen(X,symmetric = TRUE)
  return(eigen_X$vectors%*%diag(sqrt(eigen_X$values))%*%t(eigen_X$vectors))
}

huber_loss=function(x,c,is.sum=F){
  idx1= abs(x)<=c & !is.na(x)
  idx2= abs(x)>c & !is.na(x)
  e=numeric(length(x))
  e[idx1]=0.5*(x[idx1]^2)
  e[idx2]=c*abs(x[idx2])-0.5*(c^2)
  if(!is.sum){
    return(e)
  } else{
    return(sum(e))
  }
}

hill_rll=function(b,u,sig,data,c){
  rep_list=unique(data$rep)
  s=length(rep_list)
  n=c()
  for(i in 1:s){
    n=c(n,nrow(data[data$rep==rep_list[i],]))
  }
  N=sum(n)
  
  n_=c(0,cumsum(n))
  W=list()
  for(i in 1:s){
    W[[i]]=gradient(hill_vec,x=b+u[(1:4) +4*(i-1)],logdos=data$logdos[(1+n_[i]):n_[i+1]])
  }
  
  D=c()
  for(i in 1:s){
    D=rbind(D,W[[i]])
  }
  
  Z=bdiag(W)
  
  variance=exp(sig)
  variance[is.na(variance)]=0
  R=diag(variance[1],N)
  G=diag(variance[2:5],4*s)
  V=Z%*%G%*%t(Z)+R
  
  til=c()
  for(i in 1:s){
    til=c(til,hill_vec(b=b+u[1:4 + 4*(i-1)],logdos=data$logdos[(1+n_[i]):n_[i+1]]))
  }
  
  y_til=as.vector(data$resp - til + D%*%b + Z%*%u)
  res=solve(sqrt_mat(V))%*%(y_til - D%*%b)
  huber=huber_loss(res,c=c)
  k=mean(psi(res,c)*res)
  return(-0.5*k*N*log(2*pi)-0.5*k*log(det(V))-0.5*sum(huber))
}

beta_var=function(b,u,sig,data,c){
  N=nrow(data)
  rep_list=unique(data$rep)
  s=length(rep_list)
  n=c()
  for(i in 1:s){
    n=c(n,nrow(data[data$rep==rep_list[i],]))
  }
  n_=c(0,cumsum(n))
  W=list()
  for(i in 1:s){
    W[[i]]=gradient(hill_vec,x=b+u[(1:4) +4*(i-1)],logdos=data$logdos[(1+n_[i]):n_[i+1]])
  }
  D=c()
  for(i in 1:s){
    D=rbind(D,W[[i]])
  }
  Z=bdiag(W)
  variance=exp(sig)
  R=diag(variance[1],N)
  G=diag(variance[2:5],4*s)
  V=forceSymmetric(Z%*%G%*%t(Z)+R)
  til=c()
  for(i in 1:s){
    til=c(til,hill_vec(b=b+u[1:4 + 4*(i-1)],logdos=data$logdos[(1+n_[i]):n_[i+1]]))
  }
  y_til=as.vector(data$resp - til + D%*%b + Z%*%u)
  res=solve(sqrt_mat(V))%*%(y_til - D%*%b)
  cov_beta=t(D)%*%forceSymmetric(solve(V))%*%D
  psi_mean=mean(psi(res,c)^2)/(mean(psi_hat(res,c))^2)
  return(psi_mean*cov_beta)
}

iter_optim_rll=function(data,beta,u,sig,c,...,is_u=TRUE,ndep_beta=rep(1e-03,4),ndep_sig=rep(1e-03,5)){
  if(is_u){
    result_beta=optim(beta,hill_rll,c=c,u=u,sig=sig,data=data,method = "BFGS"
                      ,control = list(...,ndeps=ndep_beta,fnscale=-1))
    result_sig=optim(sig,hill_rll,c=c,b=result_beta$par,u=u,data=data,method = "BFGS"
                     ,control = list(...,ndeps=ndep_sig,fnscale=-1))
    result_u=optim(u,hill_rll,c=c,b=result_beta$par,sig=result_sig$par,data=data,method = "BFGS"
                   ,control = list(...,ndeps=ndep_u,fnscale=-1))
    return(list("beta"=result_beta$par,"u"=result_u$par
                ,"sig_para"=result_sig$par,"value"=result_sig$value))
  } else{
    result_beta=optim(beta,hill_rll,c=c,u=u,sig=sig,data=data,method = "BFGS"
                      ,control = list(...,ndeps=ndep_beta,fnscale=-1))
    result_sig=optim(sig,hill_rll,c=c,b=result_beta$par,u=u,data=data,method = "BFGS"
                     ,control = list(...,ndeps=ndep_sig,fnscale=-1))
    return(list("beta"=result_beta$par,"sig_para"=result_sig$par
                ,"value"=result_sig$value))
  }
}

iter_optim_rll2=function(data,beta,u,sig,c,...,is_u=TRUE,ndep_beta=rep(1e-03,4),ndep_sig=rep(1e-03,5)){
  if(is_u){
    result_sig=optim(sig,hill_rll,c=c,b=beta,u=u,data=data,method = "BFGS"
                     ,control = list(...,fnscale=-1))
    result_beta=optim(beta,hill_rll,c=c,u=u,sig=result_sig$par,data=data,method = "BFGS"
                      ,control = list(...,ndeps=ndep_beta,fnscale=-1))
    result_u=optim(u,hill_rll,c=c,b=result_beta$par,sig=result_sig$par,data=data,method = "BFGS"
                   ,control = list(...,fnscale=-1))
    return(list("beta"=result_beta$par,"u"=result_u$par
                ,"sig_para"=result_sig$par,"value"=result_sig$value))
  } else{
    result_sig=optim(sig,hill_rll,c=c,b=beta,u=u,data=data,method = "BFGS"
                     ,control = list(...,ndeps=ndep_sig,fnscale=-1))
    result_beta=optim(beta,hill_rll,c=c,u=u,sig=result_sig$par,data=data,method = "BFGS"
                      ,control = list(...,ndeps=ndep_beta,fnscale=-1))
    return(list("beta"=result_beta$par,"sig_para"=result_sig$par,"value"=result_sig$value))
  }
}

get_initial=function(data){
  s=length(unique(data$rep))
  N=nrow(data)
  par_result=c()
  rep_list=unique(data$rep)
  for(i in rep_list){
    dat=subset(data,rep==i)
    beta_init= getInitial(resp~hil_try(logdos,b0,b1,b2,b3),data = dat)
    par_result=rbind(par_result,beta_init)
  }
  beta_init=apply(par_result,2,mean)
  u_init=(as.vector(t(par_result))-beta_init)
  init_info=optim(par = getInitial(resp~hil_try(logdos,b0,b1,b2,b3),data =data),fn = hill_lse, dat=data)
  sig_init=log(c(init_info$value/(N-4),apply(par_result,2,var)))
  sig_init=clip(sig_init,-1,5)
  c=2
  return( list(beta=beta_init,u=u_init,sig=sig_init,c=c,s=s,N=N) )
}

get_initial2=function(data){
  s=length(unique(data$rep))
  N=nrow(data)
  par_result=c()
  rep_list=unique(data$rep)
  for(i in rep_list){
    dat=subset(data,rep==i)
    beta_init= getInitial(resp~hil_try2(logdos,b0,b1,b2,b3),data = dat)
    par_result=rbind(par_result,beta_init)
  }
  beta_init=apply(par_result,2,mean)
  u_init=(as.vector(t(par_result))-beta_init)
  init_info=optim(par = getInitial(resp~hil_try2(logdos,b0,b1,b2,b3),data =data),fn = hill_lse, dat=data)
  sig_init=log(c(init_info$value/(N-4),apply(par_result,2,var)))
  sig_init=clip(sig_init,-1,5)
  c=2
  return( list(beta=beta_init,u=u_init,sig=sig_init,c=c,s=s,N=N) )
}

step4_fn=function(data,beta_init,u_init,sig_init,c,iter_max=50,is_u=F){
  data=data[order(data$rep),]
  s=length(unique(data$rep))
  N=nrow(data)
  iter_lst=list()
  print("1st")
  iter_lst[[1]]=iter_optim_rll2(data = data,beta =beta_init ,u = u_init
                                ,sig = sig_init,c = c,maxit=1,fnscale=-1)
  is_break=F
  for(i in 1:iter_max){
    iter_lst[[i+1]]=tryCatch(expr = {
      iter_optim_rll2(data,c=c,beta = iter_lst[[i]]$beta
                      ,u=iter_lst[[i]]$u,sig = iter_lst[[i]]$sig
                      ,maxit=1,fnscale=-1,ndep_beta = c(1e-4,1e-3,1e-4,1e-03))
    },error=function(e){
      print("error")
      return(list("value"=0))
    })
    if(iter_lst[[i+1]]$value==0){break}
    if(abs(iter_lst[[i+1]]$value-iter_lst[[i]]$value)<1e-3){break}
  }
  var_beta=beta_var(b = iter_lst[[i]]$beta,u = iter_lst[[i]]$u,sig = iter_lst[[i]]$sig_para,data = data,c=c)
  if(is_u){
    return( c(iter_lst[[i]]$beta,diag(var_beta),iter_lst[[i]]$u) )
  } else{
    return( c(iter_lst[[i]]$beta,diag(var_beta)) ) 
  }
}

step4_fn2=function(data,beta_init,u_init,sig_init,c,iter_max=50,is_u=F){
  data=data[order(data$rep),]
  s=length(unique(data$rep))
  N=nrow(data)
  iter_lst=list()
  print("1st")
  iter_lst[[1]]=iter_optim_rll2(data = data,beta =beta_init ,u = u_init
                                ,sig = sig_init,c = c,maxit=1,fnscale=-1)
  is_break=F
  for(i in 1:iter_max){
    
    iter_lst[[i+1]]=tryCatch(expr = {
      iter_optim_rll2(data,c=c,beta = iter_lst[[i]]$beta
                      ,u=u_init,sig = iter_lst[[i]]$sig
                      ,maxit=1,fnscale=-1,ndep_beta = c(1e-3,1e-3,1e-3,1e-03))
    },error=function(e){
      print("error")
      return(list("value"=0))
    })
    if(iter_lst[[i+1]]$value==0){break}
    if(abs(iter_lst[[i+1]]$value-iter_lst[[i]]$value)<1e-3){break}
  }
  var_beta=beta_var(iter_lst[[i]]$beta,u_init,iter_lst[[i]]$sig_para,data,c = c)

  return( c(iter_lst[[i]]$beta,diag(var_beta)) ) 
  }


get_batch=function(data,b_size=7){
  rep_list=unique(data$rep)
  s=length(rep_list) # number of replication
  rep_list_ = sample(rep_list,size = b_size,replace = F)
  return(subset(data, rep %in% rep_list_))
}

step4_fn_batch=function(data,num_batch=50,b_size=7){
  cnt_test=matrix(numeric(8*num_batch),num_batch)
  for(i in 1:num_batch){
    print(paste(i,"th iter"))
    batch_data=get_batch(data,b_size = b_size)
    s=b_size
    init=get_initial2(batch_data)
    result=tryCatch(expr = {
      step4_fn(batch_data,init$beta,init$u,init$sig,init$c,iter_max = 10)
    },error=function(e){return(rep(NA,8))})
    cnt_test[i,]=result
  }
  return(cnt_test)
}

##############
# simulation #
##############
gen_data = function(beta,sig,num_out=0,sig2,cutoff=-100){ # hill model
  # beta is should be given outside, which is a matrix with s row and 4columns
  # sig is response error and num_out is the number of outlier
  # cutoff indicate from where the outlier comes from and sig2 is outlier response's variable
  
  
  s = nrow(beta)
  x=rep(seq(from = -22,to = -8,length.out = 15),s)
  resp=numeric(s*15)
  err=rnorm(length(resp),0,sig)
  
  idx=sample(which(x>cutoff), size = num_out,replace = F)
  err[idx]=0
  err[idx]=rnorm(num_out,0,sig2)
  
  for(i in 1:s){
    resp[1:15 + 15*(i-1)] = hill_vec(beta[i,],x[1:15])
  }
  
  return(data.frame(resp=resp+err,logdos=x,rep=rep(1:s,each=15)))
}

gen_data2=function(s,sig,num_out=0,sig2){ # non toxic data
  x=rep(seq(-22,-8,length.out = 15),s)
  
  idx=sample(1:length(x),num_out,replace = F)
  resp=rnorm(s*15,0,sig)
  resp[idx]=0
  resp[idx]=rnorm(num_out,0,sig2)
  
  return(data.frame(resp=resp,logdos=x,rep=rep(1:s,each=15)))
}

Trim=function(x,trim=0.1){
  n=length(x)
  x=x[!is.na(x)]
  x0=x[order(x)]
  
  idx=ceiling(trim*n):as.integer((1-trim)*n)
  return( x0[idx] )
}

###############
# parham test #
###############

hill2=function(x,b0,b1,b2,b3){
  b0+b1*((b3^b2)/(x^b2 + b3^b2))
}

hill_vec2=function(b,x){
  b0=b[1];  b1=b[2];  b2=b[3];  b3=b[4]
  return( b0+b1*((b3^b2)/(x^b2 + b3^b2)) )
}

hill_lse2=function(b,data){
  with(data,sum( (resp - hill_vec2(beta,logdos))^2) )
}
hill2_nls=function(data,is_plot=F){
  init=getInitial(resp~hil_try(logdos,b0,b1,b2,b3),data = data)
  result=optim(init,hill_lse,dat=data)
  grad=gradient(hill_vec,x = result$par,logdos=data$logdos)
  
  sig=sum((data$resp - hill_vec(result$par,data$logdos))^2) / (nrow(data)-4)
  
  #SVD=svd(t(grad)%*%grad)
  #mD=diag(1/SVD$d)
  #mX = SVD$v %*%t(mD)%*%t(SVD$u)
  #beta_cov=sig*abs(mX)
  if(is_plot){
    plot(resp~logdos,col=rep,data=data)
    x=seq(-23,-5,by=0.25)
    lines(x,hill_vec(result$par,x))
  }
  return(list("param"=result$par,"sigma_sqaured"=sig))
}


parham_test2= function(resp,logdos,alpha=0.05){
  data=data.frame("resp"=resp,"logdos"=logdos)
  nls_result=hill2_nls(data)
  beta=nls_result$param
  x_range=range(logdos)
  
  sse0 = with( data,sum((resp-mean(resp))^2) )
  sse1 = with( data,sum((resp-hill_vec(beta,logdos))^2) )
  h0_reject =1-pchisq((sse0-sse1),3)<alpha
  
  is_active=h0_reject & beta[3]>0 & beta[4]<x_range[2] & abs(mean(data[data$logdos==x_range[2],"resp"]))>10
  if(is_active & !is.na(is_active)){
    return(1)
  }
  is_inactive = !h0_reject | beta[3]<0
  if(is_inactive){
    return(2)} else {
      return(3)}
}

#######################
huber_loss2=function(x,c,is.sum=F){
  idx1= abs(x)<=c & !is.na(x)
  idx2= abs(x)>c & !is.na(x)
  
  e=numeric(length(x))
  e[idx1]=-((c^2)/6)*((1-(x[idx1]/c)^2)^3 - 1)
  e[idx2]=(c^2)/6
  
  if(!is.sum){
    return(e)
  } else{
    return(sum(e))
  }
}
psi2=function(x,c=1.5){
  idx=abs(x)<c
  idx2=abs(x)>=c
  
  e=numeric(length=length(x))
  
  e[idx]=x[idx]*((1-(x/c)^2)^2)
  e[idx2]=0
  
  return(e)
}
#x=seq(-4,4,0.25)
#plot(x,x^2,type="l")
#lines(x,huber_loss2(x,4),lty=2,lwd=1.5)
#lines(x,huber_loss(x,2),lty=4,lwd=1.5)
