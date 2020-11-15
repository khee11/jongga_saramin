source("helpers2.R")
###############
# create data #
###############
#### simulation active=0.15 ###
n=10000
act_rate = 0.15
ratio=c(n*act_rate,n*(1-act_rate))

x=seq(-22,-8,length.out = 15)

mean_of_beta=c(-2.208040,  36.48949, 1.158922, -13.560712)
beta_cov=diag(c(2,120,1,2))
set.seed(100)
S=sample(c(1,2,3,5,7),size=n,replace=T,prob=c(2,1,5,1,1)/10) # for computation efficiecy, no multiple_rep
num_out=clip(rpois(n,2),0,4)#sample(3:7,n,replace = T)# # num of outlier
sig_candid = sample(c(4,6,8,15,20),n,replace = T,prob=c(2,3,3,1.5,0.5)/10)
random_var=sample(seq(0.1,0.2,by = 0.1),n,prob=c(3,7)/10,replace = T)

active_df_list=list()
active_df_info_mat =matrix(numeric(ratio[1]*7),ratio[1],7)

for(i in 1:ratio[1]){
  beta_mean = rmvnorm(1,mean=mean_of_beta,sigma=beta_cov)
  beta_mean[2]=clip(beta_mean[2],30,Inf)
  beta_mean[3]=clip(beta_mean[3],0.2,1.5)
  beta_mean[4]=clip(beta_mean[4],-22,-8)
  #beta_mean
  sig=sig_candid[i] #sig_candid[i]#
  num=num_out[i]
  s=S[i]
  ran_var=random_var[i]
  
  beta=rmvnorm(s,beta_mean,sigma = diag(c(abs(beta_mean*ran_var)),4))
  df=gen_data(beta,sig,num,sig*3,cutoff = -100)
  df$case=i
  
  active_df_list[[i]]=df
  active_df_info_mat[i,]=c(beta_mean,num,sig,ran_var)
}
active_df = do.call(rbind,active_df_list)
active_df_info = data.frame(active_df_info_mat)
names(active_df_info)=c(paste("b",0:3,sep=""),"outlier","sig","ran_var")

inactive_df_info_mat =matrix(numeric(ratio[2]*7),ratio[2],7)
inactive_df_list=list() # inactive data
for(i in 1:ratio[2]){
  s=S[ratio[1]+i]
  sig=sig_candid[ratio[1]+i]
  num=num_out[ratio[1]+i]
  
  df=gen_data2(s,sig,num,sig*3)
  df$case=ratio[1]+i
  inactive_df_list[[i]]=df
  #inactive_df_info_mat[i-2500,]=c(rep(NA,4),sig,num,NA)
}
inactive_df = do.call(rbind,inactive_df_list)
inactive_df_info = data.frame(inactive_df_info_mat)
names(inactive_df_info)=c(paste("b",0:3,sep=""),"outlier","sig","ran_var")

##############
# concat all #
##############
nrow(summarise(group_by(active_df,case)))
nrow(summarise(group_by(inactive_df,case)))
#nrow(summarise(group_by(inconclu_df,case)))

newdat_simul2 = rbind(active_df,inactive_df)
length(unique(newdat_simul2$case))

#newdat_simul_info2 = rbind(rbind(active_df_info,inactive_df_info),inconclu_df_info)
#dim(newdat_simul_info2)

###################
# save result mat #
###################
simul_result2=matrix(logical(n*4),n,4)
dim(simul_result2)

##############
# Parham run #
##############
ratio
start=Sys.time()
parham_result22=summarise(group_by(newdat_simul2,case)
                          ,"test"=parham_test2(resp,logdos,0.05/n))
table(parham_result22$test)
end=Sys.time()
print(end-start)

sum(which(parham_result22[,"test"]==1) %in% 1:ratio[1])/ratio[1] # number of cases which predicted active and belong to true active
sum(which(parham_result22[,"test"]==1) %in% (ratio[1]+1):(ratio[1]+ratio[2]))/sum(parham_result22[,"test"]==1)  # FDR

##############
# step 1 run #
##############

#### compute unique cases, which are seperated by (id,clump)
unique_case=summarise(group_by(newdat_simul2,case))
nrow(unique_case)

### between ttest is a good idea but not influential
ttest_simul2=summarise(group_by(newdat_simul2,case),mean=mean(resp),sd=sd(resp),n=length(resp)
                       ,mean1=mean1(resp,logdos),mean2=mean2(resp,logdos)
                       ,var1=var1(resp,logdos),var2=var2(resp,logdos))
p_threshold2=0.05
df=with(ttest_simul2,(2*var1/n + 2*var2/n)^2 / ((2*var1/n)^2/(n/2 -1) + (2*var2/n)^2/(n/2 -1)))

# check if mu>0
t_whole_idx = with(ttest_simul2, 2*pt(q = sqrt(n)*abs(mean)/sd ,df = n-1,lower.tail = F))<p_threshold2
sum(t_whole_idx) # number of rejected : the result equal to the paper

# check if both mean difference and mu>0 TRUE
step1_idx=as.logical(t_whole_idx)
sum(step1_idx) # number of rejected
sum(!step1_idx)
unique_case2=unique_case[step1_idx,] # indices moving on to step2

length(which(step1_idx==F))

simul_result2[,1]=step1_idx # *** assign index

### the number of the rejected in active and inactive
sum(which(step1_idx==T) %in% 1:ratio[1])
sum(which(step1_idx==T) %in% (ratio[1]+1):ratio[2])




##############
# step 2 run #
##############

## get step2 data
newdat_step2=merge(newdat_simul2,unique_case2,by = "case",all.y = T)
nrow(summarise(group_by(newdat_step2,case))) # check if merged correctly

###################
# carry out step2 #
###################

# newdat_step2=merge(newdat_narm,as.data.frame(unique_case2)) # unique(newdat_step2[c("id","clump")])
step2_nls=summarise(group_by(newdat_step2,case),nls=optim_nls2(resp,logdos,maxit = 100,trace = F))
sum(step2_nls$nls>=0) # number of cases move onto step3
sum(step2_nls$nls>=0)/nrow(step2_nls)
step2_idx=step2_nls$nls>=0
unique_case3=unique_case2[step2_idx,] # moving onto step3

#step2_nls[step2_nls$nls>=0,]$case
sum(which(apply(simul_result2[,1:2],1,all)==T) %in% 1:ratio[1])
sum(which(apply(simul_result2[,1:2],1,all)==T) %in% (1+ratio[1]):ratio[2])
simul_result2[step2_nls[step2_nls$nls>=0,]$case,2]=T





##############
# step 3 run #
##############

# sort out data
newdat_step3=merge(newdat_step2,unique_case3,by="case",all.y = T)
newdat_step3$max_rep=NULL
m.rep_idx=summarise(group_by(newdat_step3,case),max_rep=length(unique(rep)))
temp_newdat=merge(newdat_step3,m.rep_idx,by="case",all.y = T)
newdat_step4_m=subset(temp_newdat,max_rep>1)
newdat_step4_s=subset(temp_newdat,max_rep==1)
rm(newdat_step3)

length((unique(newdat_step4_m[,"case"])))
length((unique(newdat_step4_s[,"case"])))

# carry out step3
step3_idx=unique(newdat_step4_s["case"])
cnt=c()
error_list=c()
x=seq(-22,-5,by=0.5)
#plot_dir="D:\\students\\ljg\\JONGGA\\simul1\\step3\\"
for(i in 1:nrow(step3_idx)){
  print(c(i,unlist(step3_idx[i,])))
  data=subset(newdat_step4_s, case==step3_idx$case[i])
  beta_init=getInitial(resp~hil_try3(logdos,b0,b1,b2,b3),data=data)
  sig_init=log(sqrt(var_init(data)))
  result=optim(par=c(beta_init,sig_init),fn=hill_hlse2,data=data,c=1.5
               ,control = list(maxit=10000,abstol=1e-08),method = "BFGS")
  result2=optim(par=beta_init,fn=hill_lse,dat=data,method = "BFGS")
  
  tryCatch(expr = {
    std=sqrt(diag(GAMMA_hat(data,beta = result$par[1:4],sig = result$par[5]))/n)
  },error=function(e){
    error_list=c(error_list,i)
    std=NULL
  }
  )
  cnt=rbind(cnt,c(unlist(step3_idx[i,]),result$par,std))
  
  #jpeg(filename = paste(plot_dir,step3_idx$case[i],".jpeg",sep=""))
  #plot_logdos(data)
  #lines(x,hill_vec(b = cnt[i,c("b0","b1","b2","b3")],x))
  #lines(x,hill_vec(b=result2$par,x),col=2)
  #dev.off()
}
colnames(cnt)=c("case","b0","b1","b2","b3","sig","sd_b0","sd_b1","sd_b2","sd_b3","sd_sig")
#options(scipen = 999)
sum(is.na(cnt[,"sd_b1"]),na.rm = T)
cnt_try=cnt

# check the result
NA_idx=is.na(apply(cnt_try[,8:11],1,sum))
sum(NA_idx)/dim(cnt_try)[1] # the ratio of NA(at least one of se of betas)
sum(abs(cnt_try[!NA_idx,"b1"])/cnt_try[!NA_idx,"sd_b1"]>qt(0.95,df=11) &
      cnt_try[!NA_idx,"b2"]>0)/(dim(cnt_try)[1]) # ratio of b1>0 : 40.22% /41.32%

# record the result
step3_idx=abs(cnt_try[,"b1"])/cnt_try[,"sd_b1"]>qt(0.95,df=11) & cnt_try[,"b2"]>0
sum(step3_idx)
sum(!step3_idx)

simul_result2[cnt_try[step3_idx,1],3]=T


##############
# step 4 run #
##############
# choose some cases for step4_mixed
idx=unlist(unique(newdat_step4_m[c("case")]))
length(idx)

cl=makeCluster(8)
clusterExport(cl,ls()[!(ls() %in% c("newdat_simul2","newdat_step2","newdat_simul_info2"
                                    ,"newdat_step4_s","inactive_df","active_df","inconclu_df"))]) # remove some heavy objects
start=Sys.time()
step4_result2=parSapply(cl = cl,X = 1:length(idx) ,function(x){
  library(ramify)
  library(rootSolve)
  library(expm)
  #print(x)
  newdat_id1=subset(newdat_step4_m,case==idx[x])
  init=get_initial2(newdat_id1)
  result2=tryCatch(expr = {
    c(idx[x],step4_fn(newdat_id1,init$beta,init$u,init$sig,init$c))
  },error=function(e){return(c(unlist(idx[x]),rep(NA,8)))})
  return(unlist(result2))
})
stopCluster(cl)
end=Sys.time()
step4_result= step4_result2 # keep the result

print(end-start)

# number of NA
error_list=numeric(dim(step4_result)[2])
for(i in 1:dim(step4_result)[2]){
  error_list[i]=!is.na(sum(step4_result[,i]))
}
sum(!error_list) # error

# remove na cases
step4_result=step4_result[,which(as.logical(error_list))]

# summarise the result
mean(abs(step4_result[3,])/(step4_result[7,])>1.96 & step4_result[4,]>0,na.rm=T)
active_idx=abs(step4_result[3,])/(step4_result[7,])>1.96 & step4_result[4,]>0
sum(!active_idx)
sum(active_idx)

# record the result
simul_result2[t(step4_result)[active_idx,1],4]=T

# power and fdr
sum(which(apply(simul_result2[,3:4],1,any)) %in% 1:ratio[1])/ratio[1] # power
mean(which(apply(simul_result2[,3:4],1,any)) %in% (1+ratio[1]):(ratio[1]+ratio[2])) # FDR
