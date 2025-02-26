library(Matrix)
normalizeF = function(v) {
  temp=v-mean(v)
  #temp=v
  temp=(temp)/sqrt(sum(temp^2))
  return(temp)
}
LARSf = function(normX,normY){
  p=ncol(normX)                               #num of predictors
  n=nrow(normX)                               #num of observations
  residualbeta=rep(0,p)                       #store beta for each step 
  currentbeta1=rep(0,p)                       #store  the needed direction for predictors in active set
  activeset=c()                               #store which predictors join in active set
  rankX=rankMatrix(normX)                     #how much step we want in fitting LARS 
  betaoutput=matrix(0,nrow=rankX-1,ncol = p)  #store beta for all steps.
  ones=rep(1,p)                               #a vector with only value 1 for finding current correlation with vectors "not" in active set
  
  for (i in 1:(rankX-1)){
    #find X in lars step
    residual=normY - normX %*% residualbeta                                         #current residual for this step
    cortemp = t(normX) %*% (residual)                                               #current correlation for each predictor to the current residual
    currentbeta1=sign(cortemp)                                                      #find the sign of the correlation for the direction of predictors in active set.
    cortemp[activeset]=0                                                            #leave the predictors in active set out of selection list
    chooseX=order(abs(cortemp),decreasing = T)[1]                                   #find the predictor with largest correlation as the chosen predictor in this step
    activeset[i]=chooseX                                                            #record the chosen predictor in this step
    
    #compute equanigular vector and its combination of original X
    currentbeta1[-activeset]=0                                                      #
    activeX=normX[,activeset]*rep(currentbeta1[activeset],each=n)                   #form a design matrix with all predictors with marked direction and whether in active set
    activeones=rep(1,length(activeset))                                             #a vector with only value 1 for equiangular vector calculating                                                                         #
    equbeta = solve(t(activeX)%*%activeX) %*% activeones                            #solve how to form equiangular vector
    equVector= activeX %*% equbeta                                                  #calculate the equiangular vector
    equbeta2=rep(0,p)                                                               #enlarge the "equiangular combination" for beta for all predictors needed
    for (j in 1:length(activeset)) {                                                #
      equbeta2[activeset[j]] = equbeta[j]*currentbeta1[activeset[j]]                #
    }                                                                               #
    
    #solve the length of active set
    #currentcor = X'(Y-X*beta) = X'Y - X'X*(beta_r+b*beta_c1) solve b with X only not in active set
    r=t(normX) %*% residual                                                         #correlation for all predictor to the current residual
    q=t(normX) %*% equVector                                                        #correlation for all predictor to the equiangular vector
    b1=ifelse(((abs(q)-abs(q[chooseX])))!=0,(r-r[chooseX])/(q-q[chooseX]),0)        #possible length for the equiangular vector
    b2=ifelse(((abs(q)-abs(q[chooseX])))!=0,(r+r[chooseX])/(q+q[chooseX]),0)        #possible length for the equiangular vector
    btotal=c(b1[-activeset],b2[-activeset])                                         #combine all the possible solutions together
    btotal=ifelse(btotal<=0,max(btotal),btotal)                                     #because the predictors in the active set happen the minimum, we need to leave them out "set as maximum
    b=min(btotal)                                                                   #the minimum of all possible solution "how long we need for the equiangular vector
    residualbeta = residualbeta + b * equbeta2                                      #the beta fitted in this lars step 
    betaoutput[i,]= residualbeta                                                    #store the beta in this step
  }
  lastresidual= normY - normX %*% residualbeta                                      #the last residual in the last LARS step
  return(list("activeset"=activeset,"beta"=betaoutput,"lastresidual"=lastresidual,  #output store
              "lastequbeta"=equbeta2, "lastequVector"=equVector))                   #
}
LARSMainFunction = function(X,Y,normalizeF,LARSf){
  normY=Y-mean(Y)
  normX=apply(X,2,normalizeF)
  return(LARSf(normX,normY))
}
