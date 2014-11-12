sae.train <- function(x,hidden=c(10),
                      activationfun="sigm",
                      learningrate=0.8,
                      momentum=0.5,
                      learningrate_scale=1,
                      output="sigm",
                      numepochs=3,batchsize=100,
                      hidden_dropout=0,visible_dropout=0.2,L2=0,L1=0,
                      momentummax=.99,momentum_scale=1,
                      adadelta=0){
  #if (!is.matrix(x)) 
  #  stop("x must be a matrix!")
  input_dim <- ncol(x)
  size <- c(input_dim, hidden)
  sae <- list(
    input_dim = input_dim,
    hidden = hidden,
    size = size
  )
  train_x <- x
  message(sprintf("training layer 1 autoencoder ..."))
  sae$encoder[[1]] <-  nn.train(train_x,train_x,hidden=c(hidden[1]),
                                activationfun=activationfun,
                                learningrate=learningrate,
                                momentum=momentum,
                                learningrate_scale=learningrate_scale,
                                output=output,
                                numepochs=numepochs,batchsize=batchsize,
                                hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,
                                L2=L2,L1=L1,
                                momentummax=momentummax,momentum_scale=momentum_scale,
                                adadelta=adadelta)
  
  if(length(sae$size) > 2){
    for(i in 2:(length(sae$size) - 1)){
      pre <- t( sae$encoder[[i-1]]$W[[1]] %*% t(train_x) + sae$encoder[[i-1]]$B[[i-1]] )
      if(sae$encoder[[i-1]]$activationfun[i-1] == "sigm"){
        post <- sigm( pre )
      }else if(sae$encoder[[i-1]]$activationfun[i-1] == "binsigm"){
        post <- binsigm( pre )
      }else if(sae$encoder[[i-1]]$activationfun[i-1] == "tanh"){
        post <- tanh(pre)
      }else{
        stop("unsupport activation function 'nn$activationfun'");
      }  
      train_x <- post
      message(sprintf("training layer %d autoencoder ...",i))
      sae$encoder[[i]] <- nn.train(train_x,train_x,hidden=c(hidden[i]),
                                   activationfun=activationfun,
                                   learningrate=learningrate,
                                   momentum=momentum,
                                   learningrate_scale=learningrate_scale,
                                   output=output,
                                   numepochs=numepochs,batchsize=batchsize,
                                   hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,
                                   L2=L2,L1=L1,
                                   momentummax=momentummax,momentum_scale=momentum_scale,
                                   adadelta=adadelta)
    }
  }
  sae
}

sae.dnn.train <- function(x,y,hidden=c(10),
                          activationfun="sigm",
                          learningrate=0.8,
                          momentum=0.5,
                          learningrate_scale=1,
                          output="sigm",
                          sae_output="linear",
                          numepochs=3,batchsize=100,
                          hidden_dropout=0,visible_dropout=0,L2=0,L1=0,
                          momentummax=.99,momentum_scale=1,
                          adadelta=0){
  output_dim <- 0
  if(is.vector(y)){
    output_dim <- 1
  }else{ # if(is.matrix(y)){
    output_dim <- ncol(y)
  }
  if (output_dim == 0) 
    stop("y must be a vector or matrix!")
  message("begin to train sae ......")
  sae <- sae.train(x,hidden=hidden,
                   activationfun=activationfun,
                   output=sae_output,
                   numepochs=numepochs,batchsize=batchsize,
                   learningrate=learningrate,learningrate_scale=learningrate_scale,
                   momentum=momentum,
                   hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,L2=L2,L1=L1,
                   momentummax=momentummax,momentum_scale=momentum_scale,
                   adadelta=adadelta)
  message("sae has been trained.")
  initW <- list()
  initB <- list()
  for(i in 1:(length(sae$size) - 1)){
    initW[[i]] <- sae$encoder[[i]]$W[[1]]   
    initB[[i]] <- sae$encoder[[i]]$B[[1]]  
  }
  #random init weights between last hidden layer and output layer
  last_hidden <- sae$size[length(sae$size)]
  initW[[length(sae$size)]] <- matrix(runif(output_dim*last_hidden,min=-0.1,max=0.1), c(output_dim,last_hidden))
  initB[[length(sae$size)]] <- runif(output_dim,min=-0.1,max=0.1)
  message("begin to train deep nn ......")
  dnn <- nn.train(x,y,initW=initW,initB=initB,hidden=hidden,
                  activationfun=activationfun,
                  learningrate=learningrate,
                  momentum=momentum,
                  learningrate_scale=learningrate_scale,
                  output=output,
                  numepochs=numepochs,batchsize=batchsize,
                  hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,L2=L2,L1=L1,
                  momentummax=momentummax,momentum_scale=momentum_scale,
                  adadelta=adadelta)
  message("deep nn has been trained.")
  dnn
}


sigm <- function(x){
  1/(1+exp(-x))
}

binsigm <- function(x,t=0.5){
  1*(sigm(x)>t)
}

nn.train <- function(x,y,initW=NULL,initB=NULL,hidden=c(10),
                     activationfun="sigm",
                     learningrate=0.8,
                     momentum=0.5,
                     learningrate_scale=1,
                     output="sigm",
                     numepochs=3,batchsize=100,
                     hidden_dropout=0,visible_dropout=0,L2=0,L1=0,
                     momentummax=.99,momentum_scale=1,
                     adadelta=0) {
  #if (!is.matrix(x)) 
  # stop("x must be a matrix!")
  input_dim <- ncol(x)
  output_dim <- 0;
  if(is.vector(y)){
    output_dim <- 1
  }else{ # if(is.matrix(y)){
    output_dim <- ncol(y)
  }
  if (output_dim == 0) 
    stop("y must be a vector or matrix!")
  size <- c(input_dim, hidden, output_dim)
  vW <- list() 
  vB <- list()
  if(is.null(initW) || is.null(initB)){
    W <- list()
    B <- list()
    #random init weights and bias between layers							 
    for( i in 2:length(size) ){
      W[[i-1]] <- matrix(runif(size[i]*size[i-1],min=-0.1,max=0.1), c(size[i],size[i-1]));
      B[[i-1]] <- runif(size[i],min=--0.1,max=0.1);
      vW[[i-1]] <- matrix(rep(0,size[i]*size[i-1]),c(size[i],size[i-1]))
      vB[[i-1]] <- rep(0,size[i])
    }
  }else{
    W <- initW
    B <- initB
    for( i in 2:length(size) ){
      vW[[i-1]] <- matrix(rep(0,size[i]*size[i-1]),c(size[i],size[i-1]))
      vB[[i-1]] <- rep(0,size[i])
      if(nrow(W[[i-1]]) != size[i] || ncol(W[[i-1]]) != size[i-1] ){
        stop("init W size is not eq to network size!")  
      }    
      if(length(B[[i-1]]) != size[i]){
        stop("init B size is not eq to network size!")  
      }
    }
  }
  
  if(length(activationfun)==1){
    activationfun=rep(activationfun,length(size)-1)  
  }
  
  nn <- list(
    input_dim = input_dim,
    output_dim = output_dim,
    hidden = hidden,
    size = size,
    activationfun = activationfun,
    learningrate = learningrate,
    momentum = momentum,
    learningrate_scale = learningrate_scale,
    hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,
    output = output,
    W = W,
    vW = vW,
    rW = vW,
    hW = vW,
    B = B,
    vB = vB,
    rB = vB,
    hB = vB,
    L2=L2,
    L1=L1,
    momentummax=momentummax,
    momentum_scale=momentum_scale,
    adadelta=adadelta
  )
  
  m <- nrow(x);
  numbatches <- m / batchsize;
  s <- 0
  for(i in 1:numepochs){
    randperm <- sample(1:m,m)
    if(numbatches >= 1){
      for(l in 1 : numbatches){
        s <- s + 1
        batch_x <- x[randperm[((l-1)*batchsize+1):(l*batchsize)], ] 
        if(is.vector(y)){
          batch_y <- y[randperm[((l-1)*batchsize+1):(l*batchsize)]] 
        }else{ # if(is.matrix(y)){
          batch_y <- y[randperm[((l-1)*batchsize+1):(l*batchsize)], ] 
        }
        nn <- nn.ff(nn,batch_x,batch_y,s)
        nn <- nn.bp(nn)						
      }
    }
    #last fraction of sample
    if(numbatches > as.integer(numbatches)){
      batch_x <- x[randperm[(as.integer(numbatches)*batchsize):m], ]
      if(is.vector(y)){
        batch_y <- y[randperm[(as.integer(numbatches)*batchsize):m]]
      }else{ # if(is.matrix(y)){
        batch_y <- y[randperm[(as.integer(numbatches)*batchsize):m], ]
      }
      s <- s + 1
      nn <- nn.ff(nn,batch_x,batch_y,s)
      nn <- nn.bp(nn)	
    }
    
    nn$learningrate <- nn$learningrate * nn$learningrate_scale;
    nn$momentum <- min(nn$momentummax,nn$momentum^nn$momentum_scale)
    
  }
  
  nn
}

nn.ff <- function(nn,batch_x,batch_y,s){
  m <- nrow(batch_x)
  #do input dropout
  if(nn$visible_dropout > 0){
    nn$dropout_mask[[1]] <- dropout.mask(ncol(batch_x),nn$visible_dropout)
    batch_x <- t( t(batch_x) * nn$dropout_mask[[1]] )
  }
  nn$post[[1]] <- batch_x
  for(i in 2:(length(nn$size) - 1)){
    nn$pre[[i]] <- t( nn$W[[i-1]] %*% t(nn$post[[(i-1)]])  + nn$B[[i-1]] )
    if(nn$activationfun[i-1] == "sigm"){
      nn$post[[i]] <- sigm(nn$pre[[i]])
    }else if(nn$activationfun[i-1]=='binsigm'){
      nn$post[[i]] <- binsigm(nn$pre[[i]])
    }else if(nn$activationfun[i-1] == "tanh"){
      nn$post[[i]] <- tanh(nn$pre[[i]])
    }else{
      stop("unsupport activation function!");
    }	
    if(nn$hidden_dropout > 0){
      nn$dropout_mask[[i]] <- dropout.mask(ncol(nn$post[[i]]),nn$hidden_dropout)
      nn$post[[i]] <- t( t(nn$post[[i]]) * nn$dropout_mask[[i]] )
    }
  }
  #output layer
  i <- length(nn$size)
  nn$pre[[i]] <- t( nn$W[[i-1]] %*% t(nn$post[[(i-1)]])  + nn$B[[i-1]] )
  if(nn$output == "sigm"){
    nn$post[[i]] <- sigm(nn$pre[[i]])
    nn$e <- batch_y - nn$post[[i]]
    nn$L[ s ] <- 0.5*sum(nn$e^2)/m
  }else if(nn$output == "linear"){
    nn$post[[i]] <- nn$pre[[i]]
    nn$e <- batch_y - nn$post[[i]]
    nn$L[ s ] <- 0.5*sum(nn$e^2)/m
  }else if(nn$output == "softmax"){
    nn$post[[i]] <- exp(nn$pre[[i]])
    nn$post[[i]] <- nn$post[[i]] / rowSums(nn$post[[i]]) 
    nn$e <- batch_y - nn$post[[i]]
    nn$L[ s ] <- -sum(batch_y * log(nn$post[[i]]))/m
  }else{
    stop("unsupport output function!");
  }	
  if(s %% 10000 == 0){
    message(sprintf("####loss on step %d is : %f",s,nn$L[ s ]))
  }
  
  nn
}


nn.bp <- function(nn){
  n <- length(nn$size)
  d <- list()
  if(nn$output %in% c("sigm","binsigm")){
    d[[n]] <- -nn$e * (nn$post[[n]] * (1 - nn$post[[n]]))
  }else if(nn$output == "linear" || nn$output == "softmax"){
    d[[n]] <- -nn$e
  }
  
  for( i in (n-1):2 ){
    if(nn$activationfun[i-1] %in% c('binsigm',"sigm")){
      d_act <- nn$post[[i]] * (1-nn$post[[i]])
    }else if(nn$activationfun[i-1]  == "tanh" ){
      d_act <- 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn$post[[i]]^2)
    }
    d[[i]] <- (d[[i+1]] %*% nn$W[[i]]) * d_act
    if(nn$hidden_dropout > 0){
      d[[i]] <- t( t(d[[i]]) * nn$dropout_mask[[i]] )
    }
  }
  
  for( i in 1:(n-1) ){
    dw <- t(d[[i+1]]) %*% nn$post[[i]] / nrow(d[[i+1]])
    dw <- dw * nn$learningrate
    if(nn$adadelta>0){
      nn$rW[[i]]=nn$adadelta*dw^2 + (1-nn$adadelta)*nn$rW[[i]]
      dw <- dw*sqrt(nn$hW[[i]]+1e-6)/sqrt(nn$rW[[i]]+1e-6)
      nn$hW[[i]]=nn$adadelta*dw^2 + (1-nn$adadelta)*nn$hW[[i]]
    }
    if(nn$momentum > 0){
      nn$vW[[i]] <- nn$momentum * nn$vW[[i]] + dw
      dw <- nn$vW[[i]]
    }
    #nn$W[[i]] <- nn$W[[i]] - dw
    #nn$W[[i]] <- nn$W[[i]]*(1-nn$learningrate*nn$L2/nrow(d[[i+1]])) - sign(nn$W[[i]])*nn$learningrate*nn$L1/nrow(d[[i+1]]) - dw  #add regularization
    nn$W[[i]] <- nn$W[[i]]*(1-nn$learningrate*nn$L2) - dw #L2 and gradient
    #nn$W[[i]] <- nn$W[[i]] - sign(nn$W[[i]])*pmin(abs(nn$W[[i]]),nn$learningrate*nn$L1)  #L1
    if(nn$L1>0){
      nn$W[[i]] <- sign(nn$W[[i]])*pmax(abs(nn$W[[i]])-nn$learningrate*nn$L1,0)  #L1
    }
    db <- colMeans(d[[i+1]])
    db <- db * nn$learningrate
    if(nn$adadelta>0){
      nn$rB[[i]]=nn$adadelta*db^2 + (1-nn$adadelta)*nn$rB[[i]]
      db <- db*sqrt(nn$hB[[i]]+1e-6)/sqrt(nn$rB[[i]]+1e-6)
      nn$hB[[i]]=nn$adadelta*db^2 + (1-nn$adadelta)*nn$hB[[i]]
    }
    if(nn$momentum > 0){
      nn$vB[[i]] <- nn$momentum * nn$vB[[i]] + db
      db <- nn$vB[[i]]
    }
    nn$B[[i]] <- nn$B[[i]] - db
  }
  nn
}

dropout.mask <- function(size,fraction){
  mask <- runif(size,min=0,max=1)
  mask[mask <= fraction] <- 0
  mask[mask > fraction] <- 1
  mask
}

nn.predict <- function(nn,x){
  m <- nrow(x)
  post <- x
  #hidden layer
  for(i in 2:(length(nn$size) - 1)){
    pre <- t( nn$W[[i-1]] %*% t(post) + nn$B[[i-1]] )
    if(nn$activationfun[i-1] == "sigm"){
      post <- sigm( pre )
    }else if(nn$activationfun[i-1] == "binsigm"){
      post <- binsigm( pre )
    }else if(nn$activationfun[i-1] == "tanh"){
      post <- tanh(pre)
    }else{
      stop("unsupport activation function 'nn$activationfun'");
    }	
    post <- post * (1 - nn$hidden_dropout)
  }
  #output layer
  i <- length(nn$size)
  pre <- t( nn$W[[i-1]] %*% t(post) + nn$B[[i-1]] )
  if(nn$output == "sigm"){
    post <- sigm( pre )
  }else if(nn$output == "linear"){
    post <- pre  
  }else if(nn$output == "softmax"){
    post <- exp(pre)
    post <- post / rowSums(post) 
  }	else{
    stop("unsupport output function!");
  }	
  post
}

##' Test new samples by Trainded NN
##'
##' Test new samples by Trainded NN,return error rate for classification
##' @param nn nerual network trained by function nn.train
##' @param x new samples to predict
##' @param y new samples' label
##' @param t threshold for classification. If nn.predict value >= t then label 1,else label 0
##' @return error rate
##' @examples
##' Var1 <- c(rnorm(50,1,0.5),rnorm(50,-0.6,0.2))
##' Var2 <- c(rnorm(50,-0.8,0.2),rnorm(50,2,1))
##' x <- matrix(c(Var1,Var2),nrow=100,ncol=2)
##' y <- c(rep(1,50),rep(0,50))
##' nn <-nn.train(x,y,hidden=c(5))
##' test_Var1 <- c(rnorm(50,1,0.5),rnorm(50,-0.6,0.2))
##' test_Var2 <- c(rnorm(50,-0.8,0.2),rnorm(50,2,1))
##' test_x <- matrix(c(test_Var1,test_Var2),nrow=100,ncol=2)
##' err <- nn.test(nn,test_x,y)
##' 
##' @author Xiao Rong
##' @export
nn.test <- function (nn,x,y,t=0.5){
  y_p <- nn.predict(nn,x)
  m <- nrow(x)
  y_p[y_p>=t] <- 1
  y_p[y_p<t] <- 0
  error_count <- sum(abs( y_p - y)) / 2
  error_count / m
}

unfoldsae=function(x,sae,
                   activationfun="sigm",
                   codeactivationfun='sigm',
                   learningrate=0.8,
                   momentum=0.5,
                   learningrate_scale=1,
                   output="linear",
                   numepochs=3,batchsize=100,
                   hidden_dropout=0.5,visible_dropout=0,L2=0,L1=0,
                   momentummax=.99,momentum_scale=1,
                   adadelta=0){
  output_dim=sae$size[1]
  initW <- list()
  initB <- list()
  for(i in 1:(length(sae$size) - 1)){
    initW[[i]] <- sae$encoder[[i]]$W[[1]]   
    initB[[i]] <- sae$encoder[[i]]$B[[1]]  
  }
  nl=length(initW)
  j=nl+1
  for(i in seq(nl,1,by=-1)){
    initW[[j]] <- sae$encoder[[i]]$W[[2]]
    #initW[[j]] <- t(initW[[i]])
    if(i>1){
      initB[[j]] <- sae$encoder[[i]]$B[[2]]
      #initB[[j]] <- initB[[i-1]]
    }else{
      initB[[j]] <- runif(output_dim,min=-0.1,max=0.1)
    }
    j=j+1
  }
  hid=c(sae$hidden,rev(sae$hidden)[-1])
  if(length(activationfun)==1){
    activationfun=rep(activationfun,length(hid)+1)
    activationfun[nl]=codeactivationfun
  }
  dnn <- nn.train(x,x,initW=initW,initB=initB,hidden=hid,
                  activationfun=activationfun,
                  learningrate=learningrate,
                  momentum=momentum,
                  learningrate_scale=learningrate_scale,
                  output=output,
                  numepochs=numepochs,batchsize=batchsize,
                  hidden_dropout=hidden_dropout,visible_dropout=visible_dropout,L2=L2,L1=L1,
                  momentummax=momentummax,momentum_scale=momentum_scale,
                  adadelta=adadelta)
  dnn
}