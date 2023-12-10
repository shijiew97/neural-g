#' @title
#' Bivariate Net-NPMLE
#'
#' @description
#' Train Bivariate Net-NPMLE (Neural-Net NPMLE).
#' python(>=3.7) and pytorch
#' are needed to be installed in advance. R pakcage 'reticulate' is also required.
#'
#' @param X predictor Variable.
#' @param Y response Variable.
#' @param dist probabilistic distribution of Y, including:
#' "Gaussian2", "Gaussian2s".
#' @param n_grid number of support grids.
#' @param num_it number of iterations for training.
#' @param param nuisance parameter value. param = 0 indicates there is no nuisance parameter.
#' @param hidden_size number of hidden neurons at each layer.
#' @param L number of hidden layers.
#' @param n number of sample size.
#' @param p the dimension of predictor Variable;should be > 1.
#' @param n0 number of mini-batch size;Default is equal to n.
#' @param lr0 learning rate, default is 0.0001.
#' @param lrDecay lrDecay = 1: using decaying learning rate.
#' @param lrpower decay rate of learning rate, default is 0.2.
#' @param gpu_ind gpu index.
#' @param verb print information while training generator.
#' @usage
#' Net_NPMLE(X=NULL, Y, param=1, dist=NULL, hidden_size=500,
#' num_it=NULL, p=1, gpu_ind=0, L=5, n_grid=100,
#' lr=0.0001, n0=n, n=NULL, lrDecay=0, lrpower=0.2,
#' verb=1)
#'
#' @return
#' Net_NPMLE function returns a list with support grids (''support'') and estimated probability mass (''prob'')
#'
#' @author
#' Shijie Wang, Chakraborty Saptarshi, Qin Qian, Ray Bai
#' @export
#' @examples
#' ### Gasussian discrete
#' Seed <- 128783;set.seed(Seed);dist <- "Gaussian2";param <- 0.5
#' n <- 1000;L <- 5;num_it <- 4000;n_grid <- 50;p <- 2
#' theta<-c(0, 2);mix_prob<-c(0.2, 0.8);theta1<-rep(0,n);theta2<-rep(0,n)
#' Y <- matrix(0, n, p)
#' for(i in 1:n){
#'     mu = base::sample(theta, size=1, prob=mix_prob)
#'     sigma = 1*(mu==0)+0.1*(mu==2)
#'     Y[i,] = mu+sigma*rnorm(p)
#'     theta1[i] = mu;theta2[i] = sigma
#' }
#' net_npmle <- Net_NPMLE2(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid, p=p)
#' mu_support <- unique(net_npmle$support[,1])
#' sig_support <- unique(net_npmle$support[,2])
#' mu_prob <- rep(0, n_grid);sigma_prob <- rep(0, n_grid)
#' for(i in 1:n_grid){
#'     mu_prob[i] <- sum( net_npmle$prob[net_npmle$support[,1]==mu_support[i]] )
#'     sigma_prob[i] <- sum( net_npmle$prob[net_npmle$support[,2]==sig_support[i]] )
#'  }
#' plot(mu_support, mu_prob, col=rgb(1,0,0,0.8), type='l', xlim=c(-1,3),
#'      xlab='support', ylab='mass', lwd=3, ylim=c(0,1.2), cex.axis=1.85, cex.lab=1.85)
#' lines(sig_support, sigma_prob, col=rgb(0,0,1,0.8), type='l', lwd=3, lty=2)
#' legend(x=2.0, y=1.3, c(expression(mu),expression(sigma^2)),
#'        col=c(rgb(1,0,0),rgb(0,0,1)), lty=c(1,2), lwd=c(3,3),
#'        bty="n", x.intersp=0.5, y.intersp=0.8, seg.len=1.0, cex=1.85)
#' ### Gasussian smooth
#' Seed <- 128783;set.seed(Seed);dist <- "Gaussian2s";param <- 0.5
#' n <- 1000;L <- 5;num_it <- 4000;n_grid <- 50;p <- 2
#' mu <- 1;sigma <- 1; shape <- 2;scale <- 0.5
#' Y <- matrix(0, n, p)
#' theta1 <- MCMCpack::rinvgamma(n, shape, scale)
#' theta2 <- rnorm(n, mu, sd = sqrt(theta1 * sigma^2))
#' for(i in 1:n){Y[i,] <- theta2[i]+theta1[i]*rnorm(p)}
#' net_npmle <- Net_NPMLE2(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid, p=p)
#' mu_support <- unique(net_npmle$support[,1])
#' sig_support <- unique(net_npmle$support[,2])
#' mu_prob <- rep(0, n_grid);sigma_prob <- rep(0, n_grid)
#' for(i in 1:n_grid){
#'     mu_prob[i] <- sum( net_npmle$prob[net_npmle$support[,1]==mu_support[i]] )
#'     sigma_prob[i] <- sum( net_npmle$prob[net_npmle$support[,2]==sig_support[i]] )
#'  }
#' plot(mu_support, mu_prob, col=rgb(1,0,0,0.8), type='l', xlim=c(-1,3),
#'      xlab='support', ylab='mass', lwd=3, ylim=c(0,0.35), cex.axis=1.85, cex.lab=1.85)
#' lines(sig_support, sigma_prob, col=rgb(0,0,1,0.8), type='l', lwd=3, lty=2)
#' legend(x=2.0, y=0.385, c(expression(mu),expression(sigma^2)),
#'        col=c(rgb(1,0,0),rgb(0,0,1)), lty=c(1,2), lwd=c(3,3),
#'        bty="n", x.intersp=0.5, y.intersp=0.8, seg.len=1.0, cex=1.85)


Net_NPMLE2 <- function(X=NULL, Y, param=0, dist=NULL, hidden_size=500,
                       num_it=NULL, p=2, gpu_ind=0, L=5, n_grid=100,
                       lr=0.0001, n0=n, n=NULL, lrdecay=0, lr_power=0.2,
                       verb=1){
    require(reticulate)

    if(p <= 1){stop("Dimension of y should be greater than 1")}
    if(is.null(n) == TRUE){stop("Sample size n is missing!")}
    if(is.null(X) == TRUE){X = matrix(1, ncol=p, nrow=n)}
    if(is.vector(Y) == TRUE){Y = matrix(Y, ncol=1, nrow=n)}
    if(is.vector(X) == TRUE){X = matrix(X, ncol=p, nrow=n)}

    if(is.null(dist) == TRUE){stop("Distribution of Y is missing!")}
    if(is.null(num_it) == TRUE){
        num_it = 4000
        print("Warning: Number of iterations is not specified! (Set to be 4,000)")}

    Have_torch = reticulate::py_module_available("torch")
    Have_numpy = reticulate::py_module_available("numpy")
    if (!Have_torch) stop("Pytorch is not installed!")
    if (!Have_numpy) stop("Numpy is not installed!")

    Mu <- apply(Y, 1, mean)
    Sigma <- (Mu-mean(Y))^2 + apply((Y-Mu)^2 ,1, mean)
    mu1 <- min(Mu);mu2 <- max(Mu)
    sigma1 <- sqrt(min(Sigma));sigma2 <- sqrt(max(Sigma))

    y1 <- r_to_py(Y, convert=FALSE)
    X1 <- r_to_py(X, convert=FALSE)
    hidden_size1 <- r_to_py(hidden_size, convert=FALSE)
    gpu_ind1 <- r_to_py(gpu_ind, convert=FALSE)
    L1 <- r_to_py(L, convert=FALSE)
    n1 <- r_to_py(n, convert=FALSE)
    p1 <- r_to_py(p, convert=FALSE)
    n01 <- r_to_py(n0, convert=FALSE)
    num_it1 <- r_to_py(num_it, convert=FALSE)
    lr1 <- r_to_py(lr, convert=FALSE)
    lrdecay1 <- r_to_py(lrdecay, convert=FALSE)
    lr_power1 <- r_to_py(lr_power, convert=FALSE)
    verb1 <- r_to_py(verb, convert=FALSE)
    n_grid1 <- r_to_py(n_grid, convert=FALSE)
    dist1 <- r_to_py(dist, convert=FALSE)
    param1 <- r_to_py(param, convert=FALSE)
    mu1 <- r_to_py(mu1, convert=FALSE)
    mu2 <- r_to_py(mu2, convert=FALSE)
    sigma1 <- r_to_py(sigma1, convert=FALSE)
    sigma2 <- r_to_py(sigma2, convert=FALSE)

    Code_Netnpmle <- paste(system.file(package="NetNPMLE"), "G2_NPMLE.py", sep="/")
    reticulate::source_python(Code_Netnpmle)

    fit <- G2_NPMLE(y1, X1, hidden_size1, gpu_ind1, L1, n1, p1, n01,num_it1, lr1,
                    lrdecay1, lr_power1, verb1, n_grid1, dist1, param1,
                    mu1, mu2, sigma1, sigma2)

    out_fit <- list("support" = as.matrix(fit[[1]]),
                    "prob" = as.matrix(fit[[2]]))

    return(out_fit)
}
