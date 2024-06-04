#' @title
#' neural-g
#'
#' @description
#' Train neural-g (neural net-g modelling).
#' python(>=3.7) and pytorch
#' are needed to be installed in advance. R pakcage 'reticulate' is also required.
#'
#' @param X predictor Variable.
#' @param Y response Variable.
#' @param dist probabilistic distribution of Y, including:
#' "Gaussian", "Poisson", "LogGaussian", "Gumbel", "Gaussian_scale".
#' @param n_grid number of support grids.
#' @param num_it number of iterations for training.
#' @param param nuisance parameter value. param = 0 indicates there is no nuisance parameter.
#' @param hidden_size number of hidden neurons at each layer.
#' @param L number of hidden layers.
#' @param n number of sample size.
#' @param p the dimension of predictor Variable;Default is 1.
#' @param c the weight parameter in weighted average gradients;Default is 0.5.
#' @param n0 number of mini-batch size;Default is equal to n.
#' @param lr0 learning rate, default is 0.0001.
#' @param lrDecay lrDecay = 1: using decaying learning rate.
#' @param lrpower decay rate of learning rate, default is 0.2.
#' @param gpu_ind gpu index.
#' @param verb print information while training generator.
#' @usage
#' neural_g(X=NULL, Y, param=1, dist=NULL, hidden_size=500,
#' num_it=NULL, p=1, gpu_ind=0, L=5, n_grid=100,
#' lr=0.0001, n0=n, n=NULL, lrDecay=0, lrpower=0.2,
#' verb=1)
#'
#' @return
#' neural-g function returns a list with support grids (''support'') and estimated probability mass (''prob'')
#'
#' @author
#' Shijie Wang, Chakraborty Saptarshi, Qin Qian, Ray Bai
#' @export
#' @examples
#' ### log-Gaussian
#' Seed <- 128783;set.seed(Seed);dist <- "LogGaussian";param <- 0.2
#' n <- 4000;L <- 5;num_it <- 4000;n_grid <- 100
#' theta <- rbeta(n, 3, 2);Y = rlnorm(n, theta, param)
#' net_g <- neural_g(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid)
#' plot(net_g$support, net_g$prob, col=rgb(1,0,0,0.8), type='l',
#'      xlab='support', ylab='mass', lwd=3, ylim=c(0, 0.5), cex.axis=1.85, cex.lab=1.85)
#' ### Gaussian-uniform
#' Seed <- 128783;set.seed(Seed);dist <- "Gaussian";param <- 1.0
#' n <- 4000;L <- 5;num_it <- 4000;n_grid <- 100
#' theta <- runif(n, -2, 2);Y = theta + rnorm(n, 0, param)
#' net_g <- neural_g(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid)
#' plot(net_g$support, net_g$prob, col=rgb(1,0,0,0.8), type='l',
#'      xlab='support', ylab='mass', lwd=3, ylim=c(0,0.2), cex.axis=1.85, cex.lab=1.85)
#' efnpmle = deconvolveR::deconv(tau=net_g$support, X=Y, deltaAt=0, family="Normal", pDegree=5, c0=1.0)
#' efnpmle20 = deconvolveR::deconv(tau=net_g$support, X=Y, deltaAt=0, family="Normal", pDegree=20, c0=0.5)
#' lines(net_g$support, efnpmle$stats[,"g"], type="l", xlab="", ylab="", col=rgb(0,0.5,0.5), lty=1, lwd=2)
#' lines(net_g$support, efnpmle20$stats[,"g"], type="l", xlab="", ylab="", col=rgb(0,0,1), lty=1, lwd=2)
#' legend("topright", c("neural_g","Efron(5)","Efron(20)"),
#'         col=c(rgb(1,0,0),rgb(0,0.5,0.5),rgb(0,0,1)), lty=c(1,1,1),
#'         lwd=c(3,2,2,2,2), bty="n", cex=1.0, x.intersp=0.8, y.intersp=0.8)

neural_g <- function(X=NULL, Y, param=0, dist=NULL, hidden_size=500,
                      num_it=NULL, p=1, c=0.5, gpu_ind=0, L=5, n_grid=100,
                      lr=0.0001, n0=n, n=NULL, lrdecay=0, lr_power=0.2,
                      verb=1){
    require(reticulate)

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

    y1 <- r_to_py(Y, convert=FALSE)
    X1 <- r_to_py(X, convert=FALSE)
    hidden_size1 <- r_to_py(hidden_size, convert=FALSE)
    gpu_ind1 <- r_to_py(gpu_ind, convert=FALSE)
    L1 <- r_to_py(L, convert=FALSE)
    n1 <- r_to_py(n, convert=FALSE)
    p1 <- r_to_py(p, convert=FALSE)
    c1 <- r_to_py(c, convert=FALSE)
    n01 <- r_to_py(n0, convert=FALSE)
    num_it1 <- r_to_py(num_it, convert=FALSE)
    lr1 <- r_to_py(lr, convert=FALSE)
    lrdecay1 <- r_to_py(lrdecay, convert=FALSE)
    lr_power1 <- r_to_py(lr_power, convert=FALSE)
    verb1 <- r_to_py(verb, convert=FALSE)
    n_grid1 <- r_to_py(n_grid, convert=FALSE)
    dist1 <- r_to_py(dist, convert=FALSE)
    param1 <- r_to_py(param, convert=FALSE)

    Code<- paste(system.file(package="neuralG"), "neural_g.py", sep="/")
    reticulate::source_python(Code)

    fit <- neural_g(y1, X1, hidden_size1, gpu_ind1, L1, n1, p1, n01,num_it1, lr1,
                   lrdecay1, lr_power1, verb1, n_grid1, dist1, param1, c1)

    out_fit <- list("support" = as.numeric(fit[[1]]),
                    "prob" = as.numeric(fit[[2]]))

    return(out_fit)
}
