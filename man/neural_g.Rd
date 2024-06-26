% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/neural_g.R
\name{neural_g}
\alias{neural_g}
\title{neural-g}
\usage{
neural_g(X=NULL, Y, param=1, dist=NULL, hidden_size=500,
num_it=NULL, p=1, gpu_ind=0, L=5, n_grid=100,
lr=0.0001, n0=n, n=NULL, lrDecay=0, lrpower=0.2,
verb=1)
}
\arguments{
\item{X}{predictor Variable.}

\item{Y}{response Variable.}

\item{param}{nuisance parameter value. param = 0 indicates there is no nuisance parameter.}

\item{dist}{probabilistic distribution of Y, including:
"Gaussian", "Poisson", "LogGaussian", "Gumbel", "Gaussian_scale".}

\item{hidden_size}{number of hidden neurons at each layer.}

\item{num_it}{number of iterations for training.}

\item{p}{the dimension of predictor Variable;Default is 1.}

\item{c}{the weight parameter in weighted average gradients;Default is 0.5.}

\item{gpu_ind}{gpu index.}

\item{L}{number of hidden layers.}

\item{n_grid}{number of support grids.}

\item{n0}{number of mini-batch size;Default is equal to n.}

\item{n}{number of sample size.}

\item{verb}{print information while training generator.}

\item{lr0}{learning rate, default is 0.0001.}

\item{lrDecay}{lrDecay = 1: using decaying learning rate.}

\item{lrpower}{decay rate of learning rate, default is 0.2.}
}
\value{
neural-g function returns a list with support grids (''support'') and estimated probability mass (''prob'')
}
\description{
Train neural-g (neural net-g modelling).
python(>=3.7) and pytorch
are needed to be installed in advance. R pakcage 'reticulate' is also required.
}
\examples{
### log-Gaussian
Seed <- 128783;set.seed(Seed);dist <- "LogGaussian";param <- 0.2
n <- 4000;L <- 5;num_it <- 4000;n_grid <- 100
theta <- rbeta(n, 3, 2);Y = rlnorm(n, theta, param)
net_g <- neural_g(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid)
plot(net_g$support, net_g$prob, col=rgb(1,0,0,0.8), type='l',
     xlab='support', ylab='mass', lwd=3, ylim=c(0, 0.5), cex.axis=1.85, cex.lab=1.85)
### Gaussian-uniform
Seed <- 128783;set.seed(Seed);dist <- "Gaussian";param <- 1.0
n <- 4000;L <- 5;num_it <- 4000;n_grid <- 100
theta <- runif(n, -2, 2);Y = theta + rnorm(n, 0, param)
net_g <- neural_g(Y=Y, param=param, dist=dist, n=n, num_it=num_it, n_grid=n_grid)
plot(net_g$support, net_g$prob, col=rgb(1,0,0,0.8), type='l',
     xlab='support', ylab='mass', lwd=3, ylim=c(0,0.2), cex.axis=1.85, cex.lab=1.85)
efnpmle = deconvolveR::deconv(tau=net_g$support, X=Y, deltaAt=0, family="Normal", pDegree=5, c0=1.0)
efnpmle20 = deconvolveR::deconv(tau=net_g$support, X=Y, deltaAt=0, family="Normal", pDegree=20, c0=0.5)
lines(net_g$support, efnpmle$stats[,"g"], type="l", xlab="", ylab="", col=rgb(0,0.5,0.5), lty=1, lwd=2)
lines(net_g$support, efnpmle20$stats[,"g"], type="l", xlab="", ylab="", col=rgb(0,0,1), lty=1, lwd=2)
legend("topright", c("neural_g","Efron(5)","Efron(20)"),
        col=c(rgb(1,0,0),rgb(0,0.5,0.5),rgb(0,0,1)), lty=c(1,1,1),
        lwd=c(3,2,2,2,2), bty="n", cex=1.0, x.intersp=0.8, y.intersp=0.8)
}
\author{
Shijie Wang, Chakraborty Saptarshi, Qin Qian, Ray Bai
}
