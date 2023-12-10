# NetNPMLE
An R package for implementing Neural-Network based NPMLE, _Net-NPMLE_ in short.

## Abstract 
Net-NPMLE is a flexible nonparametric maxium likelihood estimator (NPMLE) based on deep neural network for latent mixture model. Net-NPMLE is free of tuning hyperparameters and is capable of estimating a smooth or shapely discrete shape of latent distribution.

## Installation
To run the Net-NPMLE smoothly, there are several pre-requisites needed to be installed before the R package. The main of Net-NPMLE is worte in `Python`, especially __Pytorch__ library and we strongly recommend using `CUDA` (GPU-Based tool) to train Net-NPMLE which can be accelerated a lot than using `CPU`.
- __Python__ 3.7 or above
- __[Pytroch](https://pytorch.org/)__ 1.11.0 or above
- __[NAVID CUDA](https://developer.nvidia.com/cuda-toolkit)__ 10.2 or above

In R, we also need `reticulate` package to run `Python` in R and `devtools` to install R package from github.
```
install.package("reticulate")
install.package("devtools")
```

Now, use the following code to install `GMS` package.
```
library(devtools)
install_github(repo = "shijiew97/NetNPMLE")
library(NetNPMLE)
```
## Main function
There are two main functions in the `Net-NPMLE` package, which is detailed specified below.
- `Net_NPMLE` aims to give out Net-NPMLE estimator in uni-variate mixture model. Currently Net-NPMLE supports mixutre models such as Gaussian-location, Poisson-mixture, Lognormal-location, Gumbel-location, Gaussian-scale, Binomial-prob.
- `Net_NPMLE2` provides bi-variate Net-NPMLE estimator in bi-variate mixture model. Currently bi-variate Net-NPMLE supports Gaussian location-scale mixture model.

#### Example: Lognormal-location (Beta) mixture.
As a simple example of `NetNPMLE` pacakge, we consider a Lognorm-location mixture: $\mathbf{Y} \mid \theta \sim \text{Log-normal}(\theta, 1/5) \text{ and }  \theta \sim \text{Beta}(3,2)$ where latent distribution follows $\text{Beta}(3,2)$ have a support of $[0,1]$.

![Alt text](Image/)

























