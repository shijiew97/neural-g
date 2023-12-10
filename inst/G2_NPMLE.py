import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys
from collections import defaultdict
from itertools import product

def G2_NPMLE(y, X, hidden_size1, gpu_ind1, L1, n1, p1, n01, 
    num_it1, lr1, lrdecay1, lr_power1, verb1, n_grid1, dist1, param1,
    mu1, mu2, sigma1, sigma2):
    
    #Gpu index    
    gpu_ind = int(gpu_ind1)
    if gpu_ind == -1:
        device = 'cpu'
        print("Training G via CPU computing starts.")
        print("WARNING: CPU computing would be very slow!")
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda', gpu_ind)
            print("Training G via GPU computing starts!")
            print(device)
        else:
            try:
              if torch.backends.mps.is_available() == True:
                device = torch.device('mps')
                print("Training G via Apple M1 Metal computing starts!")
              else:
                device = torch.device('cpu')
                print("Training G via CPU computing starts!")
                print("WARNING: CPU computing would be very slow!")
            except:
              device = torch.device('cpu')
              print("Training G via CPU computing starts!")
              print("WARNING: CPU computing would be very slow!")
    
    #Data Initialization
    sys.stdout.flush()
    if torch.is_tensor(X) == False: X = torch.from_numpy(X)
    if torch.is_tensor(y) == False: y = torch.from_numpy(y)
    
    #Parameter Setup
    L = int(L1)
    n = int(n1)
    hidden_size = int(hidden_size1)
    iteration = int(num_it1)
    lr = float(lr1)
    verb = int(verb1)
    lrDecay = int(lrdecay1)
    lrPower = float(lr_power1)
    n_grid = int(n_grid1)
    dist = str(dist1)
    param = float(param1)
    p = int(p1)
    n0 = int(n01)
    J = y.shape[1]
    covar = torch.eye(3)*0.2#.to(device)
    covar = covar.to(device)
    mu1 = float(mu1)
    mu2 = float(mu2)
    sigma1 = float(sigma1)
    sigma2 = float(sigma2)
    
    #Neural Network Structure
    class GNet(nn.Module):
      def __init__(self, hidden_size, L): 
        super(GNet, self).__init__()
        
        self.relu = nn.Tanh()#nn.Sigmoid()#nn.ReLU()
        #self.relu = nn.ReLU()
        #self.fc0 = nn.Linear(n_grid, hidden_size)
        #self.fc_out = nn.Linear(hidden_size, n_grid)
        
        #self.fc0 = nn.Linear(1, hidden_size)
        self.fc0 = nn.Linear(2, hidden_size)
        if dist == "student3": self.fc0 = nn.Linear(3, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 1)
        
        self.layers = nn.ModuleList()
        for j in range(L - 1):
          self.layers.append( nn.Linear(hidden_size, hidden_size) )
          self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.ReLU() )
          self.layers.append( nn.ReLU() ) 
          #self.layers.append( nn.Linear(n_grid, hidden_size) )
          #self.layers.append( nn.LayerNorm(hidden_size) )
          #self.layers.append( nn.Linear(1, hidden_size, bias=False) )

      def forward(self, Support):
          
        out = self.relu(self.fc0(Support))
        
        u = 3
        for j in range(L-1):
          out = self.layers[u*j](out)
          out = self.layers[u*j+1](out) #+ 0.02*self.layers[u*j+3](Support.view(-1))
          out = self.layers[u*j+2](out) #+ a#self.layers[u*j+4]( self.layers[u*j+3](a) )
        
        out = self.fc_out(out)
        out = torch.exp(out)#*0.1
        #out = torch.exp(out)
        #out = torch.square(out)
        #out = torch.relu(out)
        #out = torch.square(out)
        #out = out/torch.sum(out)
        #out = torch.sort(out)[0]
        
        return out
    
    #NPMLE loss
    def Loss_npmle(Density_D, Dist, Param, T, Y, device):
        #define the NPMLE loss
        #Density: Output of D
        #Dist: y's distribution
        #Param: auxillary variable
        #T: finite support
        #T = T.view(-1)
        if Dist == "Gaussian-fram": 
          distribution = torch.distributions.normal.Normal(T[:,0], T[:,1])
          temp = torch.zeros((n, n_grid**2, J))
        if Dist == "Gaussian2": 
          distribution = torch.distributions.normal.Normal(T[:,0], T[:,1])
          temp = torch.zeros((n, n_grid**2, J))
        if Dist == "Gaussian2s": 
          distribution = torch.distributions.normal.Normal(T[:,0], T[:,1])
          temp = torch.zeros((n, n_grid**2, J))
        if Dist == "student3": 
          #distribution = torch.distributions.studentT.StudentT(T[:,0], T[:,1], T[:,2])
          distribution = torch.distributions.multivariate_normal.MultivariateNormal(loc=T, covariance_matrix=covar)
          temp = torch.zeros((n, n_grid**3))
          
        #if Dist == "Possion": distribution = torch.distributions.poisson.Poisson(T)
        #if Dist == "Binomial": distribution = torch.distributions.binomial.Binomial(total_count=Param, probs=T)
        #if Dist == "LogGaussian": distribution = torch.distributions.log_normal.LogNormal(T, Param)
        #if Dist == "Cauchy": distribution = torch.distributions.cauchy.Cauchy(T, Param)
        #if Dist == "Gumbel": distribution = torch.distributions.gumbel.Gumbel(T, Param)
        #if Dist == "Gaussian_scale": distribution = torch.distributions.normal.Normal(Param, T)
        #if Dist == "Binomial": distribution = torch.distributions.binomial.Binomial(total_count=T, probs=Param)
        if Dist == 'student3':
            for i in range(n):
                temp[i,:] = torch.exp(distribution.log_prob(Y[i,:].view(3)))
            density_y = temp.to(device)
        else:
            for i in range(J): 
                temp[:,:,i] = torch.exp(distribution.log_prob(Y[:,i].view(n,1)))
            density_y = torch.prod(temp, dim=2).to(device)#(n_y,n_grid)
        #print(density_y.shape)
        density = torch.sum(density_y*Density_D, dim=1)

        loss_npmle = -torch.mean(torch.log(density))
        #loss_npmle = torch.prod((density))
        
        return loss_npmle
        
    #Generator initilization
    #D = Net(hidden_size, L).to(device)
    D = GNet(hidden_size, L).to(device)
    #optimD = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    optimD = torch.optim.SGD(D.parameters(), lr=lr)
    #optimD = torch.optim.SGD(D.parameters(), lr=lr, momentum=0.9)
    
    #Print Info
    LOSS = torch.zeros(iteration)
    X = X.view(n,p).to(device)
    y = y.view(n,J).to(device)
    y_min = torch.min(y)
    y_max = torch.max(y)

    #y_min = -50
    #y_max = 50
    
    if dist == "Gaussian2": 
        #support1 = torch.linspace(-2, 3, n_grid)
        #support2 = torch.linspace(0.000000000001, 1.5, n_grid)
        #support = support.to(device)
        support1 = torch.linspace(mu1, mu2, n_grid)
        support2 = torch.linspace(0.000000000001, sigma2, n_grid)
        support = torch.tensor(list(product(support1, support2)))
        support = support.to(device).view(n_grid**2, 2)
    if dist == "Gaussian2s": 
        #support1 = torch.linspace(-4, 4, n_grid)
        #support2 = torch.linspace(0.000000000001, 3, n_grid)
        #support = support.to(device)
        support1 = torch.linspace(mu1, -mu1, n_grid)
        support2 = torch.linspace(0.000000000001, sigma2, n_grid)
        support = torch.tensor(list(product(support1, support2)))
        support = support.to(device).view(n_grid**2, 2)
    if dist == "Gaussian-fram":
        support1 = torch.linspace(80, 250, n_grid)
        support2 = torch.linspace(1, 20, n_grid)
        support = torch.tensor(list(product(support1, support2)))
        support = support.to(device).view(n_grid**2, 2)
    if dist == "student3":
        support1 = torch.linspace(-3, 3, n_grid)
        support2 = torch.linspace(-3, 3, n_grid)
        support3 = torch.linspace(-3, 3, n_grid)
        support = torch.tensor(list(product(support1, support2, support3)))
        support = support.to(device).view(n_grid**3, 3)
    
    support0 = torch.sum(support, 1).view(n_grid**2, 1)
    #Learning rate and mini-batch optim
    lr1 = lr
    K0 = int(n/n0)    
    grad_dict = defaultdict(int)
    
    #Training Epoches
    if dist == "Gaussian": print("--Gaussian model is evluating--")
    if dist == "Possion": print("--Possion model is evluating--")
    Generator_start = time.perf_counter()
    for it in range(iteration):
        
        #Decayiing LR
        lr1 = lr/(float(it+1.0)**lrPower) 
        for param_group in optimD.param_groups:
           param_group["lr"] = lr1
        
        #mini-batch optim
        ind = sample(range(n), n0)
        X0 = X[ind,:].to(device, dtype=torch.float)
        y0 = y[ind,:].reshape(n0,J).to(device, dtype=torch.float)
        
        #split data
        index = np.arange(n)
        np.random.shuffle(index)
        ind_split = np.split(index, K0)
        
        #support = torch.linspace(lam_min, lam_max, n_grid)
        
        #mini-batch k
        for h in range(K0):
            
            ind = sample(range(n), n0)
            if n != n0 : ind = ind_split[h]
            
            X0 = X[ind,:].to(device, dtype=torch.float)
            y0 = y[ind,:].reshape(n0,J).to(device, dtype=torch.float)
            
            #eps = (-0.001) * torch.rand(n_grid) + 0.001/2
            #support = torch.linspace(y_min, y_max, n_grid) + 1.0 * eps#torch.zeros(1)
            
            D.zero_grad()
            #optimD.zero_grad()
            
            #Out_D = D(support0).reshape(-1)
            Out_D = D(support).reshape(-1)
            Out_D = torch.log(Out_D)-torch.max(torch.log(Out_D))
            Out_D = torch.exp(Out_D)
            Out_D = Out_D/torch.sum(Out_D)
            #Out_D = Out_D/torch.sum(Out_D)
            
            #print(torch.sum(support*Out_D))
            loss = Loss_npmle(Out_D, dist, param, support, y, device)
            
            #if torch.isnan(loss): print("The loss is NaN"); break
            
            loss.backward()
            
            #nn.utils.clip_grad_norm_(D.parameters(), max_norm=35, norm_type=2)
            #grads = torch.autograd.grad(loss, D.parameters())
            #print(grads.shape)
            #print(D.parameters.shape)
            for i, para in enumerate(D.parameters()):
                if i in grad_dict:
                    cur_grad = para.grad.clone()
                    #para.grad = 0.5*para.grad + 0.5*(torch.randn_like(para.grad)*0.1+grad_dict[i])
                    para.grad = 0.5*(para.grad + grad_dict[i])
                    #para.grad = 0.5*(para.grad + grad_dict[i])#/(grad_dict[i])
                    #grad_dict[i] += (0.5**(it)) * cur_grad
                    grad_dict[i] += cur_grad
                else:
                    grad_dict[i] = torch.zeros_like(para.grad)
            #    #print(param.grad.shape)
            #    para.grad += (torch.randn_like(para.grad) * 1 + 0)
            if dist == "Gaussian-fram":
                nn.utils.clip_grad_norm_(D.parameters(), max_norm=80, norm_type=2)

            optimD.step()
        
        LOSS[it] = loss.item()
        
        if (it+1) % 100==0 and verb == 1:
            percent = float((it+1)*100) /iteration
            arrow   = '-' * int(percent/100 *20 -1) + '>'
            spaces  = ' ' * (20-len(arrow))
            train_time = time.perf_counter() - Generator_start
            print('\r[%s/%s]'% (it+1, iteration), 'Progress: [%s%s] %d %%' % (arrow, spaces, percent),
            " Current/Initial Loss: {:.4f}/{:.4f}, Learning rate: {:.5f},  Training time: {:.1f}, Condition: {:.1f}"
            .format(loss.item(), LOSS[0], lr1, train_time, torch.sum(torch.mean(torch.diff(support))*Out_D)), end='')
            sys.stdout.flush()
    
    #Generation step        
    #if dist == "Gaussian2": 
    #    support1 = torch.linspace(-2, 3, n_grid)
    #    support2 = torch.linspace(0.000000000001, 1.5, n_grid)
        #support = support.to(device)
    #    support = torch.tensor(list(product(support1, support2)))
    #    support = support.to(device).view(n_grid**2, 2)
    #if dist == "Gaussian2s": 
    #    support1 = torch.linspace(-4, 4, n_grid)
    #    support2 = torch.linspace(0.000000000001, 3, n_grid)
        #support = support.to(device)
    #    support = torch.tensor(list(product(support1, support2)))
    #    support = support.to(device).view(n_grid**2, 2)
    #if dist == "Gaussian-fram":
    #    support1 = torch.linspace(80, 250, n_grid)
        #support2 = torch.linspace(6, 14, n_grid)
    #    support2 = torch.linspace(1, 20, n_grid)
        #support = support.to(device)
    #    support = torch.tensor(list(product(support1, support2)))
    #    support = support.to(device).view(n_grid**2, 2)
    #if dist == "student3":
    #    support1 = torch.linspace(-3, 3, n_grid)
    #    support2 = torch.linspace(-3, 3, n_grid)
    #    support3 = torch.linspace(-3, 3, n_grid)
    #    support = torch.tensor(list(product(support1, support2, support3)))
    #    support = support.to(device).view(n_grid**3, 3)
     
    #npmle_density = D(support0)
    npmle_density = D(support)
    npmle_density = npmle_density/torch.sum(npmle_density)
    
    #npmle_density = npmle_density/torch.sum(support*npmle_density)
    npmle_density= npmle_density.cpu().detach().numpy() 
    support = support.cpu().detach().numpy() 
    
    return support, npmle_density
        
    
