import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import time
from random import sample
import os
import sys
from collections import defaultdict

def G_NPMLE(y, X, hidden_size1, gpu_ind1, L1, n1, p1, n01, 
    num_it1, lr1, lrdecay1, lr_power1, verb1, n_grid1, dist1, param1):
    
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
    
    #Neural Network Structure
    class GNet(nn.Module):
      def __init__(self, hidden_size, L): 
        super(GNet, self).__init__()
        
        self.relu = nn.Tanh()#nn.Sigmoid()#nn.ReLU()
        #self.relu = nn.ReLU()
        #self.fc0 = nn.Linear(n_grid, hidden_size)
        #self.fc_out = nn.Linear(hidden_size, n_grid)
        
        self.fc0 = nn.Linear(1, hidden_size)
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
        T = T.view(-1)
        if Dist == "Gaussian": distribution = torch.distributions.normal.Normal(T, Param)
        if Dist == "Gaussian-framing": distribution = torch.distributions.normal.Normal(T, Param)
        if Dist == "Possion": distribution = torch.distributions.poisson.Poisson(T)
        if Dist == "Binomial": distribution = torch.distributions.binomial.Binomial(total_count=Param, probs=T)
        if Dist == "LogGaussian": distribution = torch.distributions.log_normal.LogNormal(T, Param)
        if Dist == "Cauchy": distribution = torch.distributions.cauchy.Cauchy(T, Param)
        if Dist == "Gumbel": distribution = torch.distributions.gumbel.Gumbel(T, Param)
        if Dist == "Gaussian_scale": distribution = torch.distributions.normal.Normal(Param, T)
        #if Dist == "Binomial": distribution = torch.distributions.binomial.Binomial(total_count=T, probs=Param)
        if Dist == "Binomial-n":
            density_y = torch.zeros(n, n_grid).to(device)
            for j in range(n_grid):
                #print()
                distribution = torch.distributions.binomial.Binomial(total_count=X.reshape(-1), probs=T[j])
                density_y[:,j] = torch.exp(distribution.log_prob(Y.reshape(-1))).to(device)#(n_y,n_grid)
        else:
            density_y = torch.exp(distribution.log_prob(Y)).to(device)#(n_y,n_grid)
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
    y = y.view(n,1).to(device)
    y_min = torch.min(y)
    y_max = torch.max(y)
    c = 0.5
    
    #y_min = -50
    #y_max = 50
    
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
        #X0 = X[ind,:].to(device, dtype=torch.float)
        #y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
        
        #split data
        index = np.arange(n)
        np.random.shuffle(index)
        ind_split = np.split(index, K0)
        #print(torch.from_numpy(ind_split[0]).to(device, dtype=torch.float))
        
        #support = torch.linspace(lam_min, lam_max, n_grid)
        
        #mini-batch k
        for h in range(K0):
            
            #ind = sample(range(n), n0)
            if n != n0 : ind = ind_split[h]
            
            X0 = X[ind,:].reshape(n0,p).to(device, dtype=torch.float)
            y0 = y[ind,:].reshape(n0,1).to(device, dtype=torch.float)
            
            #eps = (-0.001) * torch.rand(n_grid) + 0.001/2
            #support = torch.linspace(y_min, y_max, n_grid) + 1.0 * eps#torch.zeros(1)
            support = torch.linspace(y_min, y_max, n_grid)
            #if dist == "Possion": support = torch.linspace(0.0, y_max, n_grid)
            if dist == "Binomial": support = torch.linspace(0.0, 1.0, n_grid)
            if dist == "Binomial-n": support = torch.linspace(0.0, 1.0, n_grid)
            if dist == "Possion": support = torch.linspace(0.0, y_max, n_grid)
            if dist == "LogGaussian": support = torch.linspace(-y_max, y_max, n_grid)
            if dist == "Gaussian_scale": support = torch.linspace(0.05, y_max, n_grid)
            if dist == "Gaussian-framing": support = torch.linspace(50, 263, n_grid)
            #support = support.to(device)
            support = support.to(device).view(n_grid,1)
            
            D.zero_grad()
            #optimD.zero_grad()
            
            
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
            #nn.utils.clip_grad_norm_(D.parameters(), max_norm=35, norm_type=2)
            #grads = torch.autograd.grad(loss, D.parameters())
            #print(grads.shape)
            #print(D.parameters.shape)
            for i, para in enumerate(D.parameters()):
                if i in grad_dict:
                    cur_grad = para.grad.clone()
                    #para.grad = 0.5*para.grad + 0.5*(torch.randn_like(para.grad)*0.1+grad_dict[i])
                    para.grad = (1-c)*para.grad + c*grad_dict[i]
                    #para.grad = 0.5*(para.grad + grad_dict[i])#/(grad_dict[i])
                    #grad_dict[i] += (0.5**(it)) * cur_grad
                    grad_dict[i] += cur_grad
                else:
                    grad_dict[i] = torch.zeros_like(para.grad)
            #    #print(param.grad.shape)
            #    para.grad += (torch.randn_like(para.grad) * 1 + 0)

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
    support = torch.linspace(y_min, y_max, n_grid)#.to(device).view(n_grid,1)
    #if dist == "Possion": support = torch.linspace(0.0, 1.0, n_grid)
    if dist == "Binomial": support = torch.linspace(0.0, 1.0, n_grid)
    if dist == "Binomial-n": support = torch.linspace(0.0, 1.0, n_grid)
    if dist == "Gaussian_scale": support = torch.linspace(0.05, y_max, n_grid)
    if dist == "Possion": support = torch.linspace(0.0, y_max, n_grid)
    if dist == "LogGaussian": support = torch.linspace(-y_max, y_max, n_grid)
    if dist == "Gaussian-framing": support = torch.linspace(50, 263, n_grid)
    support = support.to(device).view(n_grid,1)
    
    npmle_density = D(support)
    npmle_density = npmle_density/torch.sum(npmle_density)
    
    #npmle_density = npmle_density/torch.sum(support*npmle_density)
    npmle_density= npmle_density.cpu().detach().numpy() 
    support = support.cpu().detach().numpy() 
    
    return support, npmle_density
        
    
