from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torch.nn import functional as F
import torch.distributions as dists
from functools import partial
import time

GS = 1000
ts = time.time()
LOG_SAFETY_CONST = 1e-10


def synth_orth_Ey_func(t, z, alpha, theta, quadt=False):
    assert z.shape[1] == 1
    return alpha*z.sum(dim=1)/np.sqrt(z.shape[1]) + torch.mv(t, theta.view(-1)).view(-1)

normal_sigma_1 = dists.Normal(0, 1)

def generate_beta_theta(numt, v):
    print('DOES NOT WORK YET')
    # initial sample
    beta = normal_sigma_1.sample((numt,v))
    theta = normal_sigma_1.sample((numt,))
    
    beta = beta/torch.norm(beta, p=2, dim=0).view(1,-1)
    for i in range(v):
        beta_i_normed = beta[:,i].view(-1)
        assert np.abs(torch.norm(beta_i_normed, p=2)  - 1) < 1e-5, (torch.norm(beta_i_normed, p=2))
        
        assert beta_i_normed.shape == theta.shape, (theta_proj_on_beta.shape, theta.shape)

        theta_proj_on_beta_i = theta.dot(beta_i_normed)
            
        theta = theta - theta_proj_on_beta_i*beta_i_normed
            
    theta = theta/torch.norm(theta, p=2)

    return beta, theta


def generate_synth_orth_data(m, numt=2, v=1, 
                             beta=None,
                             theta_orth=None,
                             print_details=False,
                             alpha=1, gamma=0.5,
                             mu=None,
                             corrupt=0.0):
    assert numt > 1, "Please use more than 1 treatment"

    t = mu*normal_sigma_1.sample((m, numt))

    if beta is None:
        beta=torch.ones((numt,1))/np.sqrt(numt)
        
    assert (torch.norm(beta, p=2) - 1.0) < 1e-4, (torch.norm(beta))
        
    if theta_orth is None:
        theta_orth = torch.tensor([(-1)**i for i in range(numt)]).view(1,-1)/np.sqrt(numt)
        
    assert (torch.sum(theta_orth) - 0.0) < 1e-4, (torch.sum(theta_orth))
    
    assert (torch.norm(beta, p=2) - 1.0) < 1e-4, (torch.norm(beta))
        
    theta_orth = theta_orth + corrupt

    def h(t, beta):
        assert t.shape[1] == numt, (t.shape, numt)
        return gamma*torch.mm(t, beta.view(numt,-1),).view(t.shape[0], beta.shape[1])
    
    ey_func = partial(synth_orth_Ey_func, alpha=alpha, theta=theta_orth, quadt=True)
            
    z = h(t, beta)
    
    assert z.shape == (m, v)
    assert t.shape == (m, numt), t.shape

    y = 0.32 * normal_sigma_1.sample((m,)) + ey_func(t, z)
    assert y.shape == (m,), y.shape
        
    return h, ey_func, beta, z, t, y

def nonlin_Ey_func(t,z,alpha, parity):
    return (t*t*parity).sum(dim=1) + alpha*z

def generate_nonlin_sqrd_data(m, numt=2,
                             beta=None,
                             theta_orth=None,
                             print_details=False,
                             alpha=1, gamma=0.5,
                             mu=1,
                             corrupt=0.0):
    assert numt > 1, "Please use more than 1 treatment"
    assert numt%2 ==0, "GENERATION PROCESS WITH EVEN TREATMENTS ONLY WORK FOR THIS GENERATION; causal redundancy fails otherwise"
 
    t = mu*normal_sigma_1.sample((m, numt)) #+ mu*b.reshape(-1,1)

    def h(t, beta=None):
        assert t.shape[1] == numt, (t.shape, numt)
        ht = 0
        for i in range(0, numt,2):
            ht = ht + t[:, i]*t[:, i+1]
        return gamma*ht
    
    ey_func = partial(nonlin_Ey_func, alpha=alpha,
                      parity=torch.tensor([(-1)**i for i in range(numt)]).view(1,-1)/np.sqrt(numt))
    
            
    z = h(t).view(-1)
    
    assert z.shape == (m,)
    assert t.shape == (m, numt), t.shape

    y = 0.32 * normal_sigma_1.sample((m,)) + ey_func(t, z)
    assert y.shape == (m,), y.shape

    return h, ey_func, None, z, t, y