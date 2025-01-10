import torch
import torch.fft as fft

import math


#Gaussian random fields with Matern-type covariance: C = sigma^2 (-Lap + tau^2 I)^-alpha 

class GaussianRF1d(object):

    def __init__(self, s, L=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        self.s = s

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 1.0))
        else:
            self.sigma = sigma

        const = (4*(math.pi**2))/(L**2)

        k = torch.arange(start=0, end=s//2 + 1, step=1).type(dtype).to(device)

        self.sqrt_eig = s*self.sigma*((const*(k**2) + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = fft.irfft(torch.view_as_complex(xi), n=self.s)

        if self.mean is not None:
            u += self.mean
        
        return u

class GaussianRF2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, alpha=2.0, tau=3.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        # self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

    def sample(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        # print('xi',xi.shape)
        # print("xi_cplx",torch.view_as_complex(xi).shape)
        u = fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))
        # print('u', u.shape)
        if self.mean is not None:
            u += self.mean
        
        return u

