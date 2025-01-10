import torch
import torch.fft as fft
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from ParaCtrl import *
import math as mt


"""
Solver (scheme:etdRK4) for 1d KS equation: u_t+uu_x+u_xx+(\ nu)u_xxxx=0, x\in [0,l]. l: 'domain size'. 
All coefficient and configurations are in ParaCtrl.py
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gridx=torch.arange(start=1,end=N+1,step=1,dtype=torch.float64).to(device)/N*space_scaling
torch.set_printoptions(precision=8)

k=torch.cat((torch.arange(start=0,end=N//2+1),torch.arange(-N//2+1,0))).to(device)/half_period   #** changed from original
k_proj=torch.cat((torch.ones(N_proj//2+1),torch.zeros(N-N_proj),torch.ones(N_proj//2-1))).to(device)

class initial_value_lib:
    '''
    Initial Conditions.
    For random initial value (samples from Gaussian Random Field), set choose=5

    GRF Initialization: $N(\mu,C),\ \mu=\cos x(1+\sin x), wn=(3,6);$ $C=\sigma(-\Delta+\tau^2)^{-\frac \alpha 2},$
    '''

    def __init__(self,choose=choose_ic,alpha=2.0, tau=3.0,sigma=None):
        self.null_mode=max(round(half_period/mt.sqrt(niu)),1)#L_k~0 (L is the diag linear operator operated after fft)
        self.worst_mode=max(round(half_period/mt.sqrt(2*niu)),1)#L_k is the largest
        self.choose=choose
        self.s1=N
        self.device=device

        if sigma==None:
            self.sigma = 0.01*4*tau ** (0.5 * (2 * alpha - 1.0))
        else:
            self.sigma=sigma

        const1 = 1/(half_period**2)


        self.sqrt_eig =self.sigma*((const1*k**2 + tau**2)**(-alpha/2.0))[:self.s1//2+1]
        self.sqrt_eig[0]=0
    def __call__(self, *args):
        return self.forward(*args)

    def update_par(self):
        self.null_mode=max(round(half_period/mt.sqrt(niu)),1)#L_k~0 (L is the diag linear operator operated after fft)
        self.worst_mode=max(round(half_period/mt.sqrt(2*niu)),1)#L_k is the largest
        self.choose=choose_ic

    def period_original(self,x):
        # print(f"forward,{x.type}")
        y = 0.1*torch.cos(x / half_period) * (1 + torch.sin(x / half_period))
        return y

    # @staticmethod
    def nonperiod_original(self,x):
        y = torch.cos(x / half_period/space_scaling) * (1 + torch.sin(x / half_period/space_scaling))
        return y

    # @staticmethod
    def single_steady(self,x):
        return torch.sin(self.null_mode*x/half_period)

    # @staticmethod
    def single_worst(self,x):
        return torch.sin(self.worst_mode * x / half_period)
    def nonL_period(self,x):
        y=self.period_original(x)
        return torch.tanh(y)

    def random_gen(self,x):
        y = torch.cos(x) * (1 + torch.sin(x))

        #random_gen
        xi = torch.randn(Nsum,self.s1 // 2 + 1, 2, device=self.device)
        xi[..., 0] = self.sqrt_eig * xi[..., 0]
        xi[..., 1] = self.sqrt_eig * xi[..., 1]
        # print('xi',xi.shape)
        # print("xi_cplx",torch.view_as_complex(xi).shape)
        y= y+fft.irfft(torch.view_as_complex(xi),norm=norm)
        return y
    functions = {
        0:period_original,1: nonperiod_original,
        2:single_steady,3:single_worst,
        4:nonL_period,5:random_gen
    }
    def forward(self, x):
        # print(f"forward,{x.type}")
        return self.functions[self.choose](self,x)



initial_value=initial_value_lib()

u0=initial_value(gridx)  #be updated in iterations

"u: physics functions; v: its Fourier transform"

ut=(u0.clone()).unsqueeze(-2) #track and save the (u_t) traj


v=torch.fft.fft(u0,norm=norm)
vt=(v.clone()).unsqueeze(-2)


L=k**2-niu*(k**4)
E_2=torch.exp(timegrid*L/2)
E=E_2**2

class coeff_scheme():
    ''' coefficient for num-scheme (etdrk4)'''
    def __init__(self,choose=choose_coef,eta=eta_stb_change,MM=M):
        self.choose=choose
        self.eta=eta
        self.MM=M
    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def ff(self,x):

        r=torch.exp(1j*torch.pi*(torch.arange(1,M+1,dtype=torch.float64).to(device)-0.5)/M)

        lr=x.unsqueeze(1)+r
        e_lr=torch.exp(lr/2)
        e_lr2=e_lr**2
        q = torch.real(torch.mean(dim=1, input=(e_lr - 1) / lr))
        f1 =torch.real(torch.mean(dim=1, input=((-4 - lr + e_lr2 * (4 - 3 * lr + lr ** 2))) / (lr ** 3)))
        f2 =torch.real(torch.mean(dim=1, input=((2 + lr + e_lr2 * (lr - 2))) / (lr ** 3)))
        f3 =torch.real(torch.mean(dim=1, input=((-4 - 3 * lr - lr ** 2 + e_lr2 * (4 - lr))) / (lr ** 3)))

        return q,f1,f2,f3


    functions={
        0:ff
    }
    def forward(self, x):
        # print(f"forward,{x.type}")
        return self.functions[self.choose](self,x)
    def forward_scheme(self,x):
        y=self.forward(timegrid*x)
        return tuple(timegrid*i for i in y)


f=coeff_scheme()
Q,f1,f2,f3=f.forward_scheme(L)



def nonLterm(x):
    nlterm=x**2
    return nlterm

g=-0.5j*k
def nonLterm_spectual(x):#x is in Fourier space
    yy=torch.real(torch.fft.ifft(x,norm=norm))
    nltm=nonLterm(yy)
    out=torch.fft.fft(nltm,norm=norm)
    out*=g
    out*=k_proj
    return out


tt=0
import time as tm
tt1=tm.time()
for n in range(1,nmax+1):
    t=n*timegrid
    nl_v=nonLterm_spectual(v)
    a=E_2*v+Q*nl_v
    nl_a=nonLterm_spectual(a)
    b = E_2 * v + Q * nl_a
    nl_b=nonLterm_spectual(b)
    c = E_2 * a + Q * (2*nl_b-nl_v)
    nl_c=nonLterm_spectual(c)
    v=E*v+nl_v*f1+2*(nl_b+nl_a)*f2+nl_c*f3

    if n%nplt==0:
        u=torch.real(torch.fft.ifft(v,norm=norm))
        ut=torch.cat((ut,u.unsqueeze(-2)),dim=-2)
        # vt=torch.cat((vt,v.unsqueeze(-2)),dim=-2)
        tt = np.hstack((tt, t))
        if t%1==0:
            print('t=',t)

tt2=tm.time()
# with open("ks_dns.txt", "w") as file:
#     file.write(f'1traj, T=150,dt=0.001\n{tt2-tt1}')
# exit()


ut.cpu()
torch.save(ut,file_name+'_ut.pt')
