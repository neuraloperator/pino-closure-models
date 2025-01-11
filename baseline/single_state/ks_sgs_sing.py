import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from ks_ParaCtrl_sing import *
import math as mt
import torch.fft as fft

import wandb
from dict_ref import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gridx=torch.arange(start=1,end=N+1,step=1,dtype=torch.float64).to(device)/N*space_scaling
# gridx=torch.arange(start=1,end=N+1,step=1).to(device)/N*space_scaling
torch.set_printoptions(precision=8)

k=torch.cat((torch.arange(start=0,end=N//2+1),torch.arange(-N//2+1,0))).to(device)/half_period   #** changed from original
k_proj=torch.cat((torch.ones(N_proj//2+1),torch.zeros(N-N_proj),torch.ones(N_proj//2-1))).to(device)

class initial_value_lib:
    '''
    TO use it, first define an object (A) of this class.
    A=initial_value_lib()
    Whenever I want an i.c. function,
    A(x)
    '''

    def __init__(self,choose=choose_ic,alpha=2.0, tau=3.0,sigma=None):
        self.null_mode=max(round(half_period/mt.sqrt(niu)),1)#L_k~0 (L is the diag linear operator operated after fft)
        self.worst_mode=max(round(half_period/mt.sqrt(2*niu)),1)#L_k is the largest
        self.choose=choose
        self.s1=N
        self.device=device
        # if choose==5:
        #     self.random_coef=torch.randn(5)
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


ut=(u0.clone()).unsqueeze(-2) #spectral #track and save the (u_t) traj


v=torch.fft.fft(u0,norm=norm)
vt=(v.clone()).unsqueeze(-2)#spectral


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
        lr=timegrid*(x.unsqueeze(1)+r)  #N*m  -(57)

        lr=x.unsqueeze(1)+r
        e_lr=torch.exp(lr/2)
        e_lr2=e_lr**2
        q = torch.real(torch.mean(dim=1, input=(e_lr - 1) / lr))
        f1 =torch.real(torch.mean(dim=1, input=((-4 - lr + e_lr2 * (4 - 3 * lr + lr ** 2))) / (lr ** 3)))
        f2 =torch.real(torch.mean(dim=1, input=((2 + lr + e_lr2 * (lr - 2))) / (lr ** 3)))
        f3 =torch.real(torch.mean(dim=1, input=((-4 - 3 * lr - lr ** 2 + e_lr2 * (4 - lr))) / (lr ** 3)))

        return q,f1,f2,f3
    def ff_exact(self,x):
        qz=(torch.exp(x / 2) - 1) / x
        f1z=(torch.exp(x) * (4 - 3 * x + x ** 2) - 4 - x) / (x ** 3)
        f2z=(2 + x + torch.exp(x) * (x - 2)) / (x ** 3)
        f3z=(-4 - 3 * x - x ** 2 + torch.exp(x) * (4 - x)) / (x ** 3)
        return qz,f1z,f2z,f3z
    def ff_taylor_3(self,x):
        q=0.5 + x / 8 + x ** 2 / 48 + x ** 3 / 384
        f1=(1 + x + 9 * x ** 2 / 20 + 2 * x ** 3 / 15) / 6
        f2=(30 + 15 * x + 4.5 * x ** 2 + x ** 3) / 180
        f3=1 / 6 - (x + 3) * (x ** 2) / 360
        return q,f1,f2,f3

    def ff_ffexact(self, x):
        q, f1, f2, f3 = self.ff(x)
        zq, zf1, zf2, zf3 = self.ff_exact(x)
        q = torch.where(torch.abs(x) < eta_stb_change, q, zq)
        f1 = torch.where(torch.abs(x) < eta_stb_change, f1, zf1)
        f2 = torch.where(torch.abs(x) < eta_stb_change, f2, zf2)
        f3 = torch.where(torch.abs(x) < eta_stb_change, f3, zf3)
        return q, f1, f2, f3

    def exact_taylor(self, x):
        q, f1, f2, f3 = self.ff_taylor_3(x)
        zq, zf1, zf2, zf3 = self.ff_exact(x)
        q = torch.where(torch.abs(x) < eta_stb_change, q, zq)
        f1 = torch.where(torch.abs(x) < eta_stb_change, f1, zf1)
        f2 = torch.where(torch.abs(x) < eta_stb_change, f2, zf2)
        f3 = torch.where(torch.abs(x) < eta_stb_change, f3, zf3)
        return q, f1, f2, f3

    functions={
        0:ff,1:ff,
        10:ff_ffexact,
        30:exact_taylor,
        3:ff_taylor_3
    }
    def forward(self, x):
        # print(f"forward,{x.type}")
        return self.functions[self.choose](self,x)
    def forward_scheme(self,x):
        y=self.forward(timegrid*x)
        return tuple(timegrid*i for i in y)




f=coeff_scheme()
Q,f1,f2,f3=f.forward_scheme(L)


from utilities3 import *
from utilities import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# obtain the normalizing encoder-decoder
a = torch.load('KS_train_set.pt',map_location=device)

b=a**2
c=fft.irfft(fft.rfft(b,norm='forward')[:65], norm='forward')  # shape: 4*351*128
f_a=fft.irfft(fft.rfft(a,norm='forward')[:65], norm='forward')  # 4,351,128
d=c-f_a**2  #4,351,128
input_xTr=a[:3,::10,::8].clone().reshape(-1,128) #105,128
input_yTr=d[:3,::10,::8].clone().reshape(-1,128) #105,128
x_normalizer = UnitGaussianNormalizer(input_xTr)
y_normalizer = UnitGaussianNormalizer(input_yTr)
del d,f_a,c,b,a,input_yTr,input_xTr


from vision_transformer import vit_b_ks
model = vit_b_ks(num_classes=128).cuda()  # nn.Transformer(nhead=16, num_encoder_layers=12)
model.load_state_dict(torch.load('ks_model.pt'), strict=True)
model.eval()


def nonLterm(x):
    nlterm=x**2


    x=x_normalizer.encode(x).float().unsqueeze(1).unsqueeze(1)#n,1,1,x

    with torch.no_grad():
        out=model(x)#n,x
        y=y_normalizer.decode(out)#n,x
        nlterm+=y

    return nlterm#n,x

g=-0.5j*k
def nonLterm_spectual(x):#x is in Fourier space

    yy=torch.real(torch.fft.ifft(x,norm=norm))
    nltm=nonLterm(yy)
    out=torch.fft.fft(nltm,norm=norm)
    out*=g
    return out

tt=0
import time as tm
tt1=tm.time()
for n in range(1,nmax+1):
    t=n*timegrid
    nl_v=nonLterm_spectual(v) #nx
    a=E_2*v+Q*nl_v
    nl_a=nonLterm_spectual(a)
    b = E_2 * v + Q * nl_a
    nl_b=nonLterm_spectual(b)
    c = E_2 * a + Q * (2*nl_b-nl_v)
    nl_c=nonLterm_spectual(c)
    v=E*v+nl_v*f1+2*(nl_b+nl_a)*f2+nl_c*f3

    # if n%1==0:
    if n%nplt==0:

        u=torch.real(torch.fft.ifft(v,norm=norm))
        # print(u)
        ut=torch.cat((ut,u.unsqueeze(-2)),dim=-2)
        # vt=torch.cat((vt,v.unsqueeze(-2)),dim=-2)
        tt = np.hstack((tt, t))

tt2=tm.time()
print(tt2-tt1)


ut.cpu()

import sys
sys.path.append(path_sys['ks'])
import plot_stat_new as stat





filename=f'ks_sgs_singel_n={Nsum},{[start_plot_time,tmax]}'

stat.save_stat_info(dt_save=dt_save,filename=filename,tag='SGS',start_time=start_plot_time,ut=ut,use_link=0,default_flnm=0)

sgsdct=torch.load(filename+'.pt')
stat.save_all_err(sgsdct,filename=filename,default_flnm=0)



