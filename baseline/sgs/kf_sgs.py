import time as tm
import torch
import math
from dict_ref import *
from timeit import default_timer

from kf_sgs_pdes_periodic import NavierStokes2d
from random_fields import GaussianRF2d
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')


torch.backends.cuda.max_split_size_mb = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64


re=100 # inverse of the coef of laplacian term, 100/ 1000
lknm='kf' #'kf' or 'ns1w'

dsct=1 #spatial downsample rate when saving the tensor
L = 2*math.pi
s = 16 # spatial resolution.vor_save: s/dsct*s/dsct; 48 for hi reynolds case
N = 100 # total traj
bsize =20 #20  # traj per iteration

dt_save=1 #1/4

start_save=1800
t_traj_phy=3000
cs=0.85j
dt0=1e2 # no use

t_traj=int(t_traj_phy/dt_save)
t_start=int(start_save/dt_save)



solver = NavierStokes2d(s,s,L,L,device=device,dtype=dtype)
grf = GaussianRF2d(s,s,L,L,alpha=2.5,tau=3.0,sigma=None,device=device,dtype=dtype)

t = torch.linspace(0, L, s+1, dtype=dtype, device=device)[0:-1]
_, Y = torch.meshgrid(t, t, indexing='ij')

f = -4*torch.cos(4.0*Y)

vor = torch.zeros(N,(t_traj-t_start)+1,s//dsct,s//dsct,dtype=torch.float32)
print(1)

for j in range(N//bsize):
    w = grf.sample(bsize)#n,x,y

    # vor[j*bsize:(j+1)*bsize,0,:,:] = w[:,::dsct,::dsct].type(torch.float32)

    print(11)
    tt1=tm.time()
    t1=default_timer()
    for k in range(t_traj):
        if k==t_start:
            vor[j * bsize:(j + 1) * bsize, k - t_start, :, :] = w[:, ::dsct, ::dsct].cpu().type(torch.float32)


        w = solver.advance(w, f, T=dt_save, Re=re, adaptive=True,clos=cs,delta_t0=dt0)
        if k>=t_start:
            vor[j*bsize:(j+1)*bsize,k-t_start,:,:] = w[:,::dsct,::dsct].cpu().type(torch.float32)

        if k%(int(1/dt_save))==0:
            t2 = default_timer()
            print(k*dt_save, t2-t1)

            t1=t2
    tt2=tm.time()



import sys
sys.path.append(path_sys[lknm])
import kf_plot_stat_new as stat

norm='forward'


filename=f'KF{re}_cs={cs},n={N},{[start_save,t_traj_phy]}'

#exisit

sgs = {'tag': 'sgs', 'n': N, 'res': s, 'in': 0, 'out': t_traj_phy,'data':vor,'dtsave':dt_save}

stat.save_stat_info(sgs,filename=filename,tag=sgs['tag'], default_flnm=0)
sgsdct = torch.load(filename + '.pt')
stat.save_all_err(sgsdct, filename=filename, default_flnm=0)

