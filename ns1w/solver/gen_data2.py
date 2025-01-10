import torch
import math

from timeit import default_timer

from pdes_periodic import NavierStokes2d
from random_fields import GaussianRF2d


import time as tm
torch.backends.cuda.max_split_size_mb = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

dsct=1 #originally: 2  Save the dsct-times downsampling of the snapshots to save memory.
L = 2*math.pi
s = 128 # spactio grid number. vor_save: s/dsct*s/dsct
N = 1# total traj
bsize = 1 # traj per epoch

dt_save=1 # Save snapshots every dt_save time interval.
re=1000 #reynolds number
start_save=0
t_traj_phy=500
t_traj=int(t_traj_phy/dt_save)
t_start=int(start_save/dt_save)


solver = NavierStokes2d(s,s,L,L,device=device,dtype=dtype)
grf = GaussianRF2d(s,s,L,L,alpha=2.5,tau=3.0,sigma=None,device=device,dtype=dtype)

t = torch.linspace(0, L, s+1, dtype=dtype, device=device)[0:-1]
_, Y = torch.meshgrid(t, t, indexing='ij')

f = -4*torch.cos(4.0*Y)

vor = torch.zeros(N,(t_traj-t_start)+1,s//dsct,s//dsct,dtype=torch.float32)


tt1=tm.time()
for j in range(N//bsize):
    w = grf.sample(bsize)

    # vor[j*bsize:(j+1)*bsize,0,:,:] = w[:,::dsct,::dsct].type(torch.float32)

    for k in range(t_traj):#5k
        if k==t_start:
            vor[j * bsize:(j + 1) * bsize, k - t_start, :, :] = w[:, ::dsct, ::dsct].cpu().type(torch.float32)
        t1 = default_timer()

        w = solver.advance(w, f, T=dt_save, Re=re, adaptive=True)
        if k>=t_start:
            vor[j*bsize:(j+1)*bsize,k-t_start,:,:] = w[:,::dsct,::dsct].cpu().type(torch.float32)

        t2 = default_timer()

        if k%(int(1/dt_save))==0:
            print(k*dt_save, t2-t1)


    # torch.save(vor[j*bsize:(j+1)*bsize,k+1,:,:], 'sample_2_' + str(j+1) + '.pt')

tt2=tm.time()




# ''' for random init only'''
# torch.save(vor.squeeze(dim=1), 'data/KF_re100_1000trj_16dx_random_ini(1).pt')

torch.save(vor, f'data/nu{re}_grid={s}_{dsct}_N={N}_dt={dt_save}_Ttj={start_save}-{t_traj_phy}.pt')