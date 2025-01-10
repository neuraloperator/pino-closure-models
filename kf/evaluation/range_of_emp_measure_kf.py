import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import My_TOOL as wcw


from kf.data_dict import *
import torch.fft as fft


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aa=torch.load(data_dict['stat'][100][16]['link'])
bb=torch.load(data_dict['stat_long'][100][128]['link'])


a=aa[:,200:,:]
b=bb[:,2:,:]
del aa
del bb


minn=[torch.min(a).item(),torch.min(b).item()]
maxx=[torch.max(a).item(),torch.max(b).item()]
umin=min(minn)
umax=max(maxx)
a=fft.fft2(a,norm='forward').abs()#n,t,k1,k2
b=fft.fft2(b,norm='forward').abs()#n,t,K1,k2
vmax=[[0 for i in range(16)]for j in range(16)]
for i in range(8):
    for j in range(8):
        vmax[i][j]=(max(torch.max(a[:, :, i,j]).item(), torch.max(b[:, :, i,j]).item()))


torch.save({
    'u':[umin,umax],
    'v':vmax
},'../data/kf_stat_uv_emp_range.pt')

rr=torch.load('../data/kf_stat_uv_emp_range.pt')

wcw.check_dict(rr)
wcw.ppp(rr['u'])