import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import My_TOOL as wcw


from ks.data_dict import *
import torch.fft as fft


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

aa=torch.load(data_dict['ks']['plot'][128])
bb=torch.load(data_dict['ks']['plot'][1024][400])


a=aa[:,200:,:]
b=bb[:,200:,:]
del aa
del bb


minn=[torch.min(a).item(),torch.min(b).item()]
maxx=[torch.max(a).item(),torch.max(b).item()]
umin=min(minn)
umax=max(maxx)
a=fft.rfft(a,norm='forward').abs()#n,t,k
b=fft.rfft(b,norm='forward').abs()#n,t,K
vmax=[]
for i in range(65):
    vmax.append(max(torch.max(a[:,:,i]).item(),torch.max(b[:,:,i]).item()))
for i in range(65,200):
    vmax.append(torch.max(b[:,:,i]).item())



torch.save({
    'u':[umin,umax],
    'v':vmax
},'../data/ks_stat_uv_emp_range.pt')

rr=torch.load('../data/ks_stat_uv_emp_range.pt')

wcw.check_dict(rr)
wcw.ppp(rr['v'])