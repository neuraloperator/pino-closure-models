import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import My_TOOL as wcw
sinc=wcw.split_input_chkpt_advanced

from ns1w.data_dict import *

import builtins
original_print = builtins.print
def print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    original_print(*args, **kwargs)
'''import for every task'''

import time as tm
print('start')
a=torch.load(data_dict['stat'][0][48]['link'])
b=torch.load(data_dict['stat'][0][256]['link'])
print('load:ok')




t1=tm.time()
minn=[torch.min(a).item(),torch.min(b).item()]
maxx=[torch.max(a).item(),torch.max(b).item()]
umin=min(minn)
umax=max(maxx)
f1=torch.fft.fft2
splt_fft2=sinc(f1,splt=50,dim=0,dvc_cpt='cpu',fwd_dtype=torch.complex64)
la=a.shape[-1]
lb=b.shape[-1]
a=splt_fft2(a.view(-1,la,la),norm='forward').abs()#nt,k1,k2
print('a:ok')
b=splt_fft2(b.view(-1,lb,lb),norm='forward').abs()#nt,K1,k2
print('b:ok')
base_s=64 # input should be 48
vmax=[[0 for i in range(base_s)]for j in range(base_s)]
for i in range(base_s//2+1):
    for j in range(base_s//2+1):
        vmax[i][j]=(max(torch.max(a[:, i,j]).item(), torch.max(b[:, i,j]).item()))

t2=tm.time()
print('time',t2-t1)

torch.save({
    'u':[umin,umax],
    'v':vmax
},'../data/kf_stat_uv_emp_range.pt')

rr=torch.load('../data/kf_stat_uv_emp_range.pt')

wcw.check_dict(rr)
wcw.ppp(rr['u'])