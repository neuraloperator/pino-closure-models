import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



from ns1w.data_dict import *


import kf_plot_stat_new as stat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cgs=torch.load('../data/stat_save/cgs.pt')# course-grid simul without closure
pino=torch.load('../data/stat_save/pino.pt')# pino
sgs=torch.load('../data/stat_save/sgs.pt')# smagorinsky model
single=torch.load('../data/stat_save/single.pt')# single-state NN closure model
mff=torch.load('../data/stat_save/mff.pt')  # multi-fidelity FNO
dsm=torch.load('../data/stat_save/dsm.pt') # dynamical smagorinsky-lilly model

plotlist=[cgs,sgs,single,dsm,mff,pino]

taglst=['CGS','Smag.','Single','DSM','MFF','Ours']


stat.plot_all_stat(plotlist,filename=f"cpr_KF",energy_k=44,vds_k=[24,24],taglist=taglst,dns_tag='Ground Truth',k_plot=[[2,2],[5,6]])
