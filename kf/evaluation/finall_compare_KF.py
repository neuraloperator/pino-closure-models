import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from kf.data_dict import *
import kf_plot_stat_new as stat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cgs=torch.load('../data/stat_save/cgs.pt')
pino=torch.load('../data/stat_save/pino.pt')
sgs=torch.load('../data/stat_save/sgs.pt')
single=torch.load('../data/stat_save/single.pt')
mff=torch.load('../data/stat_save/mff.pt')
dsm=torch.load('../data/stat_save/dsm.pt')

plotlist=[cgs,sgs,single,dsm,mff,pino]

taglst=['CGS','Smag.','Single','DSM','MFF','Ours']
stat.plot_all_stat(plotlist,filename=f"cpr_KF",vds_k=[7,7],taglist=taglst,dns_tag='FRS(ground truth)')
