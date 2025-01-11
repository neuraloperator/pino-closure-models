import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

from ks.data_dict import *

import plot_stat_new as stat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cgs=torch.load('../data/stat_save/cgs.pt') # result for coarsegrid simulations
pino=torch.load('../data/stat_save/pino.pt') # results for pino model
sgs=torch.load('../data/stat_save/sgs.pt') # results for smagorinsky closure model
single=torch.load('../data/stat_save/single.pt') # results for single-state learning-based closure model

plotlist=[cgs,sgs,single,pino]
taglst=['CGS','Eddy-Visc.','Single','Ours']
stat.plot_all_stat(plotlist=plotlist,filename=f'cpr_fin',taglist=taglst,k_plot=[46,55],energy_k=61,acc_k=61,vds_k=61,
                   dns_tag='FRS')