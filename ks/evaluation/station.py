import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import neuralop
import my_tools as myt
# import wandb
from ks.data_dict import *
import torch.fft as fft
from neuralop import get_model
from neuralop.training import setup
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

'''import for every task'''
import time as tm
from ks.data.positional_encoding import get_grid_positional_encoding

import plot_stat_new as stat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

counter_file = "counter.txt"
file_id=myt.id_filename(counter_file)



config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ks_pino_plot.yaml", config_name="default", config_folder="../../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()


device, is_logger = setup(config)

arch = config["arch"].lower()
config_arch = config.get(arch)

pde_case=pde_info[config.wandb.pde]
pino_t_tag=0
if 'pino' in config.wandb and config.wandb.pino:
    config_arch.data_channels=pde_case['pde_dim_pino']+pde_case['function_dim']
    config_arch.domain=[[0,config.data.t_predict]]+pde_case['domain']
    if pde_case['pde_dim_pino']!=pde_case['pde_dim']:
        pino_t_tag=1

else:
    config_arch.data_channels=pde_case['pde_dim']+pde_case['function_dim']
    config_arch.domain = pde_case['domain']


'''Load in random initialized data'''

cfg_plt=config.plot


"start with random initial value"
res=128
sourcedata = {}
sourcedata[128] = torch.load('../data/ks_128_1500_random_init.pt')

n_use = config.plot.n_traj

traj_slice = slice(None, n_use)

x=sourcedata[128][traj_slice].unsqueeze(dim=1)#n,x

x = x.unsqueeze(dim=1)  # N*1*1*X
myt.sss(x)


for key in [128]:
    del sourcedata[key]
del sourcedata
if config.plot.use_hi:
    x = fft.irfft((fft.rfft(x, norm='forward')), n=1024)



x=x.repeat(1,1,config.plot.repeat_ini,1).float()#n,1,t,x
gridd=get_grid_positional_encoding(x[0], grid_boundaries=config_arch.domain,
                                   dim_pde=config_arch.data_channels-pde_case['function_dim'],channel_dim=0)#1*1*gx*hy*..

"gridd[i]: 1*1*t_repeat*X"

def out_for_ini(x,gridd_tem): #x_shape: (n,1,1,X)

    x=x.repeat(1,1,config.plot.repeat_ini,1)
    return torch.cat([x]+gridd_tem,dim=1)#n,3,t,x



model = get_model(config)
model = model.to(device)
model_link=model_dict[config.wandb.model_type][config.wandb.model_use]
cpt=torch.load(model_link,map_location=device)
model_name=myt.get_file_name(model_link)
model.load_state_dict(cpt["model"])
model.eval()
del cpt

gen_cycle=mt.ceil(cfg_plt.n_traj/cfg_plt.btz_traj)
usave=[[] for i in range(gen_cycle)]

xx=x.clone()

epochs=mt.ceil(config.plot.t_run/config.data.t_predict)
print(config.plot.t_run)
print(config.data.t_predict)



t1=0
t2=0

for ii in range(gen_cycle):
    x=xx[ii*cfg_plt.btz_traj:(ii+1)*cfg_plt.btz_traj]
    usave[ii].append(x[...,-1:,:].squeeze(dim=1).cpu())#btz_traj,1,x, '1' for last moment t

    ss=x.shape
    gridd_tem=[g.repeat(ss[0],1,1,1)for g in gridd]#n*1*t*x
    x=torch.cat([x]+gridd_tem,dim=1)


    with torch.no_grad():
        tt1=tm.time()
        for tt in range(epochs):#
            t1=tm.time()
            out=model(x)#out: n,1,t,x
            y=out[...,-1:,:]#n,1,1,x
            usave[ii].append(y.squeeze(dim=1).cpu())

            del out
            del x
            x=out_for_ini(y,gridd_tem=gridd_tem)

            t2=tm.time()

            if tt%40==0:
                print(f"epoch={tt}, time={t2-t1}")

        tt2=tm.time()


for ii in range(gen_cycle):
    usave[ii]=torch.cat(usave[ii],dim=-2)
usave=torch.cat(usave,dim=0)#n t x



modelname=myt.get_file_name(model_link)
res=1024 if cfg_plt.use_hi else 128

start=cfg_plt.starting_plot
filename=f"{modelname}_res={res}"

stat.save_stat_info(dt_save=0.1,filename=filename,tag='model',start_time=start,ut=usave,use_link=0)

linkb='../data/stat_save/'+filename+'.pt'
statdct = torch.load(linkb)

stat.save_all_err(statdct,filename='pino',default_flnm=1)
