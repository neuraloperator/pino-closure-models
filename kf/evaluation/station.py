import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import neuralop_base
sys.modules['neuralop'] = neuralop_base
import my_tools as wcw

from kf.data_dict import *
import torch.fft as fft
from neuralop import get_model
from neuralop.training import setup
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

'''import for every task'''
import time as tm
from kf.data.positional_encoding import get_grid_positional_encoding
import kf_plot_stat_new as stat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
counter_file = "counter.txt"
file_id=wcw.id_filename(counter_file)



config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./kf_plot.yaml", config_name="default", config_folder="../../config"
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

res= 16

sourcedata = {}
sourcedata[res]=torch.load(random_data[res],map_location=device)

n_use = config.plot.n_traj

traj_slice = slice(None, n_use)

x=sourcedata[res][traj_slice].unsqueeze(dim=1)#data:n,x,y
x = x.unsqueeze(dim=1)  # N*1*1*X*Y
# wcw.sss(x)


# """rfft and change to 1024 here"""

ltmp=list(sourcedata).copy()
for key in ltmp:
    del sourcedata[key]
del sourcedata



x=x.repeat(1,1,config.plot.repeat_ini,1,1).float()#n,1,t,x,y
gridd=get_grid_positional_encoding(x[0], grid_boundaries=config_arch.domain,
                                   dim_pde=config_arch.data_channels-pde_case['function_dim'],channel_dim=0)#1*1*gx*gy*..

"gridd[i]: 1*1*t_repeat*X*y"

def out_for_ini(x,gridd_tem): #x_shape: (n,1,1,X,y)

    x=x.repeat(1,1,config.plot.repeat_ini,1,1)
    return torch.cat([x]+gridd_tem,dim=1)#n,4,t,x,y





model = get_model(config)
model = model.to(device)
model_link=model_dict[config.wandb.model_type][config.wandb.model_use]
cpt=torch.load(model_link,map_location=device)
model_name=wcw.get_file_name(model_link)
model.load_state_dict(cpt["model"])
model.eval()
del cpt

gen_cycle=mt.ceil(cfg_plt.n_traj/cfg_plt.btz_traj)
usave=[[] for i in range(gen_cycle)]

xx=x.clone()

epochs=mt.ceil(config.plot.t_run/config.data.t_predict)
# print(config.plot.t_run)
# print(config.data.t_predict)

wcw.ppp(epochs)

t1=0
t2=0



for ii in range(gen_cycle):
    x=xx[ii*cfg_plt.btz_traj:(ii+1)*cfg_plt.btz_traj]
    usave[ii].append(x[...,-1:,:,:].squeeze(dim=1).cpu())#btz_traj,1,x,y '1' for last moment t


    ss=x.shape
    gridd_tem=[g.repeat(ss[0],1,1,1,1)for g in gridd]#n*1*t*x,y
    x=torch.cat([x]+gridd_tem,dim=1)


    with torch.no_grad():
        tt1=tm.time()
        for tt in range(epochs):#
            t1=tm.time()
            out=model(x)#out: n,1,t,x,y
            y=out[...,-1:,:,:]#n,1,1,x,y
            # usave.append(y.squeeze(dim=1).detach())
            if tt>config.plot.t_start_save:
                usave[ii].append(y.squeeze(dim=1).cpu())

            del out
            del x
            x=out_for_ini(y,gridd_tem=gridd_tem)

            t2=tm.time()

            if tt%40==0:
                print(f"epoch={tt}, time={t2-t1}")

        tt2=tm.time()


for ii in range(gen_cycle):
    usave[ii]=torch.cat(usave[ii],dim=-3)#btz,t,x,y
usave=torch.cat(usave,dim=0)#n t x,y


wcw.sss(usave)
wcw.mmm()

res=16


pino={'tag':'pino','n':400,'res':16,'in':2,'out':1200}
pino['data']=usave
flnm=f'pino{file_id}'
dft_flnm=1
stat.save_stat_info(pino,filename=flnm,tag=pino['tag'],default_flnm=dft_flnm)

del usave

linkb='../data/stat_save/'+flnm+'.pt'
pino_stat = torch.load(linkb)

# relpth='../data/stat_save/' if dft_flnm else ''


stat.save_all_err(pino_stat,filename=f'pino_{file_id}',default_flnm=1)


