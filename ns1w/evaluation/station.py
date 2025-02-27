import torch
import numpy
import wandb
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
import neuralop_advance
sys.modules['neuralop'] = neuralop_advance
import my_tools as wcw

from ns1w.data_dict import *
import torch.fft as fft
from neuralop import get_model
from neuralop.training import setup
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from neuralop.utils import get_wandb_api_key, count_model_params
'''import for every task'''
import time as tm
from ns1w.data.positional_encoding import get_grid_positional_encoding
import kf_plot_stat_new as stat


counter_file = "counter.txt"
file_id=wcw.id_filename(counter_file)



config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ns1w_plot.yaml", config_name="default", config_folder="../config"
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

'''use wandb, record config'''
if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    print("success!!!!")
    print(type(config))
    # if config.wandb.name:
    #     wandb_name = config.wandb.name+f"({file_id})"
    if 1:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                str(file_id),
                str('plot'),
                config.data.t_predict,
                config.wandb.model_type,
                config.wandb.model_use,
                config.plot.n_traj,
                str('use_hi'),
                config.plot.use_hi
            ]  # +f"({file_id})"
        )

    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )

    wandb.init(name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
               config={
                   'model':{'type':config.wandb.model_type,'id':config.wandb.model_use},
                   'Re':config.wandb.re,
                   'data_plot_sv':{'n_traj':config.plot.n_traj,'btz_trj':config.plot.btz_traj,
                                   't_run(s)':config.plot.t_run,'repeat_ini':config.plot.repeat_ini},
                   't_prd':config.data.t_predict
               })


    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]
    wandb.finish()


cfg_plt=config.plot

"start with random initial value"


res=cfg_plt.res_ini
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
ep_start_sv=mt.ceil(cfg_plt.t_start_save/config.data.t_predict)
# print(config.plot.t_run)
# print(config.data.t_predict)
# print(ep_start_sv)
wcw.ppp(epochs)

t1=0
t2=0



for ii in range(gen_cycle):
    x=xx[ii*cfg_plt.btz_traj:(ii+1)*cfg_plt.btz_traj]
    usave[ii].append(x[...,-1:,:,:].squeeze(dim=1).cpu())#btz_traj,1,x,y '1' for last moment t


    ss=x.shape
    gridd_tem=[g.repeat(ss[0],1,1,1,1)for g in gridd]#n*1*t*x,y
    x=torch.cat([x]+gridd_tem,dim=1)

    tt1 = tm.time()

    with torch.no_grad():
        for tt in range(epochs):#
            t1=tm.time()
            out=model(x,test=1)#out: n,1,t,x,y

            y=out[...,-1:,:,:]#n,1,1,x,y


            if tt>ep_start_sv:
                usave[ii].append(y.squeeze(dim=1).cpu())

            del out
            del x
            x=out_for_ini(y,gridd_tem=gridd_tem)

            t2=tm.time()



            if tt%20==0:
                print(f"epoch={tt*config.data.t_predict}, time={t2-t1}")

tt2=tm.time()


for ii in range(gen_cycle):
    usave[ii]=torch.cat(usave[ii],dim=-3)#btz,t,x,y
usave=torch.cat(usave,dim=0)#n t x,y
wcw.sss(usave)

if config.wandb.log and is_logger:
    wandb.finish()

wcw.sss(usave)
wcw.mmm()
res=cfg_plt.res

tagsv=f'fno{config.wandb.model_type}_{config.wandb.model_use}'
pino16k={'tag':tagsv,'n':cfg_plt.n_traj,'res':res,'in':0,'out':cfg_plt.t_run,'dtsave':config.data.t_predict}
pino16k['data']=usave
flnm=f'{tagsv}_{config.wandb.model_use}_t={[cfg_plt.t_start_save,cfg_plt.t_run]}_n={cfg_plt.n_traj}_ini={cfg_plt.res_ini}'

with open("pino_run.txt", "w") as file:
    file.write(f'{flnm}\n{tt2 - tt1}')

dft_flnm=1
stat.save_stat_info(pino16k,filename=flnm,tag=pino16k['tag'],default_flnm=dft_flnm)

del usave

relpth='../data/stat_save/' if dft_flnm else ''
pino_stat=torch.load(relpth+flnm+'.pt')




stat.save_all_err(pino_stat,filename=f'pino_{file_id}',default_flnm=1)


