import builtins
original_print = builtins.print
def print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    original_print(*args, **kwargs)

import torch
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys

import neuralop_advance
sys.modules['neuralop'] = neuralop_advance

import wandb

import torch.fft as fft

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig

from torch.nn.parallel import DistributedDataParallel as DDP
from neuralop import get_model
from neuralop.training import setup
from neuralop.training.callbacks import MGPatchingCallback, SimpleWandBLoggerCallback
from neuralop.utils import get_wandb_api_key, count_model_params
import my_tools as wcw

from t2_trainer_kf_2 import Trainer
from data_dict import *
from losses import *
from data.kf_data_load_pino import load_data_small

#basic control
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
counter_file = "counter.txt"
file_id=wcw.id_filename(counter_file)

# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./ns1w_2.yaml", config_name="default", config_folder="../config"
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder="../config"),
    ]
)
config = pipe.read_conf()

config_name = pipe.steps[-1].config_name

config.data.n_train=config.data.num_data[:2]
config.data.n_tests=config.data.num_data[2:]
config.data.train_resolution=config.data.resolution[0]


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

if config.wandb.log and is_logger:
    wandb.login(key=get_wandb_api_key())
    print("success!!!!")
    if 1:
        wandb_name = "_".join(
            f"{var}"
            for var in [
                str(file_id),
                config.wandb.loss,
                config.data.t_predict,
                config.data.num_data,
                config.opt.training_loss,
                config.opt.learning_rate,
                config_arch.hidden_channels,
                config_arch.n_modes,
            ]  # +f"({file_id})"
        )

    wandb_args = dict(
        config=config,
        name=wandb_name,
        group=config.wandb.group,
        project=config.wandb.project,
        entity=config.wandb.entity,
    )


    if config.wandb.sweep:
        for key in wandb.config.keys():
            config.params[key] = wandb.config[key]

# Make sure we only print information when needed
config.verbose = config.verbose and is_logger



wcw.ppp(is_logger)

# Print config to screen
if config.verbose and is_logger:
    pipe.log()
    sys.stdout.flush()
'''Setting dataset'''

import time as tm
t1=tm.time()


alllink={32:data_dict['data'][0][32],256:data_dict['data'][0][256],
         'pde':data_dict['data'][0]['pde']}
sourcedata={}
for i in [32,256]:
    sourcedata[i]=torch.load(alllink[i]['link'])#,map_location=device)
    print('source data at ',sourcedata[i].device)
alldata={'train':[],'test':[]}

def gen_data_pad(x,dt,dim=1,dT=None,t0:int=0,t_end:int=None,single=0,rr=None):#x.shape=NTXY ,dim(T)=1

    y=wcw.slicing_split(x,dim=dim,dt=dt,dT=dT,t0=t0,t_end=t_end,single=single) #shape=NKTXY
    if rr==None:
        if y.shape[2]*config.data.repeat>=config.data.t_step_min:
            rr=config.data.repeat
        else:
            rr=mt.ceil(config.data.t_step_min/y.shape[2])
    y=torch.repeat_interleave(y,dim=2,repeats=rr)
    return y,rr #NKTXY

maxdtuse=max([i.shape[1]for i in sourcedata.values()])+2


for i in range(len(config.data.t_start)):
    print(f'\n!!!i={i}')
    id_res = config.data.resolution[i]
    dtsave=alllink[id_res]['dtsave']
    ntraj=alllink[id_res]['N']
    t_init=config.data.t_start[i]
    dt_use=config.data.t_interval[i]  # check: if dt_use=0, only use data at t0
    if dt_use<=0:
        dt_use=maxdtuse
    t_pred=config.data.t_predict  # check: if t_pred// dt==0: raise error
    t_use_init = int(t_init / dtsave)
    t_use_per = int(dt_use / dtsave)
    t_use_pred = int(t_pred / dtsave)
    if t_use_pred==0:
        print("error: main_data: predicting identity")
        exit()
    b=sourcedata[id_res]


    # reduce the peak CPU memory: 09/09
    traj_id = 0 if id_res==32 else 1
    n_traj_use=ntraj-config.data.traj_for_train[traj_id] if config.data.train_tag[i]==0 else config.data.traj_for_train[traj_id]
    T_per_use=math.ceil(config.data.num_data[i]/n_traj_use)
    T_keep=(T_per_use+1)*t_use_per+t_use_pred

    # if config.data.train_tag[i]>0:
    #     time_slice=slice(None,T_per_use)
    # else:
    #     time_slice=slice(T_per_use,None)
    time_slice = slice(None, T_keep)


    if config.data.train_tag[i] > 0:
        # time_slice=slice(None,T_per_use)
        traj_slice = slice(None, config.data.traj_for_train[traj_id])
    else:
        # time_slice=slice(T_per_use,None)
        traj_slice = slice(config.data.traj_for_train[traj_id], None)
    # y,repeat_y_time = gen_data_pad(b[traj_slice], dim=1, t0=t_use_init, dT=t_use_per, dt=[1, t_use_pred + 1],
    #                  single=0)
    dsp=config.data.dns_dsp
    b=b[traj_slice,t_use_init:,::dsp,::dsp]
    b=b[:,time_slice]
    y,repeat_y_time=gen_data_pad(b[:,1:],dim=1,t0=0,dT=t_use_per,dt=[0,t_use_pred])

    ss = y.shape
    rr = ss[2]

    x = gen_data_pad(b, dim=1, t0=0, dT=t_use_per, dt=0, single=1, rr=rr)[0]
    x = x[:, :ss[1]]

    save_to='train'if (config.data.train_tag[i]==1) else 'test'

    wcw.sss(x)
    wcw.sss(y)
    alldata[save_to].append({'x': x.reshape(-1,x.shape[-3],b.shape[-2],b.shape[-1]),'y':y.reshape(-1, x.shape[-3],b.shape[-2],b.shape[-1]),
                             't_val':repeat_y_time})#real data: [-1::t_val]
    wcw.ppp(repeat_y_time)
    # print(rr)

t2=tm.time()
print('_________________________________________________________')
print(f"time to load data:{t2-t1}")

for key in [32,256]:
    del sourcedata[key]
del sourcedata

# Loading the dataset

train_loaders, test_loaders, output_encoder = load_data_small(
    n_train=config.data.n_train,
    batch_size=config.data.batch_size,
    positional_encoding=config.data.positional_encoding,
    test_resolutions=config.data.test_resolutions,
    n_tests=config.data.n_tests,
    test_batch_sizes=config.data.test_batch_sizes,
    encode_input=config.data.encode_input,
    encode_output=config.data.encode_output,
    train_resolution=config.data.train_resolution,
    grid_boundaries=config_arch.domain,
    dim_pde=config_arch.data_channels-pde_case['function_dim'],
    in_data=alldata,
)


for key in ['train','test']:
    for i in alldata[key]:
        for k in list(i.keys()):
            del i[k]
    del alldata[key]
del alldata

model = get_model(config,dvc=device)
model = model.to(device)
if 'model_use_type' in config.wandb:
    cpt = torch.load(model_dict[config.wandb.model_use_type][config.wandb.model_use], map_location=device)
    model.load_state_dict(cpt["model"])
    del cpt


# Use distributed data parallel
if config.distributed.use_distributed:
    model = DDP(
        model, device_ids=[device.index], output_device=device.index, static_graph=True
    )

# Create the optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.opt.learning_rate,
    weight_decay=config.opt.weight_decay,
)

if config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Creating the losses
# l2loss = LpLoss(d=2, p=2,reductions=config.opt.loss_type)
l2loss = LpLoss(d=pde_case['pde_dim'], p=2,L=pde_case['L'],reductions=config.opt.loss_type,pino_t=pino_t_tag,
                # reduce_dims=[0])
                reduce_dims=[0,2])
# h1loss = H1Loss(d=2,reductions=config.opt.loss_type)
h1loss = H1Loss(d=pde_case['pde_dim'],L=pde_case['L'],reductions=config.opt.loss_type,pino_t=pino_t_tag,fix_x_bnd=False,
                # reduce_dims=[0])
                reduce_dims=[0,2])
#fix_bnd=false because this is H1 only for x-dim.

eval_losses = {"h1": h1loss, "l2": l2loss}


"""Training Loss"""
# if type(config.opt.training_loss)==list and len(config.opt.training_loss)==1:
#     config.opt.training_loss=config.opt.training_loss[0]

train_losses=[]
aa=config.opt.training_loss
'follow the dataset order'
for i in range(len(aa)):
    if aa[i]=='l2':
        loss_tmp = LpLoss(d=pde_case['pde_dim'], p=2, L=pde_case['L'], reductions=config.opt.loss_type, pino_t=pino_t_tag,
                        # reduce_dims=[0])
                        reduce_dims=[0, 2],func=config.opt.train_loss_func[i])
        train_losses.append(loss_tmp)
    elif aa[i]=='h1':
        loss_tmp = H1Loss(d=pde_case['pde_dim'],L=pde_case['L'],reductions=config.opt.loss_type,pino_t=pino_t_tag,fix_x_bnd=False,
                # reduce_dims=[0])
                reduce_dims=[0,2],func=config.opt.train_loss_func[i])
        train_losses.append(loss_tmp)
    else:
        raise ValueError(
            f'Got training_loss={config.opt.training_loss} '
            f'but expected one of ["l2", "h1"]'
        )



if config.verbose and is_logger:
    print("\n### MODEL ###\n", model)
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULER ###\n", scheduler)
    print("\n### LOSSES ###")
    print(f"\n * Train: {train_losses}")
    print(f"\n * Test: {eval_losses}")
    print(f"\n### Beginning Training...\n")
    sys.stdout.flush()


with open('../config/wandb_api_key.txt', 'r') as file:
    api_key = file.read().strip()

wandb.login(key=api_key)
print("!!!!!!!! wandb login")

if 'gradient_accumulation' not in config.opt:
    config.opt['gradient_accumulation']=0

trainer = Trainer(
    model=model,
    n_epochs=config.opt.n_epochs,
    device=device,
    amp_autocast=config.opt.amp_autocast,
    wandb_log=config.wandb.log,
    log_test_interval=config.wandb.log_test_interval,
    log_output=config.wandb.log_output,
    use_distributed=config.distributed.use_distributed,
    verbose=config.verbose and is_logger,
    callbacks=[
        MGPatchingCallback(levels=config.patching.levels,
                                  padding_fraction=config.patching.padding,
                                  stitching=config.patching.stitching,
                                  encoder=output_encoder),
        SimpleWandBLoggerCallback(**wandb_args)
              ]
              )

# Log parameter count
if is_logger:
    n_params = count_model_params(model)

    if config.verbose:
        print(f"\nn_params: {n_params}")
        sys.stdout.flush()

    if config.wandb.log:
        to_log = {"n_params": n_params}
        if config.n_params_baseline is not None:
            to_log["n_params_baseline"] = (config.n_params_baseline,)
            to_log["compression_ratio"] = (config.n_params_baseline / n_params,)
            to_log["space_savings"] = 1 - (n_params / config.n_params_baseline)
        wandb.log(to_log)
        wandb.watch(model)

trainer.train(
    train_loaders=train_loaders,
    test_loaders=test_loaders,
    optimizer=optimizer,
    scheduler=scheduler,
    regularizer=False,
    training_loss=train_losses,
    eval_losses=eval_losses,
    losstype=config.opt.loss_type,
    K128=config.data.pino2_batch_num, #The number of cgs data batches together with one frs dara batch.
    lam128=config.data.pino2_batch_lambda, # Balancing the weight between frs data loss and cgs data loss.
    check_mem=config.opt.check_mem,
    lambda_scaling=config.data.pino2_btz_lam, #dict[scale, gamma, end_scale]
    grad_acml=config.opt.gradient_accumulation,
    quick_save=config.opt.quick_save,
    qck_sv_nm=f'model_save/model_dns_{file_id}'
)


if config.wandb.log and is_logger:
    wandb.finish()



'''test_ save model-----------------------------------------------'''
if config.wandb.savemd:
    print('start save')
    torch.save({'model':model.state_dict()},f'model_save/model_dns({file_id}).pt')
    print('ok')
