import math as mt

data_base = '/pscratch/sd/y/ypincha/ks_ac/'  
model_base = '/pscratch/sd/y/ypincha/pino-closure-models/model_save/'

data_dict = {}

a = {
    'ut': {
        'link': data_base + "T=100,niu=0.01,N=128,dt=0.01,6pi,dtsave=0.06,sample=1600_ut.pt",
        'dtsave': 0.06,
        'N': 128
    }
}

b = {
    'ut': {
        'link': data_base + "T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=200(68)._test_ut.pt",
        'dtsave': 0.1,
        'N': 1024
    }
}

ks_dict = {
    1024: b,
    'hi': b,
    128: a,
    'low': a,
}

ks_dict['plot'] = {
    128: data_base + "T=100,niu=0.01,N=128,dt=0.01,6pi,dtsave=0.06,sample=1600_ut.pt",
    1024: {
        400: data_base + "T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=200(68)._test_ut.pt",
    },
    'range': data_base + 'ks_stat_uv_emp_range.pt'  # leave as-is, may need to generate
}

data_dict['ks'] = ks_dict

# PDE info
pde_info = {}
pde_ks = {
    'domain': [[0, 6*mt.pi]],
    'pde_dim': 1,
    'pde_dim_pino': 2,
    'function_dim': 1,
    'L': [6*mt.pi]
}
pde_info['ks'] = pde_ks

# Model paths (unchanged)
model_dict = {
    128: {200: model_base + 'model_cgs.pt'},
    1024: {200: model_base + 'model_frs.pt'},
    'pde': {'200_15k': model_base + 'model_pde.pt'}
}

