data_dict={}
#Part1: dataset

'''
Name:
[1]equation:  ks/kf/ns
[2]resolution(number or adj=hi/low); train or test
[3]ut or vt
[4]link or dtsave
'''

rel_path='data/'
# rel_path=''
a={
        'ut':{'link':rel_path+"T=100,niu=0.01,N=128,dt=0.01,6pi,dtsave=0.1,sample=1600.pt",'dtsave':0.1,"N":128} #dtsave: save snapshots for all t=k*dtsave; N: spatial resolution
    }

b={
    'ut':{'link':rel_path+"T=50,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.025,sample=4.pt",'dtsave':0.025,'N':1024},
}
ks_dict={
    1024:b,'hi':b,128:a,'low':a,
}


rel_path='data/'
ks_dict['plot']={
    128:rel_path+"T=100,niu=0.01,N=128,dt=0.01,6pi,dtsave=0.1,sample=1600.pt",
    1024:{
        400:rel_path+"T=100,niu=0.01,N=1024,dt=0.001,6pi,dtsave=0.1,sample=400(69)._test_ut.pt", # ground truth: 400 traj
    },
    'range':rel_path+'ks_stat_uv_emp_range.pt'
}

data_dict['ks']=ks_dict

import math as mt
'''
###pde information'''
pde_info={}
pde_ks={
    'domain':[[0,6*mt.pi]],
    'pde_dim':1,
    'pde_dim_pino':2,
    'function_dim':1,
    'L':[6*mt.pi]
}

pde_info['ks']=pde_ks

'''Model list'''
rel_path='model_save/'
model128={200:rel_path+'model_cgs.pt',}
model1024={200:rel_path+'model_frs.pt'}
model_pde={
           '200_15k':rel_path+'model_pde.pt'
           }
model_dict={128:model128,1024:model1024,'pde':model_pde}