data_dict={}
#Part1: dataset

'''
Name:
[1]'data' or 'stat'
[2] Select 0 for the default (and only) key. 
[3]resolution(number or adj=hi/low);
[4]link , dtsave , N (N_traj), T(traj end time)
'''

rel_path='data/'
'''train/test dataset'''
a={
   0: {  # pred=1/2
       32:{'link':rel_path+"filename_cgs.pt",'N':100,'dtsave':1/32,'T':300,'res':32}, #dtsave: save snapshots for all t=k*dtsave; res: spatial resolution of the dataset (number of grids along each dimension. It should match the tensor shape in the dataset when data are downsampled and saved); N: number of trajectories; T: the traj are run for [0,T]
       256:{'link':rel_path+"filename_frs.pt",'N':1,'dtsave':1/64,'T':385,'res':128},
       'pde': {'link': rel_path + "filename_pde_input.pt", 'N': 20, 'dtsave': 1, 'T': 170, 'res': 256},
   },
   }
data_dict['data']=a
b={
    0:{
       256:{'link':rel_path+"stat_dataset_frs.pt",'N':40,'dtsave':1/4,'T':500,'res':256},
        48: {'link': rel_path + "stat_dataset_cgs.pt", 'N': 100, 'dtsave': 1/4, 'T': 500, 'res': 48,
            'init': 320},

   }
   }

data_dict['stat']=b


import math as mt
'''
###pde information'''
pde_info={}
pde_kf={
    'domain':[[0,2*mt.pi],[0,2*mt.pi]],
    'pde_dim':2,
    'pde_dim_pino':3,
    'function_dim':1,
    'L':[2*mt.pi,2*mt.pi]
}
pde_info['kf']=pde_kf


'''Evaluation'''
rel_path='data/'
random_data={48:rel_path+'random_init_course.pt'}

all_stat={
    'dns':'/data/stat_save/dns_256.pt',
    'les':'data/stat_save/les_48.pt',

}

'''Model list'''
rel_path='model_save/'
model1={0:rel_path+"model_kf_cgs.pt"} # model after stage 1: supervised loss with CGS data.
model2={0:rel_path+"model_kf_frs.pt"}# model after stage 2: supervised loss with FRS data.
model3={0: rel_path+'model_pde.pt'}# model after stage 3: pde loss.
model_dict={1:model1,2:model2,3:model3}

