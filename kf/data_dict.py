data_dict={}
#Part1: dataset

'''
Name:
[1]'data' or 'stat'
[2]Reynolds
[3]resolution(number or adj=hi/low);
[4]link , dtsave , N (N_traj), T(traj end time)
'''

rel_path='data/'
'''train/test dataset'''
#CGS: coarse-grid simulation, FRS: fully-resolved simulation
a={
   100:{16:{'link':rel_path+"filename_cgs.pt",'N':100,'dtsave':1/32,'T':500,'res':16}, #dtsave: save snapshots for all t=k*dtsave; res: spatial resolution of the dataset (number of grids along each dimension. It should match the tensor shape in the dataset when data are downsampled and saved); N: number of trajectories; T: the traj are run for [0,T]
       128:{'link':rel_path+"filename_frs.pt",'N':1,'dtsave':1/16,'T':400,'res':64},
       }}
data_dict['data']=a
b={
   100:{16:{'link':rel_path+"stat_dataset_cgs.pt",'N':100,'dtsave':1,'T':1500,'res':16,'name':'les','init':700},
       128:{'link':rel_path+"stat_dataset_frs.pt",'N':400,'dtsave':1,'T':1500,'res':64,'init':800,'name':'dns'},
       }}

data_dict['stat']=b

data_dict['stat_long']=data_dict['stat']

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
random_data={16:rel_path+'random_init_course.pt'}


'''Model list'''
rel_path='model_save/'
model1={0:rel_path+"model_kf_cgs.pt"} # model after stage 1: supervised loss with CGS data.
model2={0:rel_path+"model_kf_frs.pt"}# model after stage 2: supervised loss with FRS data.
model3={0: rel_path+'model_pde.pt'}# model after stage 3: pde loss.
model_dict={1:model1,2:model2,3:model3}
