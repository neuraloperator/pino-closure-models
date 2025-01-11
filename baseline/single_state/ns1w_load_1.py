from utilities import *
from utilities3 import *

import torch.fft as fft
# from pdb import set_trace as st
# from vision_transformer import vit_b_kf
import neuralop
import neuralop.wcw.tool_wcw as wcw

"""
data_loader
batch_size=16 ('btz')  (variable: batch_size)
Samples of data_loader:
{'x': tensor(btz,48,48,2)
'y': tensor(btz,2,2,48,48)}

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w = torch.load('ns1w_train_set.pt',map_location=device) 

u = w_to_u_ntxy(w)


def filter(v):
    '''v: dx>48(=128), out: dx=48'''
    vh = fft.rfft2(v, norm='forward')
    vh1 = torch.cat([vh[..., :24, :], vh[..., -24:, :]], dim=-2)
    vh2=vh1[...,:25]
    # vh2 = torch.cat([vh1[..., :24], vh1[..., -24:]], dim=-1)
    return torch.real(fft.irfft2(vh2, norm='forward'))

input_x=w_to_u_ntxy(filter(w)).squeeze(dim=0)# 385,48,48
wcw.sss(input_x)
print('input',input_x.shape)

input_xTe=input_x[368:].clone()
input_xTr=input_x[:368].clone() #368,48,48,2
wcw.sss(input_xTr)

tau=[[0,0],[0,0]]
for i in range(2):
    for j in range(2):
        tau[i][j]=filter(u[...,i]*u[...,j])-filter(u[...,i])*filter(u[...,j])
        tau[i][j]=tau[i][j].squeeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
input_y=torch.cat([torch.cat(tau[0],dim=2),torch.cat(tau[1],dim=2)],dim=1)#385,2,2,48,48


input_yTe=input_y[368:].clone()
input_yTr=input_y[:368].clone()
wcw.sss(input_yTr)#368,2,2,48,48

input_yTe = input_yTe.view(input_yTe.shape[0], -1, input_yTe.shape[-2], input_yTe.shape[-1])
input_yTr = input_yTr.view(input_yTr.shape[0], -1, input_yTr.shape[-2], input_yTr.shape[-1])


x_normalizer = UnitGaussianNormalizer(input_xTr)
input_xTe = x_normalizer.encode(input_xTe)
input_xTr = x_normalizer.encode(input_xTr)

y_normalizer = UnitGaussianNormalizer(input_yTr)
input_yTe = y_normalizer.encode(input_yTe)
input_yTr = y_normalizer.encode(input_yTr)

my_datasetTe=MyDataset(input_x=input_xTe,input_y=input_yTe)
my_datasetTr=MyDataset(input_x=input_xTr,input_y=input_yTr)
#data_devise=my_dataset[0]['x'].device
batch_size=16

train_loader = DataLoader(
    my_datasetTr,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    #pin_memory=True, #(data_devise=='cpu'),
    persistent_workers=False,
)

test_loader = DataLoader(
    my_datasetTe,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    #pin_memory=True, #(=(data_devise=='cpu'),
    persistent_workers=False,
)
