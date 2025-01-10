from utilities import *
from utilities3 import *

import torch.fft as fft
from pdb import set_trace as st
from vision_transformer import vit_b_kf

"""
data_loader
batch_size=16 ('btz')  (variable: batch_size)
Samples of data_loader:
{'x': tensor(btz,16,16,2)
'y': tensor(btz,2,2,16,16)}

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w = torch.load('KF_64_0503.pt',map_location=device) #1,341,64,64

u = w_to_u_ntxy(w)


def filter(v):
    vh = fft.fft2(v, norm='forward')
    vh1 = torch.cat([vh[..., :8, :], vh[..., -8:, :]], dim=-2)
    vh2 = torch.cat([vh1[..., :8], vh1[..., -8:]], dim=-1)
    return torch.real(fft.ifft2(vh2, norm='forward'))

input_x=w_to_u_ntxy(filter(w)).squeeze(dim=0)
print('input',input_x.shape)

input_xTe=(input_x[::3])[110:].clone()
input_xTr=(input_x[::3])[:110].clone() #110,16,16,2 

tau=[[0,0],[0,0]]
for i in range(2):
    for j in range(2):
        tau[i][j]=filter(u[...,i]*u[...,j])-filter(u[...,i])*filter(u[...,j])
        tau[i][j]=tau[i][j].squeeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
input_y=torch.cat([torch.cat(tau[0],dim=2),torch.cat(tau[1],dim=2)],dim=1)#341,2,2,16,16
input_y=torch.cat([torch.cat(tau[0],dim=2),torch.cat(tau[1],dim=2)],dim=1)#341,2,2,16,16

input_yTe=(input_y[::3])[110:].clone()
input_yTr=(input_y[::3])[:110].clone()

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

'''model =  vit_b_322(num_classes=1024).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)

for index_, data in enumerate(data_loader):
    x = data['x'].permute(0, 3, 1, 2).cuda()
    y = data['y'].cuda()
    output = model(x)
    output = output.reshape(-1, 4, 16, 16)'''
    

# import neuralop
# import neuralop.wcw.tool_wcw as wcw
# wcw.check_dataloader(data_loader)



