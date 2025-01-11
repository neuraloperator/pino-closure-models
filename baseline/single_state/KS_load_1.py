from utilities3 import *
from utilities import *

import torch.fft as fft
from pdb import set_trace as st
# from vision_transformer import vit_b_ks
"""
data_loader
batch_size=16 ('btz')  (variable: batch_size)
Samples of data_loader:
{'x': tensor(btz,128)
'y': tensor(btz,128)}

"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.load('KS_train_set.pt',map_location=device) #4,351,1024

b=a**2
c=fft.irfft(fft.rfft(b,norm='forward')[:65], norm='forward')  # shape: 4*351*128
f_a=fft.irfft(fft.rfft(a,norm='forward')[:65], norm='forward')  # 4,351,128
d=c-f_a**2  #4,351,128
#print('d:',d.shape)

input_xTe=a[3:,::10,::8].clone().reshape(-1,128) #105,128
input_xTr=a[:3,::10,::8].clone().reshape(-1,128) #105,128

input_yTe=d[3:,::10,::8].clone().reshape(-1,128) #105,128
input_yTr=d[:3,::10,::8].clone().reshape(-1,128) #105,128

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
    #pin_memory=True, #(=(data_devise=='cpu'),
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


'''model =  vit_b_ks(num_classes=128).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)

for index_, data in enumerate(data_loader):
    x = data['x'].unsqueeze(1).unsqueeze(1).float().cuda()
    y = data['y'].cuda()
    output = model(x)'''





