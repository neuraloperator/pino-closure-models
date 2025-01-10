import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import operator
from functools import reduce

from torch.utils.data import Dataset, DataLoader

#################################################
#
# Utilities:
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def w_to_u_ntxy(w): #ntxy
    batchsize = w.size(0)
    nt=w.size(1)
    nx = w.size(2)
    ny = w.size(3)

    device = w.device
    w = w.reshape(batchsize,nt, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[2, 3])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1,1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
       1, 1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0,0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :,:, :k_max + 1], dim=[2, 3])
    uy = torch.fft.irfft2(uy_h[:, :,:, :k_max + 1], dim=[2, 3])
    u = torch.cat([ux, uy], dim=-1)
    return u #ntxy2
def w_to_u(w): #nxy
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)

    device = w.device
    w = w.reshape(batchsize, nx, ny, -1)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx // 2
    N = nx
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1,
                                                                                                        N).reshape(
        1, N, N, 1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N,
                                                                                                        1).reshape(
        1, N, N, 1)
    # Negative Laplacian in Fourier space
    lap = (k_x ** 2 + k_y ** 2)
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    u = torch.cat([ux, uy], dim=-1)
    return u#nxy2
class MyDataset(Dataset):
    def __init__(self, input_x,input_y):
        assert (input_x.size(0) == input_y.size(0)), "Size mismatch between tensors"
        self.x = input_x
        self.y = input_y
    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return {'x': x, 'y':y}
    def __len__(self):
        return self.x.size(0)

def slicing_split(x,dim,dt,dT=None,t0:int=0,t_end:int=None,single=0):
    '''
    This function input a, a.shape=(N,T,X1,X2,...)(the number of X is not fixed),
    I want to create a tesnor b such that b.shape=(N,K,t,X1,..),
    and b[n,k]=a[n,t0+k*dT:t0+(k)*dT+ dt].  (or from dT-dt to dT-1)
    Parameters
    ----------
    x: input tensor
    dim: the dimension of transformation . Only int (one dim) is implemented
    dt: could be negative
    dT: by default only slice at t0
    t0:
    single:   if single, when dt is int: return single t=dt (but preserve this dim with size 1).
    Returns
    -------

    '''
    shapex=list(x.shape)
    slice_t0=[slice(None)]*len(shapex)
    slice_dt=slice_t0.copy()
    slice_t0[dim]=slice(t0,t_end)
    y=x[tuple(slice_t0)]

    if dT==None or dT==0:
        dT=y.shape[dim]
    K=y.shape[dim]//dT
    yy=torch.split(y,dim=dim,split_size_or_sections=dT)
    if len(yy)>1:
        if yy[-1].shape[dim]<dT:
            yy=yy[:-1]
    if type(dt)==int:
        if dt>=0:
            assert dt<=dT
            if single:
                slice_dt[dim]=slice(dt,dt+1)
            else:
                slice_dt[dim]=slice(None,dt)
        else:
            assert dt+dT>=0
            if single:
                slice_dt[dim]=slice(dt,dt+1)
            else:
                slice_dt[dim]=slice(dt,None)
    else:
        assert dt[1]<dT
        slice_dt[dim]=slice(dt[0],dt[1])
    yy=list(map(lambda z:z[tuple(slice_dt)],yy))
    y=torch.stack(yy,dim=dim)
    return y


