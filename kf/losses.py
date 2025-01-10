"""
losses.py contains code to compute standard data objective 
functions for training Neural Operators. 

By default, losses expect arguments y_pred (model predictions) and y (ground y.)
"""

import math


import torch
import torch.fft as fft
import torch.nn.functional as F
from utils import FC2D
from utils import FC3D



#Set fix{x,y,z}_bnd if function is non-periodic in {x,y,z} direction
#x: (*, s)
#y: (*, s)
def central_diff_1d(x, h, fix_x_bnd=False):
    dx = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h)

    if fix_x_bnd:
        dx[...,0] = (x[...,1] - x[...,0])/h
        dx[...,-1] = (x[...,-1] - x[...,-2])/h

    return dx

#x: (*, s1, s2)
#y: (*, s1, s2)
def central_diff_2d(x, h, fix_x_bnd=False, fix_y_bnd=False):
    if isinstance(h, float):
        h = [h, h]

    dx = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[1])

    if fix_x_bnd:
        dx[...,0,:] = (x[...,1,:] - x[...,0,:])/h[0]
        dx[...,-1,:] = (x[...,-1,:] - x[...,-2,:])/h[0]

    if fix_y_bnd:
        dy[...,:,0] = (x[...,:,1] - x[...,:,0])/h[1]
        dy[...,:,-1] = (x[...,:,-1] - x[...,:,-2])/h[1]

    return dx, dy

#x: (*, s1, s2, s3)
#y: (*, s1, s2, s3)
def central_diff_3d(x, h, fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False):
    if isinstance(h, float):
        h = [h, h, h]

    dx = (torch.roll(x, -1, dims=-3) - torch.roll(x, 1, dims=-3))/(2.0*h[0])
    dy = (torch.roll(x, -1, dims=-2) - torch.roll(x, 1, dims=-2))/(2.0*h[1])
    dz = (torch.roll(x, -1, dims=-1) - torch.roll(x, 1, dims=-1))/(2.0*h[2])

    if fix_x_bnd:
        dx[...,0,:,:] = (x[...,1,:,:] - x[...,0,:,:])/h[0]
        dx[...,-1,:,:] = (x[...,-1,:,:] - x[...,-2,:,:])/h[0]

    if fix_y_bnd:
        dy[...,:,0,:] = (x[...,:,1,:] - x[...,:,0,:])/h[1]
        dy[...,:,-1,:] = (x[...,:,-1,:] - x[...,:,-2,:])/h[1]

    if fix_z_bnd:
        dz[...,:,:,0] = (x[...,:,:,1] - x[...,:,:,0])/h[2]
        dz[...,:,:,-1] = (x[...,:,:,-1] - x[...,:,:,-2])/h[2]

    return dx, dy, dz


#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='sum',pino_t=False,func='rel'):
        super().__init__()

        self.d = d
        self.p = p
        self.pino_t=pino_t
        self.func=func
        self.out_func=getattr(self,self.func)

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d

        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y):

        diff = torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                          p=self.p, dim=-1, keepdim=False)
        ynorm = torch.norm(torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False)

        diff = diff/ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def __call__(self, y_pred, y, **kwargs):
        if not(self.pino_t):
            return self.out_func(y_pred, y)
        else:
            tt=kwargs['t_val']
            # return self.rel(y_pred[:,:,tt-1:tt],y[:,:,tt-1:tt])#(debug)/(y.shape[2]/tt)
            return self.out_func(y_pred[:,:,tt-1::tt],y[:,:,tt-1::tt])/(y.shape[2]/tt)


class H1Loss(object):
    def __init__(self, d=1, L=2*math.pi, reduce_dims=0, reductions='sum',
                 fix_x_bnd=False, fix_y_bnd=False, fix_z_bnd=False,group=False,pino_t=False,func='rel'):
        super().__init__()

        assert d > 0 and d < 4, "Currently only implemented for 1, 2, and 3-D."

        self.d = d
        self.fix_x_bnd = fix_x_bnd
        self.fix_y_bnd = fix_y_bnd
        self.fix_z_bnd = fix_z_bnd
        self.group=group
        self.pino_t=pino_t
        self.func=func
        self.out_func=getattr(self,self.func)

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L

    def compute_terms(self, x, y, h):
        dict_x = {}
        dict_y = {}

        if self.d == 1:
            dict_x[0] = x
            dict_y[0] = y

            x_x = central_diff_1d(x, h[0], fix_x_bnd=self.fix_x_bnd)
            y_x = central_diff_1d(y, h[0], fix_x_bnd=self.fix_x_bnd)

            dict_x[1] = x_x
            dict_y[1] = y_x

        elif self.d == 2:
            dict_x[0] = torch.flatten(x, start_dim=-2)
            dict_y[0] = torch.flatten(y, start_dim=-2)

            x_x, x_y = central_diff_2d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)
            y_x, y_y = central_diff_2d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-2)
            dict_x[2] = torch.flatten(x_y, start_dim=-2)

            dict_y[1] = torch.flatten(y_x, start_dim=-2)
            dict_y[2] = torch.flatten(y_y, start_dim=-2)

        else:
            dict_x[0] = torch.flatten(x, start_dim=-3)
            dict_y[0] = torch.flatten(y, start_dim=-3)

            x_x, x_y, x_z = central_diff_3d(x, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)
            y_x, y_y, y_z = central_diff_3d(y, h, fix_x_bnd=self.fix_x_bnd, fix_y_bnd=self.fix_y_bnd, fix_z_bnd=self.fix_z_bnd)

            dict_x[1] = torch.flatten(x_x, start_dim=-3)
            dict_x[2] = torch.flatten(x_y, start_dim=-3)
            dict_x[3] = torch.flatten(x_z, start_dim=-3)

            dict_y[1] = torch.flatten(y_x, start_dim=-3)
            dict_y[2] = torch.flatten(y_y, start_dim=-3)
            dict_y[3] = torch.flatten(y_z, start_dim=-3)

        return dict_x, dict_y

    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d

        dict_x, dict_y = self.compute_terms(x, y, h)

        const = math.prod(h)
        diff = const*torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2

        for j in range(1, self.d + 1):
            diff += const*torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2

        diff = diff**0.5

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def rel(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        if self.group:
            dict_x, dict_y = self.compute_terms(x, y, h)

            diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)
            ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)
            loss=diff/ynorm
            if self.reduce_dims is not None:
                loss = self.reduce_all(loss).squeeze()
            diff,ynorm=0,0
            for j in range(1, self.d + 1):
                diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False) ** 2
                ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False) ** 2

            diff = (diff ** 0.5) / (ynorm ** 0.5)

            if self.reduce_dims is not None:
                diff = self.reduce_all(diff).squeeze()
            loss+=diff
            return loss
        else:
            dict_x, dict_y = self.compute_terms(x, y, h)

            diff = torch.norm(dict_x[0] - dict_y[0], p=2, dim=-1, keepdim=False)**2
            ynorm = torch.norm(dict_y[0], p=2, dim=-1, keepdim=False)**2

            for j in range(1, self.d + 1):
                diff += torch.norm(dict_x[j] - dict_y[j], p=2, dim=-1, keepdim=False)**2
                ynorm += torch.norm(dict_y[j], p=2, dim=-1, keepdim=False)**2

            diff = (diff**0.5)/(ynorm**0.5)

            if self.reduce_dims is not None:
                diff = self.reduce_all(diff).squeeze()

            return diff


    def __call__(self, y_pred, y, h=None, **kwargs):
        if self.pino_t:
            tt=kwargs['t_val']
            return self.out_func(y_pred[:,:,tt-1::tt],y[:,:,tt-1::tt],h=h)/(y.shape[2]/tt)
        return self.out_func(y_pred, y, h=h)


class IregularLpqLoss(torch.nn.Module):
    def __init__(self, p=2.0, q=2.0):
        super().__init__()

        self.p = 2.0
        self.q = 2.0

    #x, y are (n, c) or (n,)
    #vol_elm is (n,)

    def norm(self, x, vol_elm):
        if len(x.shape) > 1:
            s = torch.sum(torch.abs(x)**self.q, dim=1, keepdim=False)**(self.p/self.q)
        else:
            s = torch.abs(x)**self.p

        return torch.sum(s*vol_elm)**(1.0/self.p)

    def abs(self, x, y, vol_elm):
        return self.norm(x - y, vol_elm)

    #y is assumed y
    def rel(self, x, y, vol_elm):
        return self.abs(x, y, vol_elm)/self.norm(y, vol_elm)

    def forward(self, y_pred, y, vol_elm, **kwargs):
        return self.rel(y_pred, y, vol_elm)


def pressure_drag(pressure, vol_elm, inward_surface_normal,
                  flow_direction_normal, flow_speed,
                  reference_area, mass_density=1.0):

    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = torch.sum(inward_surface_normal*flow_direction_normal, dim=1, keepdim=False)

    return const*torch.sum(pressure*direction*vol_elm)

def friction_drag(wall_shear_stress, vol_elm,
                  flow_direction_normal, flow_speed,
                  reference_area, mass_density=1.0):

    const = 2.0/(mass_density*(flow_speed**2)*reference_area)
    direction = torch.sum(wall_shear_stress*flow_direction_normal, dim=1, keepdim=False)

    x = torch.sum(direction*vol_elm)

    return const*torch.sum(direction*vol_elm)

def total_drag(pressure, wall_shear_stress, vol_elm,
               inward_surface_normal, flow_direction_normal,
               flow_speed, reference_area, mass_density=1.0):

    cp = pressure_drag(pressure, vol_elm, inward_surface_normal,
                       flow_direction_normal, flow_speed,
                       reference_area, mass_density)

    cf = friction_drag(wall_shear_stress, vol_elm,
                       flow_direction_normal, flow_speed,
                       reference_area, mass_density)

    return cp + cf


class WeightedL2DragLoss(object):

    def __init__(self, mappings: dict, device: str = 'cuda'):
        """WeightedL2DragPlusLPQLoss calculates the l2 drag loss
            over the shear stress and pressure outputs of a model.

        Parameters
        ----------
        mappings: dict[tuple(Slice)]
            indices of an input tensor corresponding to above fields
        device : str, optional
            device on which to do tensor calculations, by default 'cuda'
        """
        # take in a dictionary of drag functions to be calculated on model output over output fields
        super().__init__()
        self.mappings = mappings
        self.device = device


    def __call__(self, y_pred, y, vol_elm, inward_normals, flow_normals, flow_speed, reference_area, **kwargs):
        c_pred = None
        c_truth = None
        loss = 0.

        stress_indices = self.mappings['wall_shear_stress']
        pred_stress = y_pred[stress_indices].view(-1,1)
        truth_stress = y[stress_indices]

        # friction drag takes padded input
        pred_stress_pad = torch.zeros((pred_stress.shape[0], 3), device=self.device)
        pred_stress_pad[:,0] = pred_stress.view(-1,)

        truth_stress_pad = torch.zeros((truth_stress.shape[0], 3), device=self.device)
        truth_stress_pad[:,0] = truth_stress.view(-1,)

        pressure_indices = self.mappings['pressure']
        pred_pressure = y_pred[pressure_indices].view(-1,1)
        truth_pressure = y[pressure_indices]

        c_pred = total_drag(pressure=pred_pressure,
                            wall_shear_stress=pred_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )
        c_truth = total_drag(pressure=truth_pressure,
                            wall_shear_stress=truth_stress_pad,
                            vol_elm=vol_elm,
                            inward_surface_normal=inward_normals,
                            flow_direction_normal=flow_normals,
                            flow_speed=flow_speed,
                            reference_area=reference_area
                            )

        loss += torch.abs(c_pred - c_truth) / torch.abs(c_truth)

        loss = (1.0/len(self.mappings) + 1)*loss

        return loss


class KS_eqn_loss(object):
    def __init__(self,visc=0.01,domain=[0.1,6*torch.pi],method='t_fd_x_f',loss=F.mse_loss,
                 device='cuda',d=5,C=25):# domain:[t,x]
        super().__init__()
        self.visc = visc
        self.method = method
        self.loss = loss

        self.counter = 0
        self.domain_length = domain
        if not isinstance(self.domain_length, (tuple, list)):
            self.domain_length = [self.domain_length] * 2
        self.device=device
        self.k=(torch.arange(1026)*(1j*2*torch.pi/self.domain_length[1])).to(self.device) # include j
        self.coeff=self.k**2+self.visc*self.k**4
        if self.method in ['t_fc_x_f']:
            self.fc_helper = FC2D(device, d, C)
        self.out_func=getattr(self,self.method)

    def t_fd_x_f(self,u,u0):
        assert u.shape[1]==1
        nt,nx=u.shape[-2],u.shape[-1]
        dt=self.domain_length[0]/nt
        k_use=nx//2+1
        v=fft.rfft(u,dim=-1)
        Lv=v*(self.coeff[:k_use])
        nonl=0.5*torch.pow(u,2)
        Lv+=fft.rfft(nonl,dim=-1)*self.k[:k_use]

        partial_t = (torch.roll(u, -1, dims=-2) - torch.roll(u, 1, dims=-2)) / (2.0 *dt)

        partial_t[...,0,:] = (u[...,1,:] - u0[:,0:1,0,:])/(2.0*dt)
        partial_t[...,-1,:] = (u[...,-1,:] - u[...,-2,:])/dt

        Lv+=fft.rfft(partial_t,dim=-1)
        return torch.norm(Lv,p=2)

    def t_fc_x_f(self, u,u0):
        """
        Fourier differentiation for spatial dim, FC for temporal dim
        """
        assert u.shape[1] == 1
        nt, nx = u.shape[-2], u.shape[-1]
        dt = self.domain_length[0] / nt
        k_use = nx // 2 + 1
        v = fft.rfft(u, dim=-1)
        Lv = v * (self.coeff[:k_use])
        nonl = 0.5 * torch.pow(u, 2)
        Lv += fft.rfft(nonl, dim=-1) * self.k[:k_use]


        # compute u_t using Fourier continuation
        u=torch.cat([u0[:,0,0:1],u.squeeze(dim=1)],dim=1)
        u_t = self.fc_helper.diff_y(u, self.domain_length[0])


        Lv=torch.squeeze(fft.irfft(Lv,dim=-1),dim=1)

        return torch.norm(Lv+u_t[:,1:],p=2)


    def __call__(self, y_pred, x,**kwargs):
        return self.out_func(y_pred,x)

class KF_eqn_loss(object):
    def __init__(self,re,s=64,domain=[1,2*math.pi,2*math.pi],method='t_fc_x_f',loss=F.mse_loss,
                 device='cuda',d=5,C=25):# domain:[t,x]
        super().__init__()
        self.s1=s
        self.s2=s
        self.h=1/s
        self.L1=domain[1]
        self.L2=domain[2]
        self.visc = 1/re
        self.method = method
        self.loss = loss

        self.counter = 0
        self.domain_length = domain
        if not isinstance(self.domain_length, (tuple, list)):
            print('pde_loss, domain is not assigned!!')
            exit()
            # self.domain_length = [self.domain_length] * 2
        self.device=device
        if self.method in ['t_fc_x_f']:
            self.fc_helper = FC3D(device, d, C)
        self.out_func=getattr(self,self.method)
        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=self.s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-self.s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1,self.s2//2 + 1).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=self.s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(self.s1, 1).to(device)

        self.d1=1j*2*math.pi/self.L1*self.k1
        self.d2=1j*2*math.pi/self.L2*self.k2
        self.d=[self.d1,self.d2]
        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=self.s1//2, step=1),\
                                torch.arange(start=-self.s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, self.s2//2 + 1).to(device)

        freq_list2 = torch.arange(start=0, end=self.s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(self.s1, 1).to(device)

        self.G = ((4*math.pi**2)/(self.L1**2))*k1**2 + ((4*math.pi**2)/(self.L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        t = torch.linspace(0, self.L1, s + 1, device=device)[0:-1]
        _, Y = torch.meshgrid(t, t, indexing='ij')

        f = -4 * torch.cos(4.0 * Y)

        self.f_h=fft.rfft2(f)

    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h
    def nonlinear_term(self, w,real_space=True):
        #Physical space vorticity
        w_h = fft.rfft2(w, s=(self.s1, self.s2))

        #Physical space velocity

        q,v= self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)

        nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                    2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))

        nonlin-=w_h*(self.visc*self.G)


        #Add forcing function
        if self.f_h is not None:
            nonlin += self.f_h
        if real_space:
            return fft.irfft2(nonlin,s=(self.s1,self.s2))
        else:
            return nonlin

    def t_fc_x_f(self, u,u0):
        """
        Fourier differentiation for spatial dim, FC for temporal dim
        """
        assert u.shape[1] == 1
        nt=u.shape[-3]
        # nt, nx = u.shape[-2], u.shape[-1]
        dt = self.domain_length[0] / nt
        dx=self.L1/u.shape[-2]
        dy=self.L2/u.shape[-1]

        nonlin=self.nonlinear_term(u,real_space=True)


        u=torch.cat([u0[:,0,0:1],u.squeeze(dim=1)],dim=1)

        u_t = self.fc_helper.diff_t(u, self.domain_length[0])

        return torch.norm(nonlin-u_t[:,1:],p=2)*dt*dx*dy


    def __call__(self, y_pred, x,**kwargs):
        return self.out_func(y_pred,x)


class mix_loss(object):
    def __init__(self,*args):
        self.funcs=args

    def __call__(self, *args,lam=0,**kwargs):
        ans=self.funcs[0](*args,**kwargs)
        ans+=lam*self.funcs[1](*args,**kwargs)


        return ans


def null_loss(*args,**kwargs):
    return 0


