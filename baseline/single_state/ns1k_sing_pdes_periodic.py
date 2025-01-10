import torch
import torch.fft as fft

import math
import neuralop
import neuralop.wcw.tool_wcw as wcw

from utilities3 import *
from utilities import *

'''compute mean, var of dataset for normalizer'''
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
w = torch.load('1_385_128_128_dns256.pt',map_location=device) #1,341,64,64
u = w_to_u_ntxy(w)
def filter(v):
    '''v: dx>48(=128), out: dx=48'''
    vh = fft.rfft2(v, norm='forward')
    vh1 = torch.cat([vh[..., :24, :], vh[..., -24:, :]], dim=-2)
    vh2=vh1[...,:25]
    # vh2 = torch.cat([vh1[..., :24], vh1[..., -24:]], dim=-1)
    return torch.real(fft.irfft2(vh2, norm='forward'))

input_x=w_to_u_ntxy(filter(w)).squeeze(dim=0)# nt,x',y',2
print('input',input_x.shape)
input_xTr=input_x[:368].clone() #368,48,48,2
x_normalizer = UnitGaussianNormalizer(input_xTr)
tau=[[0,0],[0,0]]
for i in range(2):
    for j in range(2):
        tau[i][j]=filter(u[...,i]*u[...,j])-filter(u[...,i])*filter(u[...,j])
        tau[i][j]=tau[i][j].squeeze(dim=0).unsqueeze(dim=1).unsqueeze(dim=2)
input_y=torch.cat([torch.cat(tau[0],dim=2),torch.cat(tau[1],dim=2)],dim=1)#341,2,2,48,48
input_yTr=input_y[:368]
y_normalizer = UnitGaussianNormalizer(input_yTr)
del w,u,input_yTr,input_xTr

from vision_transformer_ns1k import vit_b_ns1k
model =  vit_b_ns1k(num_classes=9216).to(device)

model.load_state_dict(torch.load('kf_1000_final.pt'), strict=True)
model.eval()


#Solve: w_t = - u . grad(w) + (1/Re)*Lap(w) + f
#       u = (psi_y, -psi_x)
#       -Lap(psi) = w
#Note: Adaptive time-step takes smallest step across the batch
class NavierStokes2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.h = 1.0/max(s1, s2)

        #Wavenumbers for first derivatives
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.zeros((1,)),\
                                torch.arange(start=-s1//2 + 1, end=0, step=1)), 0)
        self.k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)


        freq_list2 = torch.cat((torch.arange(start=0, end=s2//2, step=1), torch.zeros((1,))), 0)
        self.k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.d1=1j*2*math.pi/self.L1*self.k1
        self.d2=1j*2*math.pi/self.L2*self.k2
        self.d=[self.d1,self.d2]
        self.d_tensor=torch.cat([i.unsqueeze(-1)for i in self.d],dim=-1).unsqueeze(-2)# 16,9,1,2
        #Negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.G = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2

        #Inverse of negative Laplacian
        self.inv_lap = self.G.clone()
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap

        #Dealiasing mask using 2/3 rule
        self.dealias = (self.k1**2 + self.k2**2 <= 0.6*(0.25*s1**2 + 0.25*s2**2)).type(dtype).to(device)
        #Ensure mean zero
        self.dealias[0,0] = 0.0

    #Compute stream function from vorticity (Fourier space)
    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def closure_tau(self,w):#w:nxy
        u=w_to_u(w)
        u=x_normalizer.encode(u)
        u=u.permute(0,3,1,2)
        with torch.no_grad():
            output=model(u)
            # wcw.sss(output)# (1,1024)   =2x2x16x16
            output=output.reshape(-1,2,2,48,48)#gai
            return y_normalizer.decode(output)# n,2,2,16,16

    def nonlinear_term(self, w_h, f_h=None,clos=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))
        self.closure_tau(w)

        # wcw.sss(w)

        #Physical space velocity
        if clos==None:
            q,v= self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                        2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))

        elif clos: # clos: C_s coefficient in sgs model
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                    2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))

            tau=self.closure_tau(w)#n,2,2,s,s
            # wcw.sss(tau)
            tau_h=fft.rfft2(tau).permute(0,3,4,1,2)#n,16,9,2,2
            # wcw.sss(tau_h)
            tau_h=(self.d_tensor@tau_h).squeeze(-2) #n,s,s,2
            # wcw.sss(tau_h)
            w_h_new=-clos*(self.d2*tau_h[...,0]+self.d1*tau_h[...,1])
            nonlin+=w_h_new


            # stress=self.s_tensor(w,real=True)#b,x,y,2,2
            # stress_norm=torch.norm(stress,dim=[-1,-2])#b,x,y
            # # wcw.sss(stress_norm)
            # stress=stress_norm.unsqueeze(dim=-1).unsqueeze(-1)*stress#bxy22
            # stress_h=(self.d_tensor@fft.rfft2(stress,dim=[-4,-3])).squeeze(-2)# b,x,y,2
            # # wcw.sss(stress_h)
            # w_h_new=-self.d2*stress_h[...,0]+self.d1*stress_h[...,1]
            # # wcw.sss(w_h_new)
            # nonlin-=(((clos*self.h)**2).real)*w_h_new



        if f_h is not None:
            nonlin += f_h

        return nonlin

    def s_tensor(self,w,real=False):
        "return s_ij tensor "

        w_h = fft.rfft2(w)
        udh = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=False)
        uh=torch.cat([item.unsqueeze(dim=-1)for item in udh],dim=-1) #btz,x,y,2

        half_s=uh.unsqueeze(-1)*self.d_tensor #bz,x,y,2,2
        s=half_s+half_s.transpose(dim0=-1,dim1=-2)
        if real:
            s=fft.irfft2(s,dim=[-4,-3])
            # wcw.sss(s)


        return s

    def time_step(self, q, v, f, Re,dtmin=1.0):
        #Maxixum speed
        max_speed = torch.max(torch.sqrt(q**2 + v**2)).item()

        #Maximum force amplitude
        if f is not None:
            xi = torch.sqrt(torch.max(torch.abs(f))).item()
        else:
            xi = 1.0
        
        #Viscosity
        mu = (1.0/Re)*xi*((self.L1/(2*math.pi))**(3.0/4.0))*(((self.L2/(2*math.pi))**(3.0/4.0)))

        if max_speed == 0:
            return 0.5*(self.h**2)/mu
        
        #Time step based on CFL condition
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu,dtmin)

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3,clos=None,delta_t0=1e-2):
        # wcw.ppp(w[0])
        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)
        # wcw.sss(w)
        # wcw.sss(w_h)
        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re,delta_t0)

        time  = 0.0
        #Advance solution in Fourier space
        sss=0
        while time < T:
            sss+=1
            if time + delta_t > T:
                current_delta_t = T - time
            else:
                current_delta_t = delta_t

            #Inner-step of Heun's method
            nonlin1 = self.nonlinear_term(w_h, f_h,clos=clos)

            w_h_tilde = (w_h + current_delta_t*(nonlin1 - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #Cranck-Nicholson + Heun update
            nonlin2 = self.nonlinear_term(w_h_tilde, f_h,clos=clos)
            w_h = (w_h + current_delta_t*(0.5*(nonlin1 + nonlin2) - 0.5*GG*w_h))/(1.0 + 0.5*current_delta_t*GG)

            #De-alias
            w_h *= self.dealias

            #Update time
            time += current_delta_t

            #New time step
            if adaptive:
                q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
                delta_t = self.time_step(q, v, f, Re,delta_t0)
        # wcw.ppp(sss)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)
#
# a=torch.rand(2,3,132,64)
# print(a.shape)
# b=fft.rfft2(a)
# print(b.shape)

