import torch
import torch.fft as fft

import math




#Solve: w_t = - u . grad(w) + (1/Re)*Lap(w) + f
#       u = (psi_y, -psi_x)
#       -Lap(psi) = w
#Note: Adaptive time-step takes smallest step across the batch
class NavierStokes2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64,alpha=4):

        self.s1 = s1
        self.s2 = s2

        self.L1 = L1
        self.L2 = L2

        self.h = 1.0/max(s1, s2)
        self.x_grid=self.L1/self.s1

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
        self.d_tensor=torch.cat([i.unsqueeze(-1)for i in self.d],dim=-1).unsqueeze(-2)# 128,65,1,2
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

        #Lilly_Smagorinsky model
        self.alpha=alpha # 2^k, times of further filtring
        self.s1f=self.s1//self.alpha
        self.s2f=self.s2//self.alpha

        freq_list1 = torch.cat((torch.arange(start=0, end=self.s1f // 2, step=1),torch.zeros((1,)),
                                torch.arange(start=-self.s1f // 2 + 1, end=0, step=1)), 0)
        self.k1f = freq_list1.view(-1, 1).repeat(1, self.s2f // 2 + 1).type(dtype).to(device)

        freq_list2 = torch.cat((torch.arange(start=0, end=self.s2f // 2, step=1), torch.zeros((1,))), 0)
        self.k2f = freq_list2.view(1, -1).repeat(self.s1f, 1).type(dtype).to(device)

        self.d1f = 1j * 2 * math.pi / self.L1 * self.k1f
        self.d2f = 1j * 2 * math.pi / self.L2 * self.k2f
        self.df = [self.d1f, self.d2f]
        self.d_tensorf = torch.cat([i.unsqueeze(-1) for i in self.df], dim=-1).unsqueeze(-2)  # 16,9,1,2


    #Compute stream function from vorticity (Fourier space)
    def filter(self,v,dim:list=[-2,-1],input_real=True,out_real=True):
        '''v: ...,X,X (res=self.s1)  out:  (res=self.sf!!!! (not low-res))'''
        for i in range(len(dim)):
            if dim[i]<0:
                dim[i]=dim[i]+v.ndim

        vh = fft.rfft2(v, norm='forward',dim=dim)if input_real else v
        modef=torch.zeros_like(vh)
        vh=fft.fftshift(vh,dim=dim[0])
        slice_1=[slice(None)]*(max(dim[0],dim[1])+1)
        slice_1[dim[0]]=slice(self.s1//2-self.s1f//2+1,self.s1//2+self.s1f//2)
        slice_1[dim[1]]=slice(None,self.s2f//2+1)
        modef[slice_1]=vh[slice_1]
        modef=fft.ifftshift(modef,dim=dim[0])
        if out_real:
            modef=fft.irfft2(modef,dim=dim,norm='forward')
        return modef


    def stream_function(self, w_h, real_space=False):
        #-Lap(psi) = w
        psi_h = self.inv_lap*w_h

        if real_space:
            return fft.irfft2(psi_h, s=(self.s1, self.s2))
        else:
            return psi_h

    #Compute velocity field from stream function (Fourier space)
    def velocity_field(self, stream_f, real_space=True):# shape = w or w_h
        #Velocity field in x-direction = psi_y
        q_h = (2*math.pi/self.L2)*1j*self.k2*stream_f

        #Velocity field in y-direction = -psi_x
        v_h = -(2*math.pi/self.L1)*1j*self.k1*stream_f

        if real_space:
            return fft.irfft2(q_h, s=(self.s1, self.s2)), fft.irfft2(v_h, s=(self.s1, self.s2))
        else:
            return q_h, v_h

    #Compute non-linear term + forcing from given vorticity (Fourier space)
    def nonlinear_term(self, w_h, f_h=None,clos=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))


        #Physical space velocity
        if clos==None:
            q,v= self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                        2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))

        elif clos: # clos: C_s coefficient in sgs model
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                    2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))
            stress=self.s_tensor(w,real=True)#b,x,y,2,2
            stress_norm=torch.norm(stress,dim=[-1,-2]).unsqueeze(-1).unsqueeze(-1)#b,x,y,1,1



            #Lij

            uf=torch.cat([self.filter(q).unsqueeze(-1),self.filter(v).unsqueeze(-1)],dim=-1)#b,x,y,2
            second_term=uf.unsqueeze(-1)*uf.unsqueeze(-2)# b,x,y,2,2
            uf=torch.cat([q.unsqueeze(-1),v.unsqueeze(-1)],dim=-1)#b,x,y,2,2
            first_term=self.filter(uf.unsqueeze(-1)*uf.unsqueeze(-2),dim=[-4,-3])
            l_matrix=first_term-second_term#b,x,y,2,2



            #Mij
            second_term=self.filter(stress_norm*stress,dim=[-4,-3])
            first_term=self.filter(stress,dim=[-4,-3])*self.filter(stress_norm,dim=[-4,-3])
            M_mat=-2*(self.x_grid**2)*((self.alpha**2+1)*first_term-second_term) #b,x',y',2,2
            if clos==1:
                cs2=torch.mean(l_matrix*M_mat,dim=[1,2,3,4])/torch.mean(M_mat**2,dim=[1,2,3,4]).view(-1,1,1,1,1)#scalar cs: b
            else:

                cs2=(torch.mean(l_matrix*M_mat,dim=[-2,-1])/torch.mean(M_mat**2,dim=[-2,-1])).unsqueeze(-1).unsqueeze(-1)#b,x,y
            cs2=torch.clamp(cs2,min=0)



            # final return

            stress=stress_norm*stress*cs2
            stress_h=(self.d_tensor@fft.rfft2(stress,dim=[-4,-3])).squeeze(-2)# b,x,y,2


            w_h_new = -self.d2 * stress_h[..., 0] + self.d1 * stress_h[..., 1]  # nabla\perp closure term(for u)


            nonlin+=2*(self.x_grid**2)*w_h_new



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



        return s/2

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

        #Rescale Laplacian by Reynolds number
        GG = (1.0/Re)*self.G

        #Move to Fourier space
        w_h = fft.rfft2(w)

        if f is not None:
            f_h = fft.rfft2(f)
        else:
            f_h = None
        
        if adaptive:
            q, v = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            delta_t = self.time_step(q, v, f, Re,delta_t0)

        time  = 0.0
        #Advance solution in Fourier space

        while time < T:

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

        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)


