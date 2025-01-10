import torch
import torch.fft as fft

import math


#Setup for indexing in the 'ij' format

#Solve: -Lap(u) = f
class Poisson2d(object):

    def __init__(self, s1, s2, L1=2*math.pi, L2=2*math.pi, device=None, dtype=torch.float64):

        self.s1 = s1
        self.s2 = s2

        #Inverse negative Laplacian
        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)
        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.inv_lap = ((4*math.pi**2)/(L1**2))*k1**2 + ((4*math.pi**2)/(L2**2))*k2**2
        self.inv_lap[0,0] = 1.0
        self.inv_lap = 1.0/self.inv_lap
    
    def solve(self, f):
        return fft.irfft2(fft.rfft2(f)*self.inv_lap, s=(self.s1, self.s2))
    
    def __call__(self, f):
        return self.solve(f)

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
    def nonlinear_term(self, w_h, f_h=None,clos=None):
        #Physical space vorticity
        w = fft.irfft2(w_h, s=(self.s1, self.s2))


        #Physical space velocity
        if clos==None:
            q,v= self.velocity_field(self.stream_function(w_h, real_space=False), real_space=True)
            nonlin = -1j * ((2 * math.pi / self.L1) * self.k1 * fft.rfft2(q * w) + (
                        2 * math.pi / self.L2) * self.k2 * fft.rfft2(v * w))

        elif clos is not None:
            uh=self.velocity_field(self.stream_function(w_h, real_space=False), real_space=False)
            udh=[uh[0],uh[1]]
            jcb_h=[[0,0],[0,0]]
            s=[[0,0],[0,0]]
            clos_term_h=[[0,0],[0,0]]

            for i in range(2):
                for j in range(2):
                    t=self.d[i]*uh[j]
                    udh.append(t)
                    jcb_h[i][j]=t
            for i in range(2):
                for j in range(2):
                    s[i][j]=fft.irfft2(jcb_h[i][j]+jcb_h[j][i],s=(self.s1,self.s2))
            u_d=[(fft.irfft2(i,s=(self.s1,self.s2)))for i in udh]
            ud=[torch.unsqueeze(i,dim=-1)for i in u_d]

            clos_in=torch.cat(ud,dim=-1).to(torch.float32) #clos_in:btz*X*X*in_channel

            clos_in=torch.squeeze(clos(clos_in).to(torch.float64),dim=-1)#clos should >0  btz*X*X*1

            for i in range(2):
                for j in range(2):
                    clos_term_h[i][j]=fft.rfft2(clos_in*s[i][j],s=(self.s1,self.s2))
            u_clos_h=[0,0]
            nonlin=0
            for i in range(2):
                for j in range(2):
                    u_clos_h[i]+=clos_term_h[i][j]*self.d[j]
                nonlin+=-self.d[i]*fft.rfft2(u_d[i]*w)
            nonlin+=self.d[0]*u_clos_h[1]-self.d[1]*u_clos_h[0]



        #Compute non-linear term in Fourier space


        #Add forcing function
        if f_h is not None:
            nonlin += f_h

        return nonlin
    def tau(self,w,res_low,real=True):
        '''out: 2*2 list tau matrix'''

        res_hi=self.s1
        w_h=fft.rfft2(w,s=(res_hi,res_hi))
        # tauu_i,j=\bar(ui uj)-bar(ui)bar(uj)
        uh= self.velocity_field(self.stream_function(w_h, real_space=False), real_space=False)
        tauu=[[0,0],[0,0]]
        u_filter=[fft.irfft2(uh[i],s=(res_low,res_low)) for i in range(2)]

        for i in range(2):
            for j in range(i,2):
                uij_ext=(fft.irfft2(uh[i],s=(res_hi,res_hi)))*fft.irfft2(uh[j],s=(res_hi,res_hi))

                tauu[i][j]=-u_filter[i]*u_filter[j]+fft.irfft2(fft.rfft2(uij_ext,s=(res_hi,res_hi)),s=(res_low,res_low))
        for i in range(2):
            for j in range(0,i):
                tauu[i][j]=tauu[j][i]
        if real:
            return tauu
        else:
            for i in range(2):
                for j in range(2):
                    tauu[i][j]=fft.rfft2(tauu[i][j],s=(res_low,res_low))
            return tauu
    def s_tensor(self,w,res_low,real=True):
        "return s_ij tensor and the input for niu (concate ui, D_j u_i (6 in total) together)"
        res_hi=self.s1
        ##
        w_h = fft.rfft2(w, s=(res_hi, res_hi))
        uh = self.velocity_field(self.stream_function(w_h, real_space=False), real_space=False)
        udh = [uh[0], uh[1]]
        jcb_h = [[0, 0], [0, 0]]
        s = [[0, 0], [0, 0]]
        clos_term_h = [[0, 0], [0, 0]]


        for i in range(2):
            for j in range(2):
                t = self.d[i] * uh[j]
                udh.append(t)
                jcb_h[i][j] = t
        for i in range(2):
            for j in range(2):

                s[i][j] = fft.irfft2(jcb_h[i][j] + jcb_h[j][i], s=(res_low, res_low))
        u_d = [(fft.irfft2(i, s=(res_low, res_low))) for i in udh]
        ud = [torch.unsqueeze(i, dim=-1) for i in u_d]
        # ud=[torch.unsqueeze((fft.irfft2(i,s=(self.s1,self.s2))),dim=-1)for i in udh]
        clos_in = torch.cat(ud, dim=-1)  # clos_in:btz*X*X*in_channel
        return s,clos_in


    def time_step(self, q, v, f, Re):
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
        return min(0.5*self.h/max_speed, 0.5*(self.h**2)/mu)

    def advance(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3,clos=None):

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
            delta_t = self.time_step(q, v, f, Re)

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
                delta_t = self.time_step(q, v, f, Re)
        
        return fft.irfft2(w_h, s=(self.s1, self.s2))
    
    def __call__(self, w, f=None, T=1.0, Re=100, adaptive=True, delta_t=1e-3):
        return self.advance(w, f, T, Re, adaptive, delta_t)

