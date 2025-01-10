import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from ks_ParaCtrl_sing import *
import math as mt
import torch.fft as fft
import neuralop.wcw.tool_wcw as wcw
import wandb
from dict_ref import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


gridx=torch.arange(start=1,end=N+1,step=1,dtype=torch.float64).to(device)/N*space_scaling
# gridx=torch.arange(start=1,end=N+1,step=1).to(device)/N*space_scaling
torch.set_printoptions(precision=8)

k=torch.cat((torch.arange(start=0,end=N//2+1),torch.arange(-N//2+1,0))).to(device)/half_period   #** changed from original
k_proj=torch.cat((torch.ones(N_proj//2+1),torch.zeros(N-N_proj),torch.ones(N_proj//2-1))).to(device)

class initial_value_lib:
    '''
    TO use it, first define an object (A) of this class.
    A=initial_value_lib()
    Whenever I want an i.c. function,
    A(x)
    '''

    def __init__(self,choose=choose_ic,alpha=2.0, tau=3.0,sigma=None):
        self.null_mode=max(round(half_period/mt.sqrt(niu)),1)#L_k~0 (L is the diag linear operator operated after fft)
        self.worst_mode=max(round(half_period/mt.sqrt(2*niu)),1)#L_k is the largest
        self.choose=choose
        self.s1=N
        self.device=device
        # if choose==5:
        #     self.random_coef=torch.randn(5)
        if sigma==None:
            self.sigma = 0.01*4*tau ** (0.5 * (2 * alpha - 1.0))
        else:
            self.sigma=sigma

        const1 = 1/(half_period**2)

        #
        # freq_list = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
        #                         torch.arange(start=-s1//2, end=0, step=1)), 0)
        # k2 = freq_list.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig =self.sigma*((const1*k**2 + tau**2)**(-alpha/2.0))[:self.s1//2+1]
        self.sqrt_eig[0]=0
    def __call__(self, *args):
        return self.forward(*args)

    def update_par(self):
        self.null_mode=max(round(half_period/mt.sqrt(niu)),1)#L_k~0 (L is the diag linear operator operated after fft)
        self.worst_mode=max(round(half_period/mt.sqrt(2*niu)),1)#L_k is the largest
        self.choose=choose_ic
    # def set_niu(cls, new_niu,new_half_per):
    #     '''when changing niu, add initial_value_lib.set_niu(new_niu)'''
    #     cls.niu = new_niu
    #     cls.half_period = new_half_per
    #     cls.update_null_mode()
    #     cls.update_worst_mode()

    # @staticmethod
    # def update_null_mode():
    #     initial_value_lib.null_mode = max(round(initial_value_lib.half_period/mt.sqrt(initial_value_lib.niu)), 1)
    # @staticmethod
    # def update_worst_mode():
    #     initial_value_lib.worst_mode = max(round(initial_value_lib.half_period/mt.sqrt(2*initial_value_lib.niu)), 1)
    # @staticmethod
    def period_original(self,x):
        # print(f"forward,{x.type}")
        y = 0.1*torch.cos(x / half_period) * (1 + torch.sin(x / half_period))
        return y

    # @staticmethod
    def nonperiod_original(self,x):
        y = torch.cos(x / half_period/space_scaling) * (1 + torch.sin(x / half_period/space_scaling))
        return y

    # @staticmethod
    def single_steady(self,x):
        return torch.sin(self.null_mode*x/half_period)

    # @staticmethod
    def single_worst(self,x):
        return torch.sin(self.worst_mode * x / half_period)
    def nonL_period(self,x):
        y=self.period_original(x)
        return torch.tanh(y)

    def random_gen(self,x):
        y = torch.cos(x) * (1 + torch.sin(x))

        #random_gen
        xi = torch.randn(Nsum,self.s1 // 2 + 1, 2, device=self.device)
        xi[..., 0] = self.sqrt_eig * xi[..., 0]
        xi[..., 1] = self.sqrt_eig * xi[..., 1]
        # print('xi',xi.shape)
        # print("xi_cplx",torch.view_as_complex(xi).shape)
        # y= fft.irfft(torch.view_as_complex(xi),norm=norm)
        y= y+fft.irfft(torch.view_as_complex(xi),norm=norm)
        return y
    functions = {
        0:period_original,1: nonperiod_original,
        2:single_steady,3:single_worst,
        4:nonL_period,5:random_gen
    }
    def forward(self, x):
        # print(f"forward,{x.type}")
        return self.functions[self.choose](self,x)


initial_value=initial_value_lib()

u0=initial_value(gridx)  #be updated in iterations


ut=(u0.clone()).unsqueeze(-2) #sp #track and save the (u_t) traj

# v=torch.fft.fft(u0,norm='forward')
v=torch.fft.fft(u0,norm=norm)
vt=(v.clone()).unsqueeze(-2)#sp


L=k**2-niu*(k**4)
E_2=torch.exp(timegrid*L/2)
E=E_2**2

class coeff_scheme():
    ''' coefficient for num-scheme (etdrk4)'''
    def __init__(self,choose=choose_coef,eta=eta_stb_change,MM=M):
        self.choose=choose
        self.eta=eta
        self.MM=M
    def __call__(self, *args, **kwargs):
        return self.forward(*args)

    def ff(self,x):

        r=torch.exp(1j*torch.pi*(torch.arange(1,M+1,dtype=torch.float64).to(device)-0.5)/M)
        lr=timegrid*(x.unsqueeze(1)+r)  #N*m  -(57)
        '''usign this for up to (57)
        WRONG!   SHould be timegrid*x.unsqueeze(1)+r
        But why it plot the same?  (That's fine, since the contour integral remains the same regardless of the radius, as long as it is correct)
        
        '''
        # e_lr=torch.exp(lr/2)
        # e_lr2=e_lr**2
        # q=timegrid*torch.real(torch.mean(dim=1,input=(e_lr-1)/lr))
        # f1=timegrid*torch.real(torch.mean(dim=1,input=((-4-lr+e_lr2*(4-3*lr+lr**2)))/(lr**3)))
        # f2=timegrid*torch.real(torch.mean(dim=1,input=((2+lr+e_lr2*(lr-2)))/(lr**3)))
        # f3=timegrid*torch.real(torch.mean(dim=1,input=((-4-3*lr-lr**2+e_lr2*(4-lr)))/(lr**3)))
        '''Above: the solver is correct (but solver for other coeef is wrong: they are using f(L) instead of hf(hL))
        But the function plot (qwer) is incorrect.'''

        lr=x.unsqueeze(1)+r
        e_lr=torch.exp(lr/2)
        e_lr2=e_lr**2
        q = torch.real(torch.mean(dim=1, input=(e_lr - 1) / lr))
        f1 =torch.real(torch.mean(dim=1, input=((-4 - lr + e_lr2 * (4 - 3 * lr + lr ** 2))) / (lr ** 3)))
        f2 =torch.real(torch.mean(dim=1, input=((2 + lr + e_lr2 * (lr - 2))) / (lr ** 3)))
        f3 =torch.real(torch.mean(dim=1, input=((-4 - 3 * lr - lr ** 2 + e_lr2 * (4 - lr))) / (lr ** 3)))

        return q,f1,f2,f3
    def ff_exact(self,x):
        qz=(torch.exp(x / 2) - 1) / x
        f1z=(torch.exp(x) * (4 - 3 * x + x ** 2) - 4 - x) / (x ** 3)
        f2z=(2 + x + torch.exp(x) * (x - 2)) / (x ** 3)
        f3z=(-4 - 3 * x - x ** 2 + torch.exp(x) * (4 - x)) / (x ** 3)
        return qz,f1z,f2z,f3z
    def ff_taylor_3(self,x):
        q=0.5 + x / 8 + x ** 2 / 48 + x ** 3 / 384
        f1=(1 + x + 9 * x ** 2 / 20 + 2 * x ** 3 / 15) / 6
        f2=(30 + 15 * x + 4.5 * x ** 2 + x ** 3) / 180
        f3=1 / 6 - (x + 3) * (x ** 2) / 360
        return q,f1,f2,f3

    def ff_ffexact(self, x):
        q, f1, f2, f3 = self.ff(x)
        zq, zf1, zf2, zf3 = self.ff_exact(x)
        q = torch.where(torch.abs(x) < eta_stb_change, q, zq)
        f1 = torch.where(torch.abs(x) < eta_stb_change, f1, zf1)
        f2 = torch.where(torch.abs(x) < eta_stb_change, f2, zf2)
        f3 = torch.where(torch.abs(x) < eta_stb_change, f3, zf3)
        return q, f1, f2, f3

    def exact_taylor(self, x):
        q, f1, f2, f3 = self.ff_taylor_3(x)
        zq, zf1, zf2, zf3 = self.ff_exact(x)
        q = torch.where(torch.abs(x) < eta_stb_change, q, zq)
        f1 = torch.where(torch.abs(x) < eta_stb_change, f1, zf1)
        f2 = torch.where(torch.abs(x) < eta_stb_change, f2, zf2)
        f3 = torch.where(torch.abs(x) < eta_stb_change, f3, zf3)
        return q, f1, f2, f3

    functions={
        0:ff,1:ff,
        10:ff_ffexact,
        30:exact_taylor,
        3:ff_taylor_3
    }
    def forward(self, x):
        # print(f"forward,{x.type}")
        return self.functions[self.choose](self,x)
    def forward_scheme(self,x):
        y=self.forward(timegrid*x)
        return tuple(timegrid*i for i in y)




f=coeff_scheme()
Q,f1,f2,f3=f.forward_scheme(L)


from utilities3 import *
from utilities import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.load('../reachout/KS_128_0503.pt',map_location=device) #4,351,1024

b=a**2
c=fft.irfft(fft.rfft(b,norm='forward')[:65], norm='forward')  # shape: 4*351*128
f_a=fft.irfft(fft.rfft(a,norm='forward')[:65], norm='forward')  # 4,351,128
d=c-f_a**2  #4,351,128
input_xTr=a[:3,::10,::8].clone().reshape(-1,128) #105,128
input_yTr=d[:3,::10,::8].clone().reshape(-1,128) #105,128
x_normalizer = UnitGaussianNormalizer(input_xTr)
y_normalizer = UnitGaussianNormalizer(input_yTr)
del d,f_a,c,b,a,input_yTr,input_xTr
from vision_transformer import vit_b_ks
model = vit_b_ks(num_classes=128).cuda()  # nn.Transformer(nhead=16, num_encoder_layers=12)


def nonLterm(x):
    nlterm=x**2

    # wcw.sss(x)
    x=x_normalizer.encode(x).float().unsqueeze(1).unsqueeze(1)#n,1,1,x
    # wcw.sss(x)
    with torch.no_grad():
        out=model(x)#n,x
        # wcw.sss(out)
        y=y_normalizer.decode(out)#n,x
        # wcw.sss(y)
        nlterm+=y
        # exit()


    return nlterm#n,x

g=-0.5j*k
def nonLterm_spectual(x):#x is in Fourier space
    # yy=torch.fft.ifft(x)
    yy=torch.real(torch.fft.ifft(x,norm=norm))
    # yy=torch.real(torch.fft.ifft(x,norm='forward'))
    nltm=nonLterm(yy)
    out=torch.fft.fft(nltm,norm=norm)
    # out=torch.fft.fft(nltm,norm='forward')
    out*=g
    # out*=k_proj

    # s_tensor=(1j)*k*x #D_x u (F)
    # s3_derivative=((1j*k)**clos_d)*x #D^3 u  (F)
    # real_s1=torch.real(torch.fft.ifft(s_tensor,norm=norm))
    # real_s3=torch.real(torch.fft.ifft(s3_derivative,norm=norm))
    # clos=torch.fft.fft((real_s1**2)*real_s3,norm=norm)
    # out+=(2*g*clos)*(((cs*x_grid)**2).real)   # 2g:-D_x

    return out



tt=0
import time as tm
tt1=tm.time()
for n in range(1,nmax+1):
    t=n*timegrid
    nl_v=nonLterm_spectual(v) #nx
    a=E_2*v+Q*nl_v
    nl_a=nonLterm_spectual(a)
    b = E_2 * v + Q * nl_a
    nl_b=nonLterm_spectual(b)
    c = E_2 * a + Q * (2*nl_b-nl_v)
    nl_c=nonLterm_spectual(c)
    v=E*v+nl_v*f1+2*(nl_b+nl_a)*f2+nl_c*f3

    # if n%1==0:
    if n%nplt==0:

        u=torch.real(torch.fft.ifft(v,norm=norm))
        # print(u)
        ut=torch.cat((ut,u.unsqueeze(-2)),dim=-2)
        # vt=torch.cat((vt,v.unsqueeze(-2)),dim=-2)
        tt = np.hstack((tt, t))
        if t%1==0:
            wcw.ppp(t)

            # print(u[0, 5])
            # wcw.sss(u)
            # wcw.sss(vt)
            # print(vt[0,-1,2])
    # if n>0:
    #     exit()
tt2=tm.time()
print(tt2-tt1)
# exit()
with open("ks_sgs 5e-3j d1.txt", "w") as file:
    file.write(f'1traj, T=150,dt=0.001\n{tt2-tt1}')
exit()

if pre_check:
    vt=torch.fft.fft(ut,norm=norm)
    # mode=torch.tensor([1,3,5,7]).to(device)
    mode=torch.tensor([1,3,5,7,16]).to(device)
    # mode=torch.tensor([20]).to(device)
    vsub=vt[:,:,mode]#sp
    print(vsub.shape) #100,501,5
    # #
    plt.figure(figsize=(8,6))
    xplot=np.arange(start=0,stop=tmax+0.01,step=dt_save)
    yplot=torch.log(torch.abs(torch.real(vsub))[0]).cpu().numpy()
    print(yplot.shape)
    for i in range(len(mode)):
        plt.plot(xplot,yplot[:,i],'-',label=f'k={mode[i].item()}')
    plt.xlabel('time')
    plt.ylabel('log|Re(Fu_k)|')
    plt.legend()
    name='temp_save/log Fk-t'
    plt.savefig(name+file_name+'jpg')
    plt.clf()

ut.cpu()

import sys
sys.path.append(path_sys['ks'])
import plot_stat_new as stat

if pre_check:
    import sys
    sys.path.append(path_sys['ks'])
    from plot_stat_new import energy_spectual as eng

    sgs_out={'eng':eng(dt_save=dt_save,start_time=start_plot_time,ut=ut),'res':128}
    # wcw.sss(sgs_out['eng'])
    # wcw.ppp(sgs_out['eng'])

    ours=torch.load(path_stat['ks']+"model_pde(416)_res=128_les=0.pt")
    dns=torch.load(path_stat['ks']+"dns.pt")
    les=torch.load(path_stat['ks']+"les.pt")
    # plotlist=[sgs_out,ours['eng'],les['eng'],dns['eng']]
    plotlist=[sgs_out,ours,les]
    taglist=['sgs','ours','les']

    linkb=path_sys['ks']+'/data/stat_save/dns.pt'
    base_gt = torch.load(linkb)
    energy_k=60
    if energy_k:
        yplot = [base_gt['eng'][1:energy_k]]
        for i in range(len(plotlist)):
            # if plotlist[i]['res']<=128:
            yplot.append(plotlist[i]['eng'][1:energy_k])

    else:
        yplot = [base_gt['eng'][1:152]]
        for i in range(len(plotlist)):
            if plotlist[i]['res'] <= 128:
                yplot.append(plotlist[i]['eng'][1:64])
            else:
                yplot.append(plotlist[i]['eng'][1:152])
    tagg=['dns']
    for i in range(len(plotlist)):
        tagg.append(taglist[i])
    taglist=tagg
    wcw.plotline(yplot, xname='k-th(Fourier mode)', yname='log Energy', yscale='log', have_x=False,
                 title='Energy Spectual', overlap=1, label=taglist, linewidth=2.5)
    name = 'temp_save/energy_'
    filename=f'cs={cs},n={Nsum},{[start_plot_time,tmax]}({file_id})'
    plt.savefig(name + filename + '.jpg')
    plt.clf()


else:
    filename=f'ks_sgs_cs={cs},closd={clos_d},n={Nsum},{[start_plot_time,tmax]}'

    stat.save_stat_info(dt_save=dt_save,filename=filename,tag='SGS',start_time=start_plot_time,ut=ut,use_link=0,default_flnm=0)

    sgsdct=torch.load(filename+'.pt')
    stat.save_all_err(sgsdct,filename=filename,default_flnm=0)






# torch.save(ut,file_name+'_test_ut.pt')
#"fig plot"
exit()
#######save data
vt.cpu()
ut.cpu()
torch.save(vt,file_name+'_test_vt.pt')
torch.save(ut,file_name+'_test_ut.pt')


plt.figure(figsize=(8, 6))
# ax = fig.gca(projection='3d')
# tt, x = np.meshgrid(tt, x)
# surf = ax.plot_surface(tt, x, uu.transpose(), cmap=cm.coolwarm, linewidth=0, antialiased=False)
# fig.colorbar(surf, shrink=0.5, aspect=5)
# plt.show()

xx=gridx.cpu().numpy()
uu_plot=ut[0].cpu().numpy()
print(tt.shape,xx.shape,uu_plot.shape)
heatmap = plt.pcolormesh(tt, xx, uu_plot.transpose(), cmap='viridis', shading='auto')

# 添加颜色条
plt.colorbar(heatmap, label='Function Value')

# 添加其他图形元素，例如轴标签和标题
plt.xlabel('t-axis')
plt.ylabel('x-axis')
plt.title('Heatmap of a 2D Function')
name='u(x,t)_'
plt.savefig(name+file_name+'jpg')
# plt.savefig('zz1(mix)+3(taylor' +str(eta_stb_change)+')_'+str(N)+',niu='+str(niu)+',M='+str(M)+',h='+str(timegrid)+'.png')
# 显示图形
plt.clf()




# mode=torch.tensor([1,3,5,7]).to(device)
mode=torch.tensor([1,3,5,7,16]).to(device)
# mode=torch.tensor([20]).to(device)
vsub=vt[:,:,mode]#sp
print(vsub.shape)
# #
plt.figure(figsize=(8,6))
xplot=np.arange(start=0,stop=tmax+0.01,step=dt_save)
yplot=torch.log(torch.abs(torch.real(vsub))[0]).cpu().numpy()
print(yplot.shape)
for i in range(len(mode)):
    plt.plot(xplot,yplot[:,i],'-',label=f'k={mode[i].item()}')
plt.xlabel('time')
plt.ylabel('log|Re(Fu_k)|')
plt.legend()
name='log Fk-t'
plt.savefig(name+file_name+'jpg')
plt.clf()

plt.figure(figsize=(8,6))
yplot_sum=torch.cumsum((torch.abs(vsub))**2,dim=-2)
# yplot_sum=torch.cumsum((torch.abs(vsub))**2,dim=0)
print(yplot_sum.shape)
tt=torch.arange(start=1,end=num_plot+1.5,step=1).to(device)
print(tt.shape)
yplot=(torch.mean(yplot_sum/(tt.view(-1,1)),dim=0)/dt_save).cpu().numpy()
for i in range(len(mode)):
    plt.plot(xplot,yplot[:,i],'-',label=f'k={mode[i].item()}')
    # plt.plot(xplot[1000:],(yplot[:,i][1000:])/yplot[:,i][1000],'-',label=f'k={mode[i].item()}')
plt.xlabel('time')
plt.ylabel('bar{Re(Fu_k)}')
plt.legend()
name='unnormalizaed energy spectual'
plt.savefig(name+file_name+'jpg')
plt.clf()

exit()
tau=[1,2,3,4,5]
tt=torch.arange(start=1,end=num_plot+1.5,step=1).to(device)
plt.figure(figsize=(8,6))
for i in range(len(tau)):
    wcw.ppp(i)
    xplotx=xplot[:-tau[i]]
    wcw.sss(xplotx)

    u1=ut[tau[i]:,:]
    u2=ut[:-tau[i],:]
    wcw.sss(u1)
    wcw.sss(u2)
    uutau=torch.sum(u1*u2,dim=1)/N
    uuplot_sum=torch.cumsum(uutau,dim=0)
    uuplot = (uuplot_sum / (tt[:-tau[i]] * dt_save)).cpu().numpy()

    plt.plot(xplotx,uuplot,'-',label=f'tau={tau[i]*dt_save}')
plt.xlabel('time')
plt.ylabel('bar_t\int u(x,t)u(x,t+tau)dx')
plt.legend()
name='correlation'
plt.savefig(name+file_name+'jpg')
plt.clf()







