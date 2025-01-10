import time as tm
import torch
import math
from dict_ref import *
from timeit import default_timer

from ns1k_sing_pdes_periodic import NavierStokes2d
from random_fields import GaussianRF2d
import numpy as np
import neuralop
import neuralop.wcw.tool_wcw as wcw
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

counter_file = "counter.txt"
file_id=wcw.id_filename(counter_file)

# torch.backends.cuda.max_split_size_mb = 0
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32

# import sys
# sys.path.append(path_sys['kf'])
# import kf_plot_stat_new as stat
# from kf_plot_stat_new import spectral_energy as eng
# exit()
re=1e3
lknm=re #'kf' or 1e4, 1e3

dsct=1#originally: 2
L = 2*math.pi
s = 48 # vor_save: s/dsct*s/dsct
N = 1 #400#00 # total traj
bsize =1  # traj per epoch
'''original code: dt_save=0.1,re=5000.t_traj=5000'''
dt_save=1/4 #1/16

start_save=0 #1800#00
t_traj_phy=100 #3000#00
cs=1# should be 1 if use closure; else None
dt0=1e2
pre_check=0
t_traj=int(t_traj_phy/dt_save)
t_start=int(start_save/dt_save)
# model=bslMLP().to(device)
# model.load_state_dict(torch.load('model_plain21.pt'))


solver = NavierStokes2d(s,s,L,L,device=device,dtype=dtype)
grf = GaussianRF2d(s,s,L,L,alpha=2.5,tau=3.0,sigma=None,device=device,dtype=dtype)

t = torch.linspace(0, L, s+1, dtype=dtype, device=device)[0:-1]
_, Y = torch.meshgrid(t, t, indexing='ij')

f = -4*torch.cos(4.0*Y)

vor = torch.zeros(N,(t_traj-t_start)+1,s//dsct,s//dsct,dtype=torch.float32)
print(1)

for j in range(N//bsize):
    w = grf.sample(bsize)#n,x,y

    # vor[j*bsize:(j+1)*bsize,0,:,:] = w[:,::dsct,::dsct].type(torch.float32)


    tt1=tm.time()
    t1=default_timer()
    for k in range(t_traj):#5k
        if k==t_start:
            vor[j * bsize:(j + 1) * bsize, k - t_start, :, :] = w[:, ::dsct, ::dsct].cpu().type(torch.float32)
        # t1 = default_timer()

        w = solver.advance(w, f, T=dt_save, Re=re, adaptive=True,clos=cs,delta_t0=dt0)
        if k>=t_start:
            vor[j*bsize:(j+1)*bsize,k-t_start,:,:] = w[:,::dsct,::dsct].cpu().type(torch.float32)

        if k%(int(1/dt_save))==0:
            t2 = default_timer()
            print(k*dt_save, t2-t1)
            # wcw.mmm()
    tt2=tm.time()
print(tt2 - tt1)
with open("ns1k_sing.txt", "w") as file:
    file.write(f'{N}traj, T={t_traj_phy},dt={dt_save}\n{tt2 - tt1}')
exit()


    # torch.save(vor[j*bsize:(j+1)*bsize,k+1,:,:], 'sample_2_' + str(j+1) + '.pt')
wcw.sss(vor)
import sys
sys.path.append(path_sys[lknm])
import kf_plot_stat_new as stat
from kf_plot_stat_new import spectral_energy as eng
norm='forward'
if pre_check:
    vt=torch.fft.fft2(vor,norm=norm).reshape(vor.shape[0],vor.shape[1],-1)
    device=vor.device
    # mode=torch.tensor([1,3,5,7]).to(device)
    mode=torch.tensor([1,5,16,56,133]).to(device)
    # mode=torch.tensor([20]).to(device)
    vsub=vt[:,:,mode]#sp
    print(vsub.shape) #100,501,5
    # #
    plt.figure(figsize=(8,6))
    xplot=np.arange(start=start_save,stop=t_traj_phy+0.01,step=dt_save)
    yplot=torch.log(torch.abs(torch.real(vsub))[0]).cpu().numpy()
    print(yplot.shape)
    for i in range(len(mode)):
        plt.plot(xplot,yplot[:,i],'-',label=f'k={mode[i].item()}')
    plt.xlabel('time')
    plt.ylabel('log|Re(Fu_k)|')
    plt.legend()
    name='temp_save/log Fk-t'
    file_name = f'KF_cs={cs},n={N},{[start_save, t_traj_phy]}({file_id})'
    plt.savefig(name+file_name+'.jpg')
    plt.clf()
if pre_check:

    sgs_out={'eng':eng(vor),'res':16}

    ours=torch.load(path_stat['kf']+"pino_fin.pt")
    dns=torch.load(path_stat['kf']+"dns_1500.pt")
    les=torch.load(path_stat['kf']+"les_1200.pt")
    # plotlist=[sgs_out,ours['eng'],les['eng'],dns['eng']]
    plotlist=[sgs_out,ours,les]
    taglist=['sgs','ours','les']

    linkb=path_sys['kf']+'/data/stat_save/dns.pt'
    base_gt = dns
    energy_k=12
    "plot spectual"
    if energy_k:
        yplot = [base_gt['eng'][:energy_k]]
        for i in range(len(plotlist)):
            # if plotlist[i]['res']<=128:
            yplot.append(plotlist[i]['eng'][:energy_k])

    else:
        yplot = [base_gt['eng'][:152]]
        for i in range(len(plotlist)):
            if plotlist[i]['res'] <= 128:
                yplot.append(plotlist[i]['eng'][1:64])
            else:
                yplot.append(plotlist[i]['eng'][1:152])
    tagg=['dns']
    for i in range(len(plotlist)):
        tagg.append(taglist[i])
    taglist=tagg
    wcw.plotline(yplot, xname='k(sum of index)', yname='log Energy', yscale='log', have_x=False,
                 title='Energy Spectual', overlap=1, label=taglist, linewidth=2.5)

    name = 'temp_save/energy_'
    filename=f'KF_cs={cs},n={N},{[start_save,t_traj_phy]}({file_id})'
    plt.savefig(name + filename + '.jpg')
    plt.clf()


else:
    # pass
    # filename=f'ks_sgs_cs={cs},n={Nsum},{[start_plot_time,tmax]}({file_id}).pt'
    filename=f'KF{re}_cs={cs},n={N},{[start_save,t_traj_phy]}'

    #exisit

    sgs = {'tag': 'single_state', 'n': N, 'res': s, 'in': 0, 'out': t_traj_phy,'data':vor,'dtsave':dt_save}
    # pino3k['data'] = torch.load('data/stat_save/pred_re100_model_pde(135)_res16_[1800, 3000].pt', map_location='cpu')
    stat.save_stat_info(sgs,filename=filename,tag=sgs['tag'], default_flnm=0)
    sgsdct = torch.load(filename + '.pt')
    stat.save_all_err(sgsdct, filename=filename, default_flnm=0)


    # stat.save_stat_info(dt_save=dt_save,filename=filename,tag='SGS',start_time=start_plot_time,ut=ut,use_link=0,default_flnm=0)

''' for random init only'''
# torch.save(vor.squeeze(dim=1), 'data/KF_re100_1000trj_16dx_random_ini(1).pt')
# exit()
# torch.save(vor, f'data/re{re}_grid={s}_{dsct}_N={N}_dt={dt_save}_Ttj={start_save}-{t_traj_phy}_(3).pt')