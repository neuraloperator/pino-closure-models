import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

import My_TOOL as myt
import math as mt
import wandb
import torch.fft as fft

import seaborn as sns
import numpy.random as random
from scipy.stats import gaussian_kde

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def energy_spectual(dt_save,start_time=0,ut=None, path=None,use_link=0,traj_btz=0):
    ''' save at ./data/stat_save/
    :param dt_save:
    :param filename: end with '.'
    :param start_time: real time(second), will be transformed into index
    :param ut: N_sample* Time_grid * X_grid
    :param path: saved pt should be ut
    :param use_link:!!!!! remember to endow it!
    :return: a k-numpy (k=res, only the first half should be used; k=0 should be ignored; for multi-res plot:128&1024, be careful to handle the wavenum larger than 64)
    '''
    print('energy:start')
    #input type: 'F/f/fourier/Fourier/vt', 'ut/u', 'link/pt'
    if not use_link:
        if ut==None:
            print('error in None ut, energy stat')
            exit()
        vt=fft.fft(ut,norm='forward')
    # elif input_type in ['link']:
    else:
        if path==None:
            print('error in None path, energy stat')
            exit()

        ut = torch.load(path, map_location=device)
        vt = fft.fft(ut, norm='forward')

    # hi_mode=vt.shape[-1]
    if traj_btz==0:
        traj_btz=vt.shape[0]*vt.shape[1]+1
    traj_cycle=mt.ceil(vt.shape[0]/traj_btz)

    start_plot=int(start_time/dt_save)
    # vsub=vt[:,start_plot:,:]
    y_stat_save=[]
    for ii in range(traj_cycle):
        vsub=vt[ii*traj_btz:(ii+1)*traj_btz,start_plot:,:]#n,t,k
        y_stat=torch.sum(2*((torch.abs(vsub)) ** 2), dim=-2)
        # y_stat=torch.mean(torch.sum(2*((torch.abs(vsub)) ** 2), dim=-2),dim=0)
        timeL=vsub.shape[-2]
        # y_stat=y_stat/timeL/dt_save   Think carefully! Should not divide dt_save. Anotherway to think: average over sample from invariant distribution
        y_stat_save.append((y_stat/timeL).cpu())
    y_stat_save=torch.cat(y_stat_save,dim=0)#n,k
    y_stat=torch.mean(y_stat_save,dim=0)#k
    '''the cutting-off or zero-pad to high mode should be done in plotting functions'''
    print('Energy:done!')
    return y_stat.numpy()

def space_correlation(dt_save,start_time=0,ut=None,vt=None, path=None,input_type='link',traj_btz=0):
    ''' save at ./data/stat_save/
    the returned result should be multiplied by L to get the true value
    plot x_shift from 0 to n/128*L, n=50
    :param dt_save:
    :param filename: filename: end with '.'
    :param start_time:
    :param ut: N_sample* Time_grid * X_grid
    :param vt:
    :param path: saved pt should be ut
    :param input_type:'link'(load tensor) or 0(use ut tensor)
    :return:numpy, size=51 (move 0,1,..50 dx, dx=L/128)
    '''
    print('Correlation:start!')
    #input type: 'F/f/fourier/Fourier/vt', 'ut/u', 'link/pt'
    if input_type in ['F','f','vt','fourier','Fourier']:
        if vt==None:
            print('error in None ut, space_correlation')
            exit()

        ut=torch.real(fft.ifft(vt,norm='forward'))
    elif input_type in ['link']:
        print('01/23: logged data should be ut')
        if path==None:
            print('error in None path, space_correlation')
            exit()
        ut = torch.load(path, map_location=device)
    else:
        if ut==None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    resolution=ut.shape[-1]
    if resolution < 128:
        print("resolution for this plot should be at least 128; now doing fft and zero padding in high mode and ifft")
        vtt = fft.rfft(ut,norm='forward')
        ut=torch.real(fft.irfft(vtt,n=128,norm='forward'))
        resolution=128

    if traj_btz==0:
        traj_btz=ut.shape[0]*ut.shape[1]+1
    traj_cycle=mt.ceil(ut.shape[0]/traj_btz)
    start_plot=int(start_time/dt_save)


    resolution=ut.shape[-1]

    dx=resolution//128
    plot_num=51
    corl_stat=[[]for i in range(plot_num)]
    for ii in range(traj_cycle):
        usub=ut[ii*traj_btz:(ii+1)*traj_btz,start_plot:]
        timeL = usub.shape[-2]
        for i in range(plot_num):
            usub1=torch.roll(usub,shifts=i*dx,dims=-1)
            usubb=usub*usub1
            corl_stat[i].append(torch.sum(torch.mean(usubb,dim=-1),dim=-1)/timeL)
    for i in range(plot_num):
        corl_stat[i]=torch.mean(torch.cat(corl_stat[i],dim=0),dim=0).cpu().item()
    print('Correlation:done!')
    return (torch.tensor(corl_stat)).numpy()


def auto_corl_coef(dt_save, start_time=0, ut=None, vt=None, path=None, input_type='link', traj_btz=0):
    ''' save at ./data/stat_save/
    the returned result should be multiplied by L to get the true value
    plot x_shift from 0 to n/128*L, n=50
    :param dt_save:
    :param filename: filename: end with '.'
    :param start_time:
    :param ut: N_sample* Time_grid * X_grid
    :param vt:
    :param path: saved pt should be ut
    :param input_type:'link'(load tensor) or 0(use ut tensor)
    :return: complex tensor, len=res//2+1
    '''
    print('acc:start')
    # input type: 'F/f/fourier/Fourier/vt', 'ut/u', 'link/pt'
    if input_type in ['F', 'f', 'vt', 'fourier', 'Fourier']:
        if vt == None:
            print('error in None ut, space_correlation')
            exit()

        ut = torch.real(fft.ifft(vt, norm='forward'))
    elif input_type in ['link']:
        print('01/23: logged data should be ut')
        if path == None:
            print('error in None path, space_correlation')
            exit()
        ut = torch.load(path, map_location=device)
    else:
        if ut == None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    resolution = ut.shape[-1]
    if resolution < 128:
        print("resolution for this plot should be at least 128; now doing fft and zero padding in high mode and ifft")
        vtt = fft.rfft(ut, norm='forward')
        ut = torch.real(fft.irfft(vtt, n=128, norm='forward'))
        resolution = 128

    if traj_btz == 0:
        traj_btz = ut.shape[0] * ut.shape[1] + 1
    traj_cycle = mt.ceil(ut.shape[0] / traj_btz)
    start_plot = int(start_time / dt_save)

    # usub=(ut[:,start_plot:,:]).permute([2,0,1]) #X,N,T

    resolution = ut.shape[-1]
    corl_stat=[[]for i in range(resolution)]
    for ii in range(traj_cycle):
        usub = ut[ii * traj_btz:(ii + 1) * traj_btz, start_plot:]#.cuda()
        ubar=torch.mean(usub,dim=-1,keepdim=True)
        usub=usub-ubar
        del ubar
        ubar2=torch.mean(usub**2,dim=-1)#n.t
        for i in range(resolution):
            usub1=torch.roll(usub,shifts=i,dims=-1)
            usubb=usub*usub1
            corl_stat[i].append(torch.mean(torch.mean(usubb,dim=-1)/ubar2,dim=1))
            if i%20==0:
                myt.ppp(i)
    for i in range(resolution):
        corl_stat[i] = torch.mean(torch.cat(corl_stat[i], dim=0), dim=0).cpu().item()
    print('acc:done')
    return (fft.rfft(torch.tensor(corl_stat),norm='forward')) #128->65; 1024->513

from pathlib import Path
path = Path(__file__).resolve().parent.as_posix()
emp_range=torch.load(path+'/../data/ks_stat_uv_emp_range.pt')#u:[2], v:[200]

def u_denity(dt_save,start_time=0,ut=None,path=None,input_type='link',bin=200,gamma=1.1,data_sp=2000000):
    '''
    :param dt_save:
    :param start_time:
    :param ut:
    :param path:
    :param input_type:
    :param bin:
    :param gamma: will compute cempirical measure with range =1.1x range of ground truth
    :return: hist,bins,x_vals,y_vals:
            first two for histo plot; last two for kde plot.
            first two: len= bin, bin+1; last two: bin+1, bin+1
    '''
    print('u:start')
    if input_type in ['link']:
        print('01/23: logged data should be ut')
        if path == None:
            print('error in None path, space_correlation')
            exit()
        ut = torch.load(path, map_location=device)
    else:
        if ut == None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    start_plot=int(start_time/dt_save)
    b=ut[:,start_plot:].cpu().numpy()
    a = b.reshape(-1)[random.permutation(len(b.reshape(-1)))[:data_sp]]
    left=gamma*emp_range['u'][0]
    right=gamma*emp_range['u'][1]
    hist, bins = np.histogram(a, bins=bin, density=True,range=(left,right))
    print('u:hist:done!')
    kde=gaussian_kde(a)
    x_vals = np.linspace(left, right, bin+1,endpoint=True)
    y_vals = kde(x_vals)
    print('u:done!')
    return hist,bins,x_vals,y_vals
def v_density(dt_save,start_time=0,ut=None,path=None,input_type='link',bin=200,gamma=1.1):
    '''
    :param dt_save:
    :param start_time:
    :param ut:
    :param path:
    :param input_type:
    :param bin: should try: 50,100,200
    :param gamma:
    :return: a list of valule-gir-value-grid  (histogram first, kpe next)
    '''
    print('v:start!')
    if input_type in ['link']:
        print('01/23: logged data should be ut')
        if path == None:
            print('error in None path, space_correlation')
            exit()
        ut = torch.load(path, map_location=device)
    else:
        if ut == None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    res=ut.shape[-1]
    start_plot=int(start_time/dt_save)
    out=[]
    rho=(fft.rfft(ut[:,start_plot:],norm='forward')).abs().cpu()
    for k in range(min(res//2+1,200)):
        left =0
        right = gamma * emp_range['v'][k]
        a=(rho[:,:,k]).reshape(-1).numpy()
        hist, bins = np.histogram(a, bins=bin, density=True, range=(left, right))
        kde = gaussian_kde(a)
        x_vals = np.linspace(left, right, bin + 1, endpoint=True)
        y_vals = kde(x_vals)
        out.append([hist,bins,x_vals,y_vals])
        if k%5==0:
            print(f'v-k={k}:done')

    print('v:done!')
    return out

def save_stat_info(dt_save,filename, tag,start_time=0, ut=None, path=None, use_link=1,gamma=1.1,data_sp=2000000,traj_btz=0,default_flnm=1,*kwargs):
    if use_link:
        if path==None:
            print('L283 in plot_stat_new.py:no path link')
            exit()
        ut=torch.load(path,map_location=device)
    else:
        if ut==None:
            print('L288 in plot_stat_new.py:no data')
            exit()
    out={}
    out.update(kwargs)
    out['dtsave']=dt_save
    out['n']=ut.shape[0]
    out['t']=[start_time,ut.shape[1]/dt_save]
    out['res']=ut.shape[-1]
    out['eng']=energy_spectual(dt_save=dt_save,start_time=start_time,ut=ut,use_link=0,traj_btz=traj_btz)
    out['cor']=space_correlation(dt_save=dt_save,start_time=start_time,ut=ut,input_type=0,traj_btz=traj_btz)
    out['acc']=auto_corl_coef(dt_save=dt_save,start_time=start_time,ut=ut,input_type=0,traj_btz=traj_btz)
    out['u']={}
    out['v']={}
    def f(n):
        out['u'][n]=u_denity(dt_save=dt_save,start_time=start_time,ut=ut,input_type=0,bin=n,gamma=gamma,data_sp=data_sp)
        out['v'][n] = v_density(dt_save=dt_save, start_time=start_time, ut=ut, input_type=0, bin=n, gamma=gamma)

    f(50) # Larger number of discretization grids will make eastimation of pdf inaccurate, since it requires far more snapshots to converge. Have verified that 50-100 reach the optimal for current setting.

    out['tag']=tag
    if default_flnm:
        path_name='../data/stat_save/'+filename+'.pt'
    else:
        path_name=filename+'.pt'
    torch.save(out,path_name)
    myt.mmm('save_stat')
    return

def plot_all_stat(plotlist,filename,taglist=None,k_plot=55,energy_k=0,acc_k=0,vds_k=64,dns_tag='FRS'):
    '''
    :param plotlist: list of dict- from loading pt
    :param filename:
    :param taglist: list of str(for plotting labels;if none: use pt.tag)
    :return:
    '''

    linwid=3
    stcksize=18
    titlesz=25
    legendsz=18


    linkb=path+'/../data/stat_save/frs.pt'
    base_gt = torch.load(linkb)
    if not isinstance(k_plot,list):
        k_plot=[k_plot]
    if not isinstance(plotlist,list):
        plotlist=[plotlist]
    if taglist==None:
        taglist=[dns_tag]
        for i in range(len(plotlist)):
            taglist.append(plotlist[i]['tag'])
    else:
        tagg=[dns_tag]
        for i in range(len(plotlist)):
            tagg.append(taglist[i])
        taglist=tagg
    if energy_k:
        yplot=[base_gt['eng'][1:energy_k]]
        for i in range(len(plotlist)):
            # if plotlist[i]['res']<=128:
            yplot.append(plotlist[i]['eng'][1:energy_k])

    else:
        yplot=[base_gt['eng'][1:152]]
        for i in range(len(plotlist)):
            if plotlist[i]['res']<=128:
                yplot.append(plotlist[i]['eng'][1:64])
            else:
                yplot.append(plotlist[i]['eng'][1:152])

    myt.plotline(yplot, xname='k-th(Fourier mode)', yname='log Energy', yscale='log', have_x=False,
                 title='Energy Spectual', overlap=1, label=taglist,  linewidth=linwid,titlesz=titlesz,legendsz=legendsz,stcksize=stcksize)
    name = '../fig_save/energy_'
    plt.savefig(name + filename + '.jpg')
    plt.clf()

    if acc_k:
        yplot=[base_gt['acc'][1:acc_k].abs()]
        for i in range(len(plotlist)):

            yplot.append(plotlist[i]['acc'][1:acc_k].abs())

    else:
        yplot=[base_gt['acc'][1:122].abs()]
        for i in range(len(plotlist)):
            if plotlist[i]['res']<=128:
                yplot.append(plotlist[i]['acc'][1:65].abs())
            else:
                yplot.append(plotlist[i]['acc'][1:122].abs())

    myt.plotline(yplot, xname='k-th(Fourier mode)', yname='log |acc|', yscale='log', have_x=False, title='Auto Correlation Coefficient',
                 overlap=1, label=taglist,  linewidth=linwid,titlesz=titlesz,legendsz=legendsz,stcksize=stcksize)
    name = '../fig_save/acc_'
    plt.savefig(name + filename + '.jpg')
    plt.clf()

    plt.figure(figsize=(8.5, 6))
    yplot=[base_gt['cor']]
    for i in range(len(plotlist)):
        yplot.append(plotlist[i]['cor'])
    xx=np.linspace(0,6*mt.pi*50/128,51,endpoint=True)
    yplot.append(xx)

    myt.plotline(yplot, xname='h', yname='Correlation', yscale='linear', have_x=True,
                 title='Spatial Correlation', overlap=1, label=taglist,  linewidth=linwid,titlesz=titlesz,legendsz=legendsz,stcksize=stcksize)
    name = '../fig_save/cor_'
    plt.savefig(name + filename + '.jpg')
    plt.clf()
    myt.mmm(381)

    def f(n):
        x=base_gt['u'][n]
        plt.figure(figsize=(8.5, 6))
        plt.plot(x[2], x[3], label=dns_tag, linewidth=linwid)
        for i in range(len(plotlist)):
            x=plotlist[i]['u'][n]
            plt.plot(x[2],x[3],label=taglist[i+1],linewidth=linwid)
        plt.title(f'U Density')
        plt.legend(fontsize=legendsz)
        plt.tight_layout()
        name = f'../fig_save/u_{n}bin'
        plt.savefig(name + filename + '.jpg')
        plt.clf()

        for k in k_plot:
            plt.figure(figsize=(8.5, 6))
            x = base_gt['v'][n][k]
            plt.plot(x[2], x[3], label=dns_tag, linewidth=linwid)
            for i in range(len(plotlist)):
                x = plotlist[i]['v'][n][k]
                plt.plot(x[2], x[3], label=taglist[i + 1], linewidth=linwid)
            plt.title(f'|V_{k}| Density')
            plt.legend(fontsize=legendsz)
            plt.tight_layout()
            name = f'../fig_save/v{k}_{n}bin'
            plt.savefig(name + filename + '.jpg')
            plt.clf()

        x = base_gt['v'][n]

        def g(q, tag, lin):
            # scc = 'linear' if lin else 'log'
            lst = []
            for k in range(1, vds_k):
                qq = np.sum(np.abs(q[k][3] - x[k][3]) * (q[k][2][2] - q[k][2][1]))/2# q,k,2,1;q,k,2,1,  end 2 and 1 stands for arbitrary t+1 and t (due to uniform grid)

                lst.append(qq)

            plt.plot(np.array(lst), label=tag , linewidth=linwid)

        for lin in [0,1]:
            plt.figure(figsize=(8.5, 6))
            for i in range(len(plotlist)):
                g(plotlist[i]['v'][n],taglist[i+1],lin=lin)
            scc = 'linear' if lin else 'log'
            plt.yscale(scc)
            plt.title(f'Total Variation Error of Fourier Modes ({scc} scale)')
            plt.legend(fontsize=legendsz)
            plt.tight_layout()
            name = f'../fig_save/prob_dist_vk_{scc}_{n}bin'
            plt.savefig(name + filename + '.jpg')
            plt.clf()

    f(50)

import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter

def var(x_emp,y_emp):
    E=np.sum(x_emp*y_emp)/(np.sum(y_emp))
    E2=np.sum((x_emp**2)*y_emp)/(np.sum(y_emp))
    return E2-E**2


def save_all_err(statdct:dict,filename,taglist=None,energy_k=61,acc_k=61,vds_k=61,default_flnm=1):

    linkb=path+'/../data/stat_save/frs.pt'
    base_gt = torch.load(linkb)
    if 'var' not in base_gt:
        base_gt['var']=var(base_gt['u'][50][2],base_gt['u'][50][3])
        print('variance of ground truth!')
        print(base_gt['var'])
    if 'var' not in statdct:
        statdct['var']=var(statdct['u'][50][2],statdct['u'][50][2])
        print(f"var of data_stat:")
        print(statdct['var'])

    str_list=['eng_avg','eng_max','acc_avg','acc_max','cor(absERR)_avg','cor(absERR)_max',
              'tv(u)','tv(v)_avg','tv(v)_max', 'cor(rel)_avg','cor_rel_max']
    format_lst=['per','per','per','per','flt','flt','flt','flt','flt','per','per']
    formats = {
        'per': '0.0000%',
        'sci': '0.0000E+00',
        'flt': '0.0000'
    }

    save_lst=[]
    eng_rel_er=np.abs(base_gt['eng'][1:energy_k]-statdct['eng'][1:energy_k])/np.abs(base_gt['eng'][1:energy_k])
    save_lst.append(np.mean(eng_rel_er))
    save_lst.append(np.max(eng_rel_er))


    "acc"

    baseacc=(base_gt['acc'][1:acc_k].abs()).cpu().numpy()
    sttacc=(statdct['acc'][1:acc_k].abs()).cpu().numpy()
    acc_rel_er=np.abs(baseacc-sttacc)/np.abs(baseacc)

    save_lst.append(np.mean(acc_rel_er))
    save_lst.append(np.max(acc_rel_er))
    "cor"
    cor_abs_er= np.abs(base_gt['cor'] - statdct['cor'])
    save_lst.append(np.mean(cor_abs_er))
    save_lst.append(np.max(cor_abs_er))

    "tv_u"
    tv_u=np.sum(np.abs(statdct['u'][50][3]-base_gt['u'][50][3])*(statdct['u'][50][2][2]-statdct['u'][50][2][1]))/2
    save_lst.append(tv_u)

    "tv_v"
    lst = []
    q=statdct['v'][50]
    x=base_gt['v'][50]
    for k in range(1, vds_k):
        qq = np.sum(np.abs(q[k][3] - x[k][3]) * (q[k][2][2] - q[k][2][1])) / 2  # q,k,2,1;q,k,2,1,  end 2 and 1 stands for arbitrary t+1 and t (due to uniform grid)
        # print(qq.type,'qq')
        lst.append(qq)
    tv_v=np.array(lst)
    save_lst.append(np.mean(tv_v))
    save_lst.append(np.max(tv_v))

    "rel_cor"
    cor_rel_er = np.abs(base_gt['cor']- statdct['cor']) / np.abs(base_gt['cor'])
    save_lst.append(np.mean(cor_rel_er))
    save_lst.append(np.max(cor_rel_er))
    data_sv=[item.item() for item in save_lst]
    save_lst=[data_sv]
    df = pd.DataFrame(save_lst, columns=str_list)

    if default_flnm:
        path_name = '../data/stat_save/' + filename + '.xlsx'
    else:
        path_name = filename + '.xlsx'

    print('begin-record')

    with pd.ExcelWriter(path_name, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)

        workbook = writer.book
        worksheet = writer.sheets['Sheet1']

        for col_idx, col_format in enumerate(format_lst, start=1):
            if col_format in formats:

                col_letter = get_column_letter(col_idx)
                for row_idx in range(2, len(df) + 2):
                    cell = worksheet[f'{col_letter}{row_idx}']
                    cell.number_format = formats[col_format]
        print('end record')
        return


"""Old Code for plots"""
def plot_cvg_eng(wavenum,dt_save,file_name,start_time=0,ut=None,vt=None, path=None,input_type=0,multi=False,overlap=False):
    '''
    :param wavenum: int or list: if int plot all wavenumber smaller than wavenum; if list, plot wavenums in list
    :param file_name: str, suffix of final file name
    :param start_plot: Starting time average from which point (index in the file, not real time)
    :dt_save: time interval between contigious plot(real time)
    :param ut: u_{x,t}
    :param vt: F(u)
    :param path: list or str; if multi=True, plot several experiment in the list togethor
    :param input_type: real space function or fourier space function
    :param multi:
    :param overlap: If lines entangle togethor, let overlap=1
    :return:
    '''
    # input type: 'F/f/fourier/Fourier/vt', 'ut/u', 'link/pt'
    if input_type in ['ut', 'u']:
        if ut == None:
            print('error in None ut, energy stat')
            exit()
        vt = fft.fft(ut,norm='forward')
    elif input_type in ['link']:
        if path == None:
            print('error in None path, energy stat')
            exit()
        vt = torch.load(path, map_location=device)
    else:
        if vt == None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    # hi_mode=vt.shape[-1]
    start_plot = int(start_time / dt_save)
    N=vt.shape[-1]
    if type(wavenum)==int:
        wn=wavenum
        if wavenum>N//2-1:
            print(f"wavenum is too large. it could only handle modes up to {N//2-1}")
            wn=N//2-1

        wn=list(range(wn))

    elif type(wavenum)==list:
        wn=[i for i in wavenum if i<N//2-1]

    vsub = vt[:, start_plot:, wn]

    y_stat = (torch.cumsum(2 * ((torch.abs(vsub)) ** 2), dim=-2))
    timeL = vsub.shape[-2]
    tt = torch.arange(start=1, end=timeL + 0.5, step=1).to(device)
    yplot=torch.mean(y_stat/(tt.view(-1,1)),dim=0)
    # myt.sss(yplot)
    tt=(tt.unsqueeze(0)).repeat(len(wn),1)
    # myt.sss(tt)

    yplot=torch.cat([yplot.permute([1,0]),dt_save*(start_plot+tt)],dim=0)


    myt.plotline(yplot,xname='T',yname='Time average of Energy Spectual',title='Convergence of Statistics along Time',
                 label=wn)
    name = '../fig_save/energy_cvg_'
    plt.savefig(name + file_name + 'jpg')
    plt.clf()
    return

def plot_eng_spec(file_name,path_list,label_list,exp_label='experiment',exp_plot=None,yscale='log',wn_max=150):
    '''

    :param file_name: end with '.'
    :param path_list: list of '.pt's (with path ../.....)
    :param label_list: list of str
    :param exp_label: if there are handiful experiment result (energy spectual starting from k=0)
    :param exp_plot: could be None, add if there are handiful experiment result (energy spectual starting from k=0)
                    !!! should be halved(only takes the real part)

    :param yscale:
    :return:
    '''
    if wn_max==None:
        wn_max=4099
    yplot=[]
    for i in path_list:
        yi=torch.load(i,map_location=device)
        N=yi.shape[0]
        yplot.append(yi[1:min(N//2-1,wn_max)])

    # yplot=[torch.load(i,map_location=device)for i in path_list]
    yan=0
    if exp_plot!=None:
        yan=1
        yplot.append(exp_plot)
        label_list.append(exp_label)



    myt.plotline(yplot,xname='k-th(Fourier mode)',yname='log Energy',yscale=yscale,have_x=False,title='Energy Spectual',overlap=1,label=label_list,linewidth=linwid)
    name = '../fig_save/energy_result_'
    plt.savefig(name + file_name + 'jpg')
    plt.clf()
    return

def plot_cvg_cor(cor_num,dt_save,file_name,start_time=0,ut=None,vt=None, path=None,input_type=0,multi=False,overlap=False):

    # input type: 'F/f/fourier/Fourier/vt', 'ut/u', 'link/pt'
    if input_type in ['F', 'f', 'vt', 'fourier', 'Fourier']:
        if vt == None:
            print('error in None ut, energy stat')
            exit()

        ut = torch.real(fft.ifft(vt,norm='forward'))
    elif input_type in ['link']:
        print('01/23: logged data should be ut')
        if path == None:
            print('error in None path, energy stat')
            exit()
        ut = torch.load(path, map_location=device)
    else:
        if ut == None:
            print('error in None vt or wrong input_type, energy stat')
            exit()
    # print("cha cpplx",ut[0,5,5])
    resolution=ut.shape[-1]
    if resolution < 128:
        print("resolution for this plot should be at least 128; now doing fft and zero padding in high mode and ifft")
        vtt = fft.fft(ut,norm='forward')
        a, b = torch.split(vtt, dim=-1, split_size_or_sections=[resolution // 2, resolution // 2])
        sss = vtt.shape
        sss[-1] = 128 - resolution
        vtt = torch.cat([a, torch.zeros(sss).to(device), b], dim=-1)
        ut = torch.real((fft.ifft(vtt)))
        resolution = 128
    # hi_mode=vt.shape[-1]
    start_plot = int(start_time / dt_save)
    usub = (ut[:, start_plot:, :]).permute([2, 0, 1])  # X,N,T
    timeL = usub.shape[-1]
    tt = torch.arange(start=1, end=timeL + 0.5, step=1).to(device)
    resolution = usub.shape[0]

    dx = resolution // 128
    # plot_num = 51
    corl_stat =tt.clone()
    corl_stat=((corl_stat+start_plot)*dt_save).view(1,-1)
    num_gridd = torch.arange(resolution).to(device)
    # print('check_corl stat')
    # print(tt[-1],start_plot,dt_save,corl_stat[-1])
    # print('type',type(cor_num))
    if type(cor_num)==int:
        plot_num=list(range(cor_num))
    else:
        plot_num=cor_num
    # print(plot_num,'plot_num')
    for i in plot_num:
        num_grid = (dx*i + num_gridd) % resolution
        usub1 = usub[num_grid]
        usubb = usub * usub1
        a=torch.mean(torch.cumsum(torch.mean(usubb, dim=0), dim=-1) / tt.view(1,-1), dim=0)
        myt.sss(a)
        corl_stat=torch.cat([a.view(1,-1),corl_stat],dim=0)
        myt.sss(corl_stat)
    plot_num.reverse()
    wn=[f'{i}dx' for i in plot_num]
    myt.plotline(corl_stat,xname='T',yname='Time average of Space Correlation',title='Convergence of Statistics along Time',
                 label=wn)
    name = '../fig_save/correlation_cvg_'
    plt.savefig(name + file_name + 'jpg')
    plt.clf()
    return

def plot_space_cor(file_name,path_list,label_list,exp_label='experiment',exp_plot=None,yscale='linear'):
    '''

    :param file_name: end with '.'
    :param path_list: list of '.pt's (with path ../.....)
    :param label_list: list of str
    :param exp_label: if there are handiful experiment result (energy spectual starting from k=0)
    :param exp_plot: could be None, add if there are handiful experiment result (energy spectual starting from k=0)
                        !!!!! should have length=51 (cor=0~50 unit), 1 unit=L/128
    :param yscale:
    :return:
    '''

    yplot=[]
    for i in path_list:
        yi=torch.load(i,map_location=device)
        N=yi.shape[0]
        yplot.append(yi)
    # yplot=[torch.load(i,map_location=device)for i in path_list]
    yan=0
    if exp_plot!=None:
        yan=1
        yplot.append(exp_plot)
        label_list.append(exp_label)
    myt.plotline(yplot,xname='x',yname='Spatial correlation',yscale=yscale,have_x=False,title='Spatial Correlation',overlap=1,label=label_list)
    name = '../fig_save/correlation_result_'
    plt.savefig(name + file_name + 'jpg')
    plt.clf()
    return


def reduce_dim(datadict):

    if datadict['in']==None or 'in' not in datadict:
        datadict['in']=0
    if datadict['out']==None or 'out' not in datadict:
        datadict['out']=None


    if datadict['n'] == None or 'n' not in datadict:
        datadict['n']=None
        if 'name' not in datadict:
            datadict['name'] = f"{datadict['n']}traj_{datadict['res']}dx_T={[datadict['in'],datadict['out']]}"

        return datadict['data']
    else:
        if 'name' not in datadict:
            datadict['name'] = f"{datadict['n']}traj_{datadict['res']}dx_T={[datadict['in'],datadict['out']]}"
        return datadict['data'].reshape(-1, datadict['res'])
def plot_u_density(datalist:list[dict],filename,fpath='../fig_save/'):
    '''
    :param datalist: dict
                'data': tensor: N*T*X or T*X, have been sorted/spliced (e,g, [t_in:t_out]) before coming in
                'dtsave': float
                'res': resolution
                'n': number of traj
                'name': str, for label
                'in' and 'out': physical time
    :param filename:
    :param k:
    :param fpath:
    :return:
    '''
    umax=max([torch.max(i['data']).item()for i in datalist])
    umin=min([torch.min(i['data']).item()for i in datalist])
    fig, ax = plt.subplots(figsize=(10, 10))
    linewidth = 3
    for dct in datalist:
        # print('begin!!!!')
        b = reduce_dim(dct).cpu().numpy()
        print(dct['name'])
        sns.distplot(ax=ax, a=b.reshape(-1)[random.permutation(len(b.reshape(-1)))[:200000]],
                     kde=True, hist=False, bins=50, label=dct['name'],
                     kde_kws=dict(linewidth=linewidth, linestyle='-'))
    # ax.set_xlim(-3,3)
    # ax.set_ylim(0.05,0.25)
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)

    plt.xlabel('velocity u(x)', fontsize=15)
    plt.ylabel('density', fontsize=15)

    leg = plt.legend(loc='upper right', prop={'size': 15})
    leg.get_frame().set_alpha(0.5)
    plt.savefig(fpath+'u_density'+filename+'.jpg')
    plt.close()
    return

def plot_v_density(datalist:dict,filename,k:list[int],fpath='../fig_save/'):
    '''
    :param datalist:
    :param filename:
    :param k:  modes to plot
    :param fpath:
    :return:
    '''

    if isinstance(k,int):
        k=[k]



    thetamin=0
    thetamax=2*np.pi
    r_list=[]
    theta_list=[]
    for dct in datalist:

        b = fft.fft(reduce_dim(dct),dim=-1,norm='forward')
        print(dct['name'])
        myt.sss(b)
        r_list.append(b.abs().cpu())
        theta_list.append(b.angle().cpu())
    for kk in k:
        fig, ax = plt.subplots(figsize=(10, 10))
        linewidth = 3
        for i in range(len(datalist)):
            if datalist[i]['res']//2<=kk:
                continue
            b=r_list[i][...,kk]
            b = b.cpu().numpy()
            sns.distplot(ax=ax, a=b.reshape(-1),
                         kde=True, hist=False, bins=50, label=datalist[i]['name'],
                         kde_kws=dict(linewidth=linewidth, linestyle='-'))
        # ax.set_xlim(-3,3)
        # ax.set_ylim(0.05,0.25)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        plt.xlabel('k-th mode of u(x)_r', fontsize=15)
        plt.ylabel('density', fontsize=15)

        leg = plt.legend(loc='upper right', prop={'size': 15})
        leg.get_frame().set_alpha(0.5)
        plt.savefig(fpath+f'rho,vk_density_k={kk}_'+filename+'.jpg')
        plt.close()
    for kk in k:
        fig, ax = plt.subplots(figsize=(10, 10))
        linewidth = 3
        for i in range(len(datalist)):
            if datalist[i]['res'] // 2 <= kk:
                continue
            b = theta_list[i][..., kk]
            b = b.cpu().numpy()
            sns.distplot(ax=ax, a=b.reshape(-1),
                         kde=True, hist=False, bins=50, label=datalist[i]['name'],
                         kde_kws=dict(linewidth=linewidth, linestyle='-'))
        # ax.set_xlim(-3,3)
        # ax.set_ylim(0.05,0.25)
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.tick_params(axis='both', which='minor', labelsize=10)

        plt.xlabel('k-th mode of u(x)_theta', fontsize=15)
        plt.ylabel('density', fontsize=15)

        leg = plt.legend(loc='upper right', prop={'size': 15})
        leg.get_frame().set_alpha(0.5)
        plt.savefig(fpath + f'theta,vk_density_k={kk}_' + filename + '.jpg')
        plt.close()
    return




