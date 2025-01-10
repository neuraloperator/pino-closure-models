import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use('Agg')
from utilities import *
from dissipative_utils import *
from ns1w.data_dict import *
import numpy.random as random
from scipy.stats import gaussian_kde
import torch.fft as fft
import My_TOOL as wcw
sinc=wcw.split_input_chkpt_advanced
import math
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import builtins
original_print = builtins.print
def print(*args, **kwargs):
    if 'flush' not in kwargs:
        kwargs['flush'] = True
    original_print(*args, **kwargs)


'''Input type_dict: {
'data': data tensor; 
'legend'/'name': str for plot; 
'n': num of traj: if n==None: data should be T*X*Y; else: N*T*X*Y
'in': start_plotting index (might be different from physical time)
'out' : T for stop plotting
'res': resolution

}'''

def reduce_dim(datadict:dict,keepT=0):
    '''
    :param datadict:  dict: dtsave=1, n=1(traj), in,out: phy time; res: resolution; name: default
    :param keepT: n,t,x,y or nt,x,y
    :return: tensor, n,t,x,y if keepT=1; else: nt,x,y
    '''
    if 'dtsave' not in datadict or datadict['dtsave']==None:
        datadict['dtsave']=1
        print('!!!!!!!!!!!!!!!!!!!! Default: dt_save=1 ----------------------------')

    if datadict['n'] == None or 'n' not in datadict: # single traj, T,X,Y
        datadict['n']='Single'
        if 'name' not in datadict:
            datadict['name'] = f"{datadict['n']}traj_{datadict['res']}dx_T={[datadict['in'],datadict['out']]}"

        return datadict['data'][datadict['in']:datadict['out'],:,:]
    else: # n,t,x,y
        if 'name' not in datadict:
            datadict['name'] = f"{datadict['n']}traj_{datadict['res']}dx_T={[datadict['in'],datadict['out']]}"
        if not keepT:
            return datadict['data'][:datadict['n'], int(datadict['in']/datadict['dtsave']):int(datadict['out']/datadict['dtsave']),
              :, :].reshape(-1, datadict['res'], datadict['res'])
            ## nt,x,y
        else:
            return datadict['data'][:datadict['n'],
                   int(datadict['in'] / datadict['dtsave']):int(datadict['out'] / datadict['dtsave']),
                   :, :]
            # n,t,x,y

'''vorticity spectrum________________________________________________________'''
f1=torch.fft.fft2
splt_fft2=sinc(f1,splt=4,dim=0,dvc_cpt='cpu',fwd_dtype=torch.complex64)
def spectral_energy(u):
    '''
    :param u: either ntxy or txy
    :param s: delted, size of X
    :return: numpy, length of s (1-s), while only the first half is full compute.
    '''
    s=u.shape[-1]
    # T = u.shape[0]
    u = u.reshape(-1, s, s)
    T=u.shape[0]

    # u = torch.fft.fft2(u,norm='forward')
    u = splt_fft2(u,norm='forward')
    print('eng:fft done!')
    # 2d wavenumbers following Pytorch fft convention
    k_max = s // 2
    wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1), \
                            torch.arange(start=-k_max, end=0, step=1)), 0).repeat(s, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k.numpy()
    del k_y,k_x

    # Remove symmetric components from wavenumbers
    index=sum_k


    spectrum = np.zeros((T, s))
    for j in range(1, s + 1):
        ind = np.where(index == j)
        spectrum[:, j - 1] = np.sqrt( ((u[:, ind[0], ind[1]].abs()**2).sum(axis=1)))#sum

    print('eng:done!')
    spectrum = spectrum.mean(axis=0)
    return spectrum
'''Plot vorticity energy spectrual______'''
"""Histogtms+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"""

from pathlib import Path
path = Path(__file__).resolve().parent.as_posix()
emp_range=torch.load(path+'/../data/kf_stat_uv_emp_range.pt')#u:[2], v:[200]

def w_denity(dt_save,start_time=0,ut=None,path=None,input_type='no',bin=50,gamma=1.1,data_sp=800000,down_sample=8,snpashot_down_sample=1):
    '''
    :param dt_save:
    :param start_time: Finally, will be set to 0. intitial time is pre-processed before inputting ut to this function
    :param ut:
    :param path:
    :param input_type:
    the domaing is set to [-3,3], following mno
    :param bin:
    :param gamma:
    :param data_sp:
    down_sample: the whole tensor will take days to compute. In isotropic flow spatial down_sampling is fine for stats.
    :return:
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
    #ut:n,t,x,y
    s=ut.shape[-1]
    ut=ut[:,start_plot:].reshape(-1,s,s)
    snapshot=ut.shape[0]
    snap_samples=math.ceil(snapshot/snpashot_down_sample)
    snap_select=random.permutation(snapshot)[:snap_samples]
    b=ut[snap_select,::down_sample,::down_sample].cpu().numpy()
    a = b.reshape(-1)[random.permutation(len(b.reshape(-1)))[:data_sp]]
    left=-4.5
    right=4.5
    hist, bins = np.histogram(a, bins=bin, density=True,range=(left,right))
    print('u:hist:done!')
    kde=gaussian_kde(a)
    x_vals = np.linspace(left, right, bin+1,endpoint=True)
    y_vals = kde(x_vals)
    print('u:done!')
    return hist,bins,x_vals,y_vals
def v_density(dt_save,start_time=0,ut=None,path=None,input_type='no',bin=50,gamma=1):
    '''
    :param dt_save:
    :param start_time:
    :param ut:
    :param path:
    :param input_type:
    :param bin: should try: 50,100,200
    :param gamma:
    :return: a 2-list of grid-value (kde)  start from 0
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
    wav_nm_les=48
    out=[[0 for i in range(wav_nm_les)]for j in range(wav_nm_les)]
    wcw.sss(ut[:,start_plot:])
    rho=(splt_fft2(ut[:,start_plot:],norm='forward')).abs().cpu()
    wcw.sss(rho)
    for k in range(wav_nm_les//2):
        for kk in range(wav_nm_les//2):
            left =0
            right = gamma * emp_range['v'][k][kk]

            a=(rho[:,:,k,kk]).reshape(-1).numpy()
            # hist, bins = np.histogram(a, bins=bin, density=True, range=(left, right))
            kde = gaussian_kde(a)
            x_vals = np.linspace(left, right, bin + 1, endpoint=True)

            y_vals = kde(x_vals)
            out[k][kk]=[x_vals,y_vals]
        if k%1==0:
            print(f'v-k={k},{kk}:done')

    print('v:done!')
    return out
"""Distribution of dissipation__________________________________________________________-"""
def f2(w,Re=1000):
    return torch.mean(w**2, dim=1) / Re
splt_dssp=sinc(f2,splt=2,dim=0,dvc_cpt='cpu')

def dissipation(w, Re=1000):
    T = w.shape[0]
    s = w.shape[1]
    w = w.reshape(T, s*s)
    return splt_dssp(w,Re=Re)
def dspt_density(w,re=1000,bin=100,data_sp=400000,**kwargs):
    '''
    :param w: ntxy
    :param re:
    :param bin:
    :param data_sp:
    :param kwargs:
    :return:  two elemetn:  gird, histo  (len=bin+1, bin)
    '''
    nn=w.shape[0]
    tt=w.shape[1]
    s=w.shape[-1]
    a=dissipation(w.reshape(-1,s,s),re)[:data_sp] #nt (1d)

    left=torch.min(a).item()
    right=min(torch.max(a).item(),990.03)
    kde=gaussian_kde(a)
    x_vals = np.linspace(left, right, bin+1,endpoint=True)
    y_vals = kde(x_vals)
    print('dspt:done!')
    return x_vals,y_vals
"""Kinetic eenrgy distribuion---------------------------------------------------------"""
splt_w_to_u=sinc(w_to_u,splt=4,dim=0,dvc_cpt='cpu')# the input is n,x,y



def TKE(u,N,T):#u: nt,x,y,2
    # N=u.shape[0]
    # T = u.shape[1]
    s = u.shape[-2]
    u = u.reshape(N,T,-1)#n,t,2s^2
    umean = torch.mean(u, dim=1,keepdim=True)#n,1,2s^2

    u-=umean
    u**=2
    return torch.mean((u), dim=2)#n,t
def tke_density(w, bin=100,data_sp=200000,**kwargs):
    '''
    :param w:   ntxy
    :param bin:
    :param data_sp:
    :param kwargs:
    :return:
    '''
    nn=w.shape[0]
    tt=w.shape[1]
    s=w.shape[-1]
    a=TKE(splt_w_to_u(w.reshape(-1,s,s)),nn,tt).reshape(-1) #nt (1d)


    left=torch.min(a).item()
    right=torch.max(a).item()
    kde=gaussian_kde(a)
    x_vals = np.linspace(left, right, bin+1,endpoint=True)
    y_vals = kde(x_vals)
    print('TKE:done!')
    return x_vals,y_vals,torch.mean(a).item()# square of velocity^2, do not need to /2
def var(w,sp=4000):
    '''
    :param w: ntxy
    # :sp: only sample sp from s*2 spatial grids for each snapshots... useless
    :return: .item()  -pure value
    '''
    w=w.reshape(w.shape[0],w.shape[1],-1)[:sp]# sp here is useless; And we don't need downsample for a scalar, so it is fine.
    var=torch.mean(w**2)-(torch.mean(w))**2
    print('var:done!')
    return var.item()
def cov(w,sp=4000):
    return 0
    s=w.shape[-1]
    u=splt_w_to_u(w.reshape(-1, s, s))#nt,x,y,2
    # u=w_to_u_ntxy(w)#ntxy2
    u1=(u[...,0].reshape(u.shape[0],-1)[:sp]).reshape(-1)
    u2=(u[...,1].reshape(u.shape[0],-1)[:sp]).reshape(-1)
    # u2=u[...,1].reshape(-1)
    cov=torch.mean(u1*u2)-torch.mean(u1)*torch.mean(u2)
    print('cov:done!')
    return cov.item()

def save_stat_info(datadct,filename, tag,re=1000,gamma=1.1,data_sp=2000000,traj_btz=0,default_flnm=1,*kwargs):
    w=reduce_dim(datadct,keepT=1)
    wcw.sss(w)
    out={}
    out.update(kwargs)
    out['dtsave']=datadct['dtsave']
    out['n']=datadct['n']
    out['t']=[datadct['in'],datadct['out']]
    out['res']=w.shape[-1]
    out['eng']=spectral_energy(w)
    out['w']=w_denity(dt_save=out['dtsave'],ut=w)
    out['v']=v_density(dt_save=out['dtsave'],ut=w)
    out['dsp']=dspt_density(w,re=re)
    out['tke']=tke_density(w)
    out['mean_velocity_square']=out['tke'][-1]
    out['var']=var(w)
    out['cov']=cov(w)


    out['default_name']=datadct['name']

    out['tag']=tag
    if default_flnm:
        path_name=path+'/data/stat_save/'+filename+'.pt'
    else:
        path_name=filename+'.pt'
    torch.save(out,path_name)

    return

def plot_all_stat(plotlist,filename,taglist=None,k_plot=[[5,6],[2,2]],energy_k=47,vds_k=[24,24],dns_tag='dns'):
    '''
    :param plotlist:
    :param filename: no'.jpg'
    :param taglist:
    :param k_plot:   for ploting v-desntiy, should be 2-ele-list or a list of 2-ele-lists, betweeen [0,8)*[0,8); 0,0 is nonsense
    :param energy_k:
    :param vds_k:
    :return:
    '''

    linwid=3
    stcksize=18
    titlesz=25
    legendsz=18
    "set up ground truth"

    linkb=path+'/../data/stat_save/frs.pt'
    base_gt = torch.load(linkb)
    if not isinstance(k_plot[0],list):
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
    "plot spectual"
    'energy save start from k=1'
    if energy_k:
        yplot=[base_gt['eng'][:180]]#180
        for i in range(len(plotlist)):
            # if plotlist[i]['res']<=128:
            yplot.append(plotlist[i]['eng'][:energy_k])

    else:
        yplot=[base_gt['eng'][:125]]
        for i in range(len(plotlist)):
            if plotlist[i]['res']<=48:
                yplot.append(plotlist[i]['eng'][1:45])
            else:
                yplot.append(plotlist[i]['eng'][1:125])

    wcw.plotline(yplot, xname='log k(sum of index)', yname='log Energy', yscale='log',xscale='log', have_x=False,
                 title='Energy Spectual', overlap=0, label=taglist, linewidth=linwid,titlesz=titlesz,legendsz=legendsz,stcksize=stcksize)
    name = '../fig_save/energy_'  # each method file has its own temp_save folder
    plt.savefig(name + filename + '.jpg')
    plt.clf()
    plt.figure(figsize=(8.5, 6))
    "plot w(vorticity)_density"
    x = base_gt['w']
    plt.plot(x[2], x[3], label=dns_tag, linewidth=linwid)
    for i in range(len(plotlist)):
        x = plotlist[i]['w']
        plt.plot(x[2], x[3], label=taglist[i + 1], linewidth=linwid)
    plt.title(f'Vorticity Density',fontsize=titlesz)
    plt.xticks(fontsize=stcksize)
    plt.yticks(fontsize=stcksize)
    plt.xlabel('Value',fontsize=titlesz)
    plt.ylabel('Frequency',fontsize=titlesz)
    plt.legend(fontsize=legendsz)
    plt.tight_layout()
    name = f'../fig_save/w_'
    plt.savefig(name + filename + '.jpg')
    plt.clf()

    "plot v_density"
    for k in k_plot:
        x = base_gt['v'][k[0]][k[1]]
        plt.plot(x[0], x[1], label=dns_tag, linewidth=linwid)
        for i in range(len(plotlist)):
            x = plotlist[i]['v'][k[0]][k[1]]
            plt.plot(x[0], x[1], label=taglist[i + 1], linewidth=linwid)
        plt.title(f'|V_{k}| Density',fontsize=titlesz)
        plt.xticks(fontsize=stcksize)
        plt.yticks(fontsize=stcksize)
        plt.xlabel('Value',fontsize=titlesz)
        plt.ylabel('Frequency',fontsize=titlesz)
        plt.legend(fontsize=legendsz)
        plt.tight_layout()
        name = f'../fig_save/v{k}_'
        plt.savefig(name + filename + '.jpg')
        plt.clf()


    "plot_d_tv of v(heat)"
    # if len(plotlist)==1:
    x=base_gt['v']
    def g(q, tag, lin):
        # scc = 'linear' if lin else 'log'
        print('Go this way')
        lst = [[0 for kk in range(vds_k[1])]for k in range(vds_k[0])]
        for k in range(0, vds_k[0]):
            for kk in range(0,vds_k[1]):
                if k==0 and kk==0:
                    continue

                qq = np.sum(np.abs(q[k][kk][1] - x[k][kk][1]) * (q[k][kk][0][2] - q[k][kk][0][1]))/2# q,k,2,1;q,k,2,1,  end 2 and 1 stands for arbitrary t+1 and t (due to uniform grid)

                lst[k][kk]=qq
        plot_x=np.array(list(range(vds_k[0])))
        plot_y=np.array(list(range(vds_k[1])))
        if lin:
            z=np.array(lst).transpose()
            ss=''
        else:

            z=np.log10(np.array(lst).transpose())

            ss='log'
        cbar=[0,1] if lin else [-1.8,0]
        wcw.plotheat(x=plot_x,y=plot_y,z=z,barname='TV dist',title=ss+f'TV error: {tag}',vmin=cbar[0],vmax=cbar[1])
        # plt.plot(np.array(lst), label=tag + '_kde', linewidth=linwid)


    for lin in [0,1]:
        # for i in range(len(plotlist)):
        for i in range(0,len(plotlist)):
            g(plotlist[i]['v'],taglist[i+1],lin=lin)
            scc = 'linear' if lin else 'log'

            name = f'../fig_save/prob_dist_vk_{scc}'
            plt.tight_layout()
            plt.savefig(name +f'({taglist[i+1]})'+ filename + '.jpg')
            plt.clf()

    plt.figure(figsize=(8.5, 6))

    "plot dsption"
    x = base_gt['dsp']
    plt.plot(x[0], x[1], label=dns_tag, linewidth=linwid)
    for i in range(len(plotlist)):
        x = plotlist[i]['dsp']
        plt.plot(x[0], x[1], label=taglist[i + 1], linewidth=linwid)
    plt.xlim(0, 0.2)
    plt.title(f'Dissipation',fontsize=titlesz)
    plt.xticks(fontsize=stcksize)
    plt.yticks(fontsize=stcksize)
    plt.xlabel('Value',fontsize=titlesz)
    plt.ylabel('Frequency',fontsize=titlesz)
    plt.legend(fontsize=legendsz)
    name = f'../fig_save/dspt_'
    plt.tight_layout()
    plt.savefig(name + filename + '.jpg')
    plt.clf()
    "plot tke"
    plt.figure(figsize=(8.5, 6))
    x = base_gt['tke']
    plt.plot(x[0], x[1], label=dns_tag, linewidth=linwid)
    for i in range(len(plotlist)):
        x = plotlist[i]['tke']
        plt.plot(x[0], x[1], label=taglist[i + 1], linewidth=linwid)
    plt.title(f'Kinetic Energy',fontsize=titlesz)
    plt.xticks(fontsize=stcksize)
    plt.yticks(fontsize=stcksize)
    plt.xlabel('Value',fontsize=titlesz)
    plt.ylabel('Frequency',fontsize=titlesz)
    plt.legend(fontsize=legendsz)
    name = f'../fig_save/kte_'
    plt.tight_layout()
    plt.savefig(name + filename + '.jpg')
    plt.clf()

    "var"
    print('\n var')
    print(f'tag={"dns"}: {base_gt["var"]}')
    for i in range(len(plotlist)):
        print(f'tag={taglist[i+1]}: {plotlist[i]["var"]}')
    "mean_squared velocity"
    print('\n mean_velocity_square')
    print(f'tag={"dns"}: {base_gt["mean_velocity_square"]}')
    for i in range(len(plotlist)):
        print(f'tag={taglist[i+1]}: {plotlist[i]["mean_velocity_square"]}')
    "cov"
    print('\n cov')
    print(f'tag={"dns"}: {base_gt["cov"]}')
    for i in range(len(plotlist)):
        print(f'tag={taglist[i + 1]}: {plotlist[i]["cov"]}')



import pandas as pd
import openpyxl
from openpyxl.utils import get_column_letter
def save_all_err(statdct:dict,filename,energy_k=47,vds_k=[24,24],default_flnm=1):
    '''
    :param statdct:
    :param filename:
    :param energy_k:
    :param vds_k:
    :param default_flnm:
    :return:
    '''
    linkb = path + '/../data/stat_save/frs.pt'
    base_gt = torch.load(linkb)
    str_list = ['eng_avg', 'eng_max', 'tv(w)', 'tv(v)_avg', 'tv(v)_max', 'cov','var','var_rel','dsp','tke']
    format_lst = ['per', 'per', 'flt','flt','flt','sci','flt','per','flt','flt']
    formats = {
        'per': '0.0000%',
        'sci': '0.0000E+00',
        'flt': '0.0000'
    }
    print('begin record')
    save_lst = []
    'eng'
    eng_rel_er=np.abs(base_gt['eng'][:energy_k]-statdct['eng'][:energy_k])/np.abs(base_gt['eng'][:energy_k])
    save_lst.append(np.mean(eng_rel_er))
    save_lst.append(np.max(eng_rel_er))

    'tv of w'
    tv_u=np.sum(np.abs(statdct['w'][1+2]-base_gt['w'][1+2])*(statdct['w'][0+2][2]-statdct['w'][0+2][1]))/2
    save_lst.append(tv_u)

    "tv_v"
    lst = [[0 for i in range(vds_k[1])]for j in range(vds_k[0])]
    q=statdct['v']
    x=base_gt['v']
    for k in range(0, vds_k[0]):
        for kk in range(0, vds_k[1]):
            if k == 0 and kk == 0:
                continue
            qq = np.sum(np.abs(q[k][kk][1] - x[k][kk][1]) * (q[k][kk][0][2] - q[k][kk][0][
                1])) / 2  # q,k,2,1;q,k,2,1,  end 2 and 1 stands for arbitrary t+1 and t (due to uniform grid)

            lst[k][kk] = min(qq,1)

    tv_v=np.array(lst)
    save_lst.append(np.mean(tv_v))
    save_lst.append(np.max(tv_v))

    "cov"
    save_lst.append(np.array(abs(statdct['cov'])))

    'var'
    save_lst.append(np.array(statdct['var']))
    save_lst.append(np.array(abs(statdct['var']-base_gt['var'])/base_gt['var']))

    'tv of dsp'
    sm=np.sum(base_gt['dsp'][1])*(statdct['dsp'][0][2]-statdct['dsp'][0][1])
    wcw.ppp(sm)
    tv_dsp=np.sum(np.abs(statdct['dsp'][1]-base_gt['dsp'][1])*(statdct['dsp'][0][2]-statdct['dsp'][0][1]))/2
    save_lst.append(tv_dsp)
    wcw.ppp(tv_dsp)



    'tv of tke'
    tv_tke=np.sum(np.abs(statdct['tke'][1]-base_gt['tke'][1])*(statdct['tke'][0][2]-statdct['tke'][0][1]))/2
    save_lst.append(tv_tke)

    print('end record')


    data_sv = [item.item() for item in save_lst]
    save_lst = [data_sv]
    df = pd.DataFrame(save_lst, columns=str_list)

    if default_flnm:
        path_name = '../data/stat_save/' + filename + '.xlsx'
    else:
        path_name = filename + '.xlsx'


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