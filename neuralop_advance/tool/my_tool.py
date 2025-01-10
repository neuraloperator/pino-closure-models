import torch
import inspect
import numpy as np

import matplotlib.pyplot as plt
import os

import math as mt
from torch.utils.checkpoint import checkpoint
from torch import nn
# logistics
def id_filename(counter_file):
    '''Everytime use this func, write----
    counter_file = "counter.txt"
    file_id=wcw.id_filename(counter_file)

        -----in that code file.
        This func will create a txt, in which is an int, starting from 0.
        Everytime I run the code, num in txt ++. Use this to identify file name.
    '''
    if os.path.exists(counter_file):
        with open(counter_file, "r") as file:
            counter = int(file.read())
    else:
        counter = 0


    count_int = counter + 1


    with open(counter_file, "w") as file:
        file.write(str(count_int))

    return count_int
def get_file_name(file_path):
    file_name_with_extension = os.path.basename(file_path)  # Get the file name with the suffix
    file_name, extension = os.path.splitext(file_name_with_extension)  # Separate the file name and the suffix
    return file_name

class Choice_Function:
    '''
    input: functions: dictionary, key=str(name) or int, value =f_i (different functions)
    k: the key defined in hyper-para-Ctrl

    Sample Code:
    functions = {
    1: f1,
    2: f2,
    3: f3,
    }
    selector = Choice_Function(functions)

    k=2
    selected_function = selector.select_function(k)
    '''
    def __init__(self, functions):
        self.functions = functions

    def select_function(self, k):
        return self.functions.get(k, None)


torch.set_printoptions(precision=8)
#*******  DEBUG!!!

def ppp(x):
    '''print out x
    x must be variable.
    When x is sth like a[i], I'd better print directly.
    Alternatively: ppp(i);a_i=a[i],ppp(a_i)
    '''
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame, 2)[1]
    local_vars = calling_frame.frame.f_locals

    var_name = None
    for name, value in local_vars.items():
        if value is x:
            var_name = name
            break

    if var_name is not None:
        if isinstance(x,torch.Tensor) or isinstance(x,np.ndarray):
            print(f'{var_name} = \n{x}')
        else:
            print(f'{var_name} = {x}')
    else:
        print(f'Variable (name unknown) = \n{x}')
def sss(x):
    '''print out size of x
    x must be variable.
    When x is sth like a[i], I'd better print directly.
    Alternatively: ppp(i);a_i=a[i],ppp(a_i)
    '''
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame, 2)[1]
    local_vars = calling_frame.frame.f_locals

    var_name = None
    for name, value in local_vars.items():
        if value is x:
            var_name = name
            break

    if var_name is not None:
        if isinstance(x,torch.Tensor) or isinstance(x,np.ndarray):
            print(f'{var_name}.shape= {x.shape}')
        else:
            print(f'{var_name}.shape = {len(x)}')
    else:
        print(f'Variable (name unknown).shape = {x.shape}')
def ccc(x):
    if isinstance(x,torch.Tensor):
        if x.device.type == 'cuda':
            print("Tensor is on GPU")
        else:
            print("Tensor is on CPU")
    elif type(x)==np.ndarray:
        print("It is numpy")
    else:
        print("Neither tensor nor numpy")
def mmm(tag=None,all_gpu=False,check_id:list=[0]):
    "cuda memory check"
    if torch.cuda.is_available():
        num_devices = list(range(torch.cuda.device_count()))
        if not all_gpu:
            num_devices=check_id
        for device_id in num_devices:
            print(f"GPU Memory Usage for cuda:{device_id}----: {tag}:")
            allocated_memory = torch.cuda.memory_allocated(device_id)
            reserved_memory = torch.cuda.memory_reserved(device_id)
            free_memory = torch.cuda.get_device_properties(device_id).total_memory - allocated_memory
            peak_memory_usage = torch.cuda.max_memory_allocated(device_id)

            print(f"Allocated: {allocated_memory / 1024 / 1024:.2f} MB")
            print(f"Reserved: {reserved_memory / 1024 / 1024:.2f} MB")
            print(f"Free: {free_memory / 1024 / 1024:.2f} MB")
            print(f"Peak memory usage: {peak_memory_usage / (1024 * 1024):.2f} MB")
            torch.cuda.reset_peak_memory_stats(device_id)
            print()
    else:
        if (not torch.cuda.is_available()) or (not all_gpu):
            print(f"GPU Memory Usage----: {tag}:")
            allocated_memory=torch.cuda.memory_allocated()
            print(f"Allocated: {allocated_memory/1024/1024:.2f} MB")
            print(f"Reserved: {torch.cuda.memory_reserved()/1024/1024:.2f} MB")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            device_properties = torch.cuda.get_device_properties(device)
            free_memory = device_properties.total_memory - allocated_memory
            print(f"Free: {free_memory/ 1024 / 1024:.2f} MB")
            peak=torch.cuda.max_memory_allocated()
            # stats = torch.cuda.memory_stats(torch.device('cuda'))
            peak_memory_usage = peak
            print(f"Peak memory usage: {peak_memory_usage / (1024 * 1024):.2f} MB")
            torch.cuda.reset_peak_memory_stats()
            print()
def mmmc(tag=None):
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    virtual_memory = psutil.virtual_memory()

    # Current memory usage
    current_memory = memory_info.rss / (1024 ** 2)  # in MB

    # Peak memory usage
    peak_memory = memory_info.vms / (1024 ** 2)  # in MB

    # Available memory
    available_memory = virtual_memory.available / (1024 ** 2)  # in MB

    # Total memory
    total_memory = virtual_memory.total / (1024 ** 2)  # in MB

    # Reserved memory (Total memory - Available memory)
    reserved_memory = total_memory - available_memory
    print(f"CPU Memory Usage----: {tag}:")
    print(f"Current Memory Usage: {current_memory:.2f} MB")
    print(f"Reserved Memory: {reserved_memory:.2f} MB")
    print(f"Peak Memory Usage: {peak_memory:.2f} MB")
    print(f"Available Memory: {available_memory:.2f} MB")
    print("-" * 30)
    # return current_memory, peak_memory, available_memory
def mm(tag=None,all_gpu=True,check_id:list=[0]):
    mmm(tag,all_gpu,check_id)
    mmmc(tag)
def check_dataloader(train_loader):
    current_frame = inspect.currentframe()
    calling_frame = inspect.getouterframes(current_frame, 2)[1]
    local_vars = calling_frame.frame.f_locals

    var_name = None
    for name, value in local_vars.items():
        if value is train_loader:
            var_name = name
            break

    #Total num of samples in the dataloader
    total_samples = len(train_loader.dataset)
    print(f"Total samples in the dataset {var_name} : {total_samples}")

    # iter for a minibatch and check the shape in each batch
    for idx, sample in enumerate(train_loader):
        x_batch, y_batch = sample['x'], sample['y']  # !!!The key in the dataset should be 'x''y' !!!
        print(f"Batch {idx} : x_batch shape: {x_batch.shape}, y_batch shape: {y_batch.shape}")
        # Add further operations here
        if idx == 0:
            break  # Avoid printing too much
def check_pt(file_path):#str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.load(file_path, map_location=device)
    if hasattr(data, 'shape'):
        print(f'Inside the file is {type(data)},')
        print(f"Its shape is {data.shape}\n")
    if isinstance(data, dict):
        print("Keys in the dictionary:", data.keys())
        for key, value in data.items():
            print(
                f"Key: {key}, Value Type: {type(value)}, Value Shape: {value.shape if hasattr(value, 'shape') else None}")
        print('\n')
def check_npy(file_name):#str
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = np.load(file_name)
    print(f"Data Structure: {type(data)}")
    print(f"Data Shape:{data.shape}")

    if isinstance(data, np.ndarray) and data.ndim > 1:
        print(f"Data Type:{data.dtype}")
        print(f"Number of Dimensions:{data.ndim}")

        print(f"Number of Elements:{data.size}")

def check_dict(input_dict):
    for key, value in input_dict.items():
        print(f"Key: {key}")

        value_type = type(value)
        print(f"  Value Type: {value_type}")

        if isinstance(value, np.ndarray):
            print(f"  Value Shape: {value.shape}")

        elif torch.is_tensor(value):
            print(f"  Value Shape: {value.shape}")
        print()



### Draw plots
def tensor_for_draw(x):
    if isinstance(x,torch.Tensor):
        return x.cpu().numpy()
    else:
        return x
def plotline(*args,x_change=0,figsize=(8.5,6),xname='x',yname='y',title='y~x',have_x=True,label=None,linewidth=1.5,overlap=0,xnum=9,ynum=0,xscale='linear',yscale='linear',
             stcksize=14,titlesz = 20,legendsz = 19):
    '''
    Draw y~x line or multiple y~x lines. Format:y/ y1,y2,....,x When x_change==1(which means the plot does not adopt union grid), 'x' is necessary.
    !!!This func only plot the fig. Write plt.show(), savefig, saving into wandb, and plt.clf() in the main code!
    Default fontsize is 14 for numbers and 20 for labels.
    When I want plot to (x,y1,y2,...), input label as (tag1,tag2,tag3..) could be int/float/str.
    !!!!Recommended choice of figsize for long plot is (12,4).
    Several remark lines near bottom: when I want to increase the number of y-ticks adopt it.

    !!! in the original setting, it is a seq of several inputs, not one list of this inputs
    :param args:y or y1,y2,,,x  tensor(should be vector, e.g.a[0]) or numpy, all fine!  Also fine if it is a 2D tensor/numpy with rows yi....,x.
                No need that the length are the same. Will plot all y_i starting from x[0] to whenevre it ends.
                x should be the last one of the input;
                If do not add x in input, should have have_x=False
    :param x_change: if ==1: the x grid are not uniform grids
    :param figsize:
    :param xname:  use r'$...$..' for LaTeX as name, e.g. xname=r'$\frac 1 {\int\iff e^x}$'
    :param yname:
    :param title:
    :param label: a list Format:input label as (tag1,tag2,tag3..) could be int/float/str.
    :param linewidth:
    :param overlap:If ==1: several lines overlap together, use -- lines with different period and phase.
    :param xnum: The number of marked grids in x-axis
    :param ynum: The number of marked grid in y-axis
    :have_x: whether x_plot is at [-1] if input or not
    :return:
    '''
    if type(args[0])==list:
        args=args[0]
    elif type(args[0])==torch.Tensor:
        if args[0].dim()>2:
            print('error_wcwPlot:input data(tensor) more than 2d')
            exit()
        elif args[0].dim()==2:
            args=args[0]
    elif type(args[0])==np.ndarray:
        if args[0].ndim>2:
            print('error_wcwPlot:input data(numpy) more than 2d')
            exit()
        elif args[0].ndim==2:
            args=args[0]

    data_draw=[tensor_for_draw(i) for i in args]
    n_draw=len(data_draw)
    lenmax=max([len(i)for i in data_draw])
    if label==None:
        label=[None for i in range(n_draw)]

    plt.figure(figsize=figsize)
    # plt.rc('text', usetex=True)
    '''Could not use LATEX in GPU!!!'''

    if x_change==0:
        if n_draw==1:

            plt.plot(data_draw[0],linestyle='-',label=label[0],linewidth=linewidth)
            plt.xticks(fontsize=stcksize)
            plt.yticks(fontsize=stcksize)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xlabel(xname,fontsize=titlesz)
            plt.ylabel(yname,fontsize=titlesz)
            plt.title(title,fontsize=titlesz)
            plt.legend(fontsize=legendsz)
            plt.tight_layout()
        else:
            if have_x==False:
                xx=np.arange(lenmax+2)
                data_draw.append(xx)
                n_draw+=1


            if overlap==0:
                for i in range(n_draw-1):
                    plt.plot(data_draw[-1][:len(data_draw[i])],data_draw[i],linestyle='-',label=label[i],linewidth=linewidth)
            else:
                for i in range(n_draw-1):
                    plt.plot(data_draw[-1][:len(data_draw[i])],data_draw[i],linestyle='-',label=label[i],linewidth=linewidth,dashes=(i+2,1+0.2*i))


            plt.xlabel(xname,fontsize=titlesz)
            plt.ylabel(yname,fontsize=titlesz)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xticks(fontsize=stcksize)
            plt.yticks(fontsize=stcksize)
            plt.title(title,fontsize=titlesz)
            plt.legend(fontsize=legendsz)
            plt.tight_layout()
    else:
        if n_draw==1:
            print('plot ERROR:should have x as input')
        else:
            xlen=[len(data_draw[i]) for i in range(0,n_draw-1)]
            maxlen=max(xlen)
            plotx=np.arange(maxlen)
            if overlap==0:
                for i in range(0,n_draw-1):
                    plt.plot(plotx[:xlen[i]],data_draw[i],linestyle='-',label=label[i],linewidth=linewidth)
            else:
                for i in range(0,n_draw-1):
                    plt.plot(plotx[:xlen[i]],data_draw[i],linestyle='-',label=label[i],linewidth=linewidth,dashes=(i+2,1+0.2*i))

            # if ynum:
            #     n_ticks = 8
            #     yticks_positions = np.arange(int(min(y)), int(max(y)) + 1, int((max(y) - min(y)) / (n_ticks - 1)))
            #     yticks_labels = [value for value in yticks_positions]

            mark_int=maxlen//(xnum-1)
            mark_x = plotx[::mark_int]
            # mark_c = c[:len(mark_x)] #i-th mark=c[i]
            mark_c=data_draw[-1][::mark_int] #c contains all the plotted x

            if isinstance(mark_c[0],np.int32)==0:

                formatted_mark_c = [f'{value:.3f}' for value in mark_c]

                # formatted_mark_c = [f'{value:.4g}' for value in mark_c] #4-digit sci-format
            else:
                formatted_mark_c=mark_c
            plt.xticks(mark_x, formatted_mark_c, fontsize=stcksize)

            plt.yticks(fontsize=stcksize)
            plt.xscale(xscale)
            plt.yscale(yscale)
            plt.xlabel(xname, fontsize=titlesz)
            plt.ylabel(yname, fontsize=titlesz)
            plt.title(title, fontsize=titlesz)
            plt.legend(fontsize=legendsz)
            plt.tight_layout()
def plotheat(x,y,z,figsize=None,xname='x',yname='y',barname='Function Value',title='f(x,y)',label=None,linewidth=1.5,overlap=0,xnum=9,ynum=0,
             vmin=None,vmax=None,sticksize=30,ftsz=30):
    '''Draw heatmap. !!!IMPORTANT: shape of z.!!!!!!shape: len(y)*len(x), use z.transpose() for np and z.transpose(dim0,dim1) a for tensor
            as input if z need a transpose.
            x will be the x-axis, and y:y-axis. (12,5), 12=l de x, 5=l de y.
    !!!This func only plot the fig. Write plt.show(), savefig, saving into wandb, and plt.clf() in the main code!
    Default fontsize is 14 for numbers and 20 for labels.
    When I want plot (x,y1,y2,...), input label as (tag1,tag2,tag3..) could be int/float/str.
    !!!!Standard choice of figsize for long plot is (6.4,4.8).
    figsize: (12,5) for long plots
    :param x: Tensor or numpy.ndarray, all fine
    :param y:
    :param z: !!!!!!shape: len(y)*len(x), use z.transpose() for np and z.transpose(dim0,dim1) a for tensor
            as input if z need a transpose.
    :param figsize:
    :param xname: use r'$...$..' for LaTeX as name, e.g. xname=r'$\frac 1 {\int\iff e^x}$'
                !!update: delete that line: NO latex on gpu

    :param yname:
    :param barname: name near colorbar
    :param title:
    :param label:
    legendsz: default=20
    titlesz: default=20
    sticksize: defualt=14
    :return:
    '''
    data_draw = [tensor_for_draw(i) for i in [x,y,z]]
    # n_draw = len(data_draw)
    # if label == None:
    #     label = [None for i in range(n_draw)]

    plt.figure(figsize=figsize)
    # plt.rc('text', usetex=True)

    heatmap = plt.pcolormesh(data_draw[0], data_draw[1], data_draw[2], cmap='viridis', shading='auto',vmin=vmin,vmax=vmax) #cmap=viridis: green=0; bwr: white=0



    # cbar=plt.colorbar(heatmap, label=barname,pad=0.01)
    cbar=plt.colorbar(heatmap,pad=0.02)
    cbar.ax.tick_params(labelsize=ftsz)

    plt.xlabel(xname, fontsize=ftsz)
    plt.ylabel(yname, fontsize=ftsz)
    plt.title(title, fontsize=ftsz)
    plt.xticks(fontsize=sticksize)
    plt.yticks(fontsize=sticksize)
    # plt.xlabel(xname, fontsize=20)
    # plt.ylabel(yname, fontsize=20)
    # plt.title(title, fontsize=20)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)

#tensor operation
def span_view(inputx, d, r):
    """
    Input x should be a vector (1d tensor)
    Reshape the input vector to a d-dimensional tensor with x's shape at the r-th(counting start from 0) dimension.
    Negative index is supported.
    """
    if r < 0:
        r += d  # Convert negative index to positive

    # Determine the number of dimensions to add before and after x's shape
    num_dims_before = r
    num_dims_after = d - r-1

    # Create a tuple of dimensions to unsqueeze
    dimensions_to_unsqueeze = (1,) * num_dims_before + inputx.shape + (1,) * num_dims_after

    # Unsqueeze the input tensor to create the desired shape
    reshaped_tensor = inputx.view(dimensions_to_unsqueeze)
    # reshaped_tensor = inputx.unsqueeze(dim=r).view(dimensions_to_unsqueeze)

    return reshaped_tensor

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


# function decorations

use_reentrant=False
class split_input_chkpt(nn.Module):
    def __init__(self,f,chkpt=0,splt=0,dim=2,ndim=None,close_grad_end=False):
        super().__init__()
        '''
        Parameters
        ----------
        f: original function
        chkpt: use gradient checkpoint or not
        splt: dim:
        split the 'dim' dimension (counting from 0)of input x into 'splt' parts.  default: btz, feature, x
        ndim: ndim of input tensor
        close_grad_end: whether or not set x.requires_grad=False at the end of self.forward
        '''
        self.f=f
        self.chkpt=chkpt
        self.splt=splt
        assert dim>=0
        self.dim=dim
        self.slice_index=[slice(None)]*(dim+1) # later: tupe(slice_index)
        self.close_end=close_grad_end

    def __call__(self, x, **kwargs):
        if not self.chkpt or self.splt==0:
            if self.splt<=1:
                return self.f(x,**kwargs)
            else:
                split_len = mt.ceil(x.shape[self.dim] / self.splt)
                split = x.shape[self.dim] // split_len
                remainder = int(x.shape[self.dim] - split * split_len)
                out = []

                for k in range(split):
                    self.slice_index[-1]=slice(k*split_len,(k+1)*split_len)
                    xk=x[self.slice_index]
                    # xk = x[:, :, k * split_len:(k + 1) * split_len]

                    out.append(self.f(xk,**kwargs))
                if remainder:
                    self.slice_index[-1]=slice(-remainder,None)
                    xk=x[self.slice_index]
                    out.append(self.f(xk,**kwargs))

                x = torch.cat(out, dim=self.dim)
                return x
        else:
            # x.requires_grad=True
            split_len = mt.ceil(x.shape[self.dim] / self.splt)
            split = x.shape[self.dim] // split_len
            remainder = int(x.shape[self.dim] - split * split_len)
            out = []

            for k in range(split):
                self.slice_index[-1] = slice(k * split_len, (k + 1) * split_len)
                xk=checkpoint(self.f,x[self.slice_index],use_reentrant=use_reentrant)
                '''ADD ,**kwargs ?'''
                # xk = x[self.slice_index]
                # xk = x[:, :, k * split_len:(k + 1) * split_len]

                out.append(xk)
            if remainder:
                self.slice_index[-1] = slice(-remainder, None)
                xk=checkpoint(self.f,x[self.slice_index],use_reentrant=use_reentrant)
                '''Add ,**kwargs?'''
                out.append(xk)

            x = torch.cat(out, dim=self.dim)
            if self.close_end:
                x.requires_grad=False
            return x

def str_to_dvc(x):
    if isinstance(x,torch.device):
        return x
    if isinstance(x,int):
        return f"cuda:{x}"
    elif x=='c':
        return 'cpu'
    else:
        assert isinstance(x,str)
        return x
class split_input_chkpt_advanced(nn.Module):
    def __init__(self,f,chkpt=0,splt=0,splt_bp=None,dim=2,dvc_cpt='cuda',dvc_chkpt='cpu',no_splt=False,fwd_dtype=None):
        super().__init__()
        '''
        Parameters
        ----------
        f: original function
        chkpt: use gradient checkpoint or not
        splt: dim:
        split the 'dim' dimension (counting from 0)of input x into 'splt' parts.  default tensor: btz, feature, x
        splt_bp: split in Backward path
        
        dvc: device for compute(cpt) and saving checkpoints(chkpt), 'cuda:i' 'cuda'  'cpu'. Supported: i, 'c'
        fwd_type: the dtype of output of forward. (Need to set torch.commplex64 when splitting fft!!!)
        '''
        self.f=f
        self.chkpt=chkpt
        self.splt=splt
        self.splt_bp=self.splt_bp if splt_bp is not None else self.splt
        assert dim>=0
        self.dim=dim
        self.slice_index=[slice(None)]*(dim+1) # later: tupe(slice_index)
        self.dvc_cpt=str_to_dvc(dvc_cpt)
        self.dvc_chkpt=str_to_dvc(dvc_chkpt)
        self.no_splt=no_splt
        self.type=fwd_dtype


    def __call__(self, x_in,test=False,**kwargs):
        x = x_in if x_in.device == self.dvc_cpt else x_in.to(self.dvc_cpt)
        '''this can be further optimized(to reduce peak cuda memory): 
                only load one slice to cuda a time; and the empty output is allocated at chkpt_dvc
                (but slow due to lots of transferring data)'''
        dtype_out=self.type if self.type is not None else x.dtype
        if not self.chkpt or self.splt==0 or test:
            "no checkpoint  in this branch!"
            if self.splt<=1 or self.no_splt:
                # yet: data_device_transfer
                return self.f(x,**kwargs)
            else:
                split_len = mt.ceil(x.shape[self.dim] / self.splt)
                split = x.shape[self.dim] // split_len
                remainder = int(x.shape[self.dim] - split * split_len)

                out=0
                for k in range(split):
                    self.slice_index[-1]=slice(k*split_len,(k+1)*split_len)
                    # x[self.slice_index]=self.f(x[self.slice_index],**kwargs)
                    # xk = x[self.slice_index]
                    if k==0:
                        yk=self.f(x[self.slice_index],**kwargs)
                        shp=list(yk.shape)
                        shp[self.dim]=x.shape[self.dim]
                        out=torch.empty(shp,device=yk.device,dtype=dtype_out)
                        out[self.slice_index]=yk
                        del yk
                    else:
                        out[self.slice_index]=self.f(x[self.slice_index],**kwargs)
                if remainder:
                    self.slice_index[-1]=slice(-remainder,None)
                    # x[self.slice_index]=self.f(x[self.slice_index],**kwargs)
                    # out.append(self.f(xk,**kwargs))
                    out[self.slice_index] = self.f(x[self.slice_index],**kwargs)
                return out
        else:
            split_len = mt.ceil(x.shape[self.dim] / self.splt)
            split = x.shape[self.dim] // split_len
            remainder = int(x.shape[self.dim] - split * split_len)
            out = 0
            # mmm('in')
            with torch.no_grad():
                for k in range(split):
                    self.slice_index[-1] = slice(k * split_len, (k + 1) * split_len)
                    # x[self.slice_index]=self.f(x[self.slice_index],**kwargs)
                    # xk = x[self.slice_index]
                    if k == 0:
                        yk = self.f(x[self.slice_index],**kwargs)
                        shp = list(yk.shape)
                        shp[self.dim] = x.shape[self.dim]
                        out = torch.empty(shp, device=yk.device,dtype=dtype_out)
                        out[self.slice_index] = yk[:]
                        # sss(yk)
                        # mmm('k=0')
                        del yk
                        # mmm('k=0,delyk')
                    else:
                        out[self.slice_index] = self.f(x[self.slice_index],**kwargs)[:]
                        # mmm(f'iter={k}')
                if remainder:
                    self.slice_index[-1] = slice(-remainder, None)
                    # x[self.slice_index]=self.f(x[self.slice_index],**kwargs)
                    # out.append(self.f(xk,**kwargs))
                    out[self.slice_index] = self.f(x[self.slice_index],**kwargs)
                if out.device != self.dvc_chkpt:
                    out.contiguous()
                    out=out.to(self.dvc_chkpt)
                # mmm('before return')
                return out

    def bp(self,x,y_grad=None,keep_x_grad=True,dvc_grad_x=None,**kwargs):
        '''
        Parameters
        ----------
        x: input of f
        y_grad: grad of loss w.r.t. y(=f(x)).   y_grad is move to and back device_compute per slice
        keep_x_grad: keep/return grad_x or not
        dvc_grad_x: device to save grad_x. This tensor will be created there. By default x.device
        Returns
        -------

        '''
        """Didn't include del y,del y_grad, SHould be added outside the function"""
        '''y_grad may not be on cuda；
        By default save grad_x on x.device'''

        if not self.chkpt or self.splt_bp == 0:
            return
        grad_dvc=dvc_grad_x if dvc_grad_x is not None else x.device
        split_len = mt.ceil(x.shape[self.dim] / self.splt_bp)
        split = x.shape[self.dim] // split_len
        remainder = int(x.shape[self.dim] - split * split_len)
        if keep_x_grad:
            x_grad=torch.empty(x.shape,device=grad_dvc)
            for k in range(split):
                self.slice_index[-1] = slice(k * split_len, (k + 1) * split_len)
                xx=x[self.slice_index]
                xx.requires_grad=True
                yy=self.f(xx.to(self.dvc_cpt),**kwargs)
                yy.backward(y_grad[self.slice_index].to(self.dvc_cpt))
                xx.requires_grad=False
                x_grad[self.slice_index]=xx.grad.contiguous().to(x.device)
                del yy
            if remainder:
                self.slice_index[-1] = slice(-remainder, None)
                xx = x[self.slice_index]
                xx.requires_grad = True
                yy = self.f(xx.to(self.dvc_cpt),**kwargs)
                yy.backward(y_grad[self.slice_index].to(self.dvc_cpt))
                xx.requires_grad = False
                x_grad[self.slice_index] = xx.grad.contiguous().to(x.device)
                del yy
            return x_grad
        else:
            for k in range(split):
                self.slice_index[-1] = slice(k * split_len, (k + 1) * split_len)
                xx=x[self.slice_index]
                yy=self.f(xx.to(self.dvc_cpt),**kwargs)
                yy.backward(y_grad[self.slice_index].to(self.dvc_cpt))
                del yy
            if remainder:
                self.slice_index[-1] = slice(-remainder, None)
                xx = x[self.slice_index]
                yy = self.f(xx.to(self.dvc_cpt),**kwargs)
                yy.backward(y_grad[self.slice_index].to(self.dvc_cpt))
                del yy
            return 0
