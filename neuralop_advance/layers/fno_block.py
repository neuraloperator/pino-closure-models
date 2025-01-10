from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP
from .normalization_layers import AdaIN
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor
from ..tool.my_tool import split_input_chkpt_advanced as sinc
from ..tool.my_tool import str_to_dvc,mmm
from functools import partial
Number = Union[int, float]
# from line_profiler import LineProfiler
"New"
class sinc4fft(nn.Module):
    '''
    For peak memory reduction. Split the channel dimension in fft and ifft.
    !! Not applied in current version.

    Variable names:
    bp=back propagation
    dvc=device
    cpt=compute (device that handle computaions, e.g. tensor operations)
    chkpt=checkpoint (device that save the checkpoints)
    '''
    def __init__(self,f_multi,num_func,param_name:str,chkpt=0,splt_ls=0,splt_bp_ls=None,dim=1,dvc_cpt='cuda',dvc_chkpt='cpu',no_splt=False):
        super().__init__()
        self.f=f_multi
        self.n=num_func
        self.key_nm=param_name
        self.chkpt = chkpt
        self.splt = splt_ls
        self.splt_bp = splt_bp_ls if splt_bp_ls is not None else [None]*self.n
        self.dim = dim
        self.no_splt=no_splt

        self.ff=[partial(self.f,**{self.key_nm:i})for i in range(self.n)]

        self.ff=[sinc(self.ff[i],chkpt=chkpt,splt=self.splt[i],splt_bp=self.splt_bp[i],dim=self.dim,dvc_cpt=dvc_cpt,dvc_chkpt=dvc_chkpt,no_splt=self.no_splt)for i in range(self.n)]

    def forward(self, x: torch.Tensor, indices=0,test=False,**kwargs):
        return self.ff[indices](x,test=test,**kwargs)
    def bp(self,x,indices=0,y_grad=None,keep_x_grad=True,dvc_grad_x=None,**kwargs):
        return self.ff[indices].bp(x,y_grad,keep_x_grad,dvc_grad_x,**kwargs)

class composite(nn.Module):
    def __init__(self,nonl,mlp):
        super().__init__()
        self.nonl=nonl
        self.mlp=mlp
    def forward(self,x,**kwargs):
        return self.mlp(self.nonl(x),**kwargs)
class FNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        check_mem=False,
        check_mem_bp=False,
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        'New'
        '''For peak memory reduction.
        The original FNO corresponds to setting chkpt=False and no_splt=True
        chkpt: whether or not use checkpointing to reduce peak mem.
        
        [Variable Names]
        blk=block (fno blcok)
        dvc=device ('c' for cpu and int:i for GPU cuda:i)
        y3 and y0: intermediate tensors during the computation.
        
        check_mem: if True, print out memory consumption
        '''
        self.chkpt = kwargs['save_mem'] if 'save_mem' in kwargs else 0
        self.no_splt=kwargs['no_splt'] if 'no_splt' in kwargs else 0
        self.dvc_cpt=kwargs['dvc_cpt'] if 'dvc_cpt' in kwargs else 0
        self.dvc_cpt=str_to_dvc(self.dvc_cpt)
        self.blk_splt_fft = kwargs['blk_splt_fft']if self.chkpt else [0]*n_layers  # Split channel dim into blk_splt_fft times with for loop in FFT
        self.blk_splt_mlp = kwargs['blk_splt_mlp']if self.chkpt else [0]*n_layers # Split X,Y,Z,... dim into blk_splt_mlp times with for loop in mlp

        self.blk_dvc = kwargs['blk_dvc']if self.chkpt else [0]*n_layers
        self.blk_dvc=[str_to_dvc(x)for x in self.blk_dvc]
        self.blk_dvc_y3 = kwargs['blk_dvc_y3']if self.chkpt else [0]*n_layers
        self.blk_dvc_y3 = [str_to_dvc(x) for x in self.blk_dvc_y3]
        self.blk_dvc_y0 = kwargs['blk_dvc_y0']if self.chkpt else [0]*n_layers
        self.blk_dvc_y0 = [str_to_dvc(x) for x in self.blk_dvc_y0]
        self.save_ls = [[]for i in range(self.n_layers)]  # intermediates after lifting, block,1,2,3,4, proj (i.e. output)
        self.check_mem=False
        self.check_mem_bp=False#check_mem_bp

        # self.check_mem=check_mem

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
            #new
            chkpt=self.chkpt,
            no_splt=self.no_splt,
            dvc_cpt=self.dvc_cpt,
            splt=self.blk_splt_fft,
            check_mem=self.check_mem,
            check_mem_bp=self.check_mem_bp,
        )
        # convs_s is not applied in the newest (current) version.
        self.convs_s = sinc4fft(self.convs, self.n_layers, 'indices',
                                chkpt=self.chkpt,splt_ls=self.blk_splt_fft,splt_bp_ls=None,dim=1,
                                dvc_cpt=self.dvc_cpt,dvc_chkpt=self.dvc_cpt,no_splt=self.no_splt) #if self.chkpt \
            # else sinc4fft(self.convs,self.n_layers,'indices',0,dvc_cpt=self.dvc_cpt,no_splt=self.no_splt)

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )
        'New'
        # if self.chkpt:
        self.fno_skips = nn.ModuleList([sinc(self.fno_skips[i], chkpt=self.chkpt, splt=self.blk_splt_mlp[i], dim=2,
                                             dvc_cpt=self.dvc_cpt,dvc_chkpt=self.dvc_cpt) for i in
                                       range(n_layers)])  # split in time dimension if it is btz,feature,t,x,y; no need to save output/checkpointing here

        if use_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout=mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp=nn.ModuleList([composite(nonl=self.non_linearity,mlp=self.mlp[i])for i in range(n_layers)])
            self.mlp=nn.ModuleList([sinc(self.mlp[i],chkpt=self.chkpt,
                                         splt=self.blk_splt_mlp[i],dim=2,dvc_cpt=self.dvc_cpt,dvc_chkpt=self.dvc_cpt,
                                         no_splt=self.no_splt)for i in range(n_layers)])
            self.mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            # if self.chkpt:
            self.mlp_skips = nn.ModuleList(
                [sinc(self.mlp_skips[i], chkpt=self.chkpt, splt=self.blk_splt_mlp[i], dim=2,
                      dvc_cpt=self.dvc_cpt,dvc_chkpt=self.dvc_cpt,no_splt=self.no_splt) for i in
                 range(n_layers)])
            # else:
            #     self.mlp_skips=nn.ModuleList([sinc(self.mlp_skips[i],0,no_splt=self.no_splt,dvc_cpt=self.dvc_cpt)for i in range(n_layers)])
        else:
            self.mlp = None


        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [
                    getattr(nn, f"InstanceNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None,test=False):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape,test=test)
    def bp(self,x,y_grad,index=0,output_shape=None):
        if self.preactivation:
            print('Not implemented bp for preactivation yet!')
            exit()
        else:

            return self.backward_with_postactivation(x,y_grad,index,output_shape)



    def forward_with_postactivation(self, x, index=0, output_shape=None,test=False):
        if not self.chkpt or test:
            x_skip_fno = self.fno_skips[index](x,test=test)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape) #unchanged

            if self.mlp is not None:
                x_skip_mlp = self.mlp_skips[index](x,test=test)
                x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)# unchanged


            if self.stabilizer == "tanh":# not this way
                x = torch.tanh(x)


            x_fno = self.convs(x, index, output_shape=output_shape,test=test)#old:convs_s

            if self.norm is not None:
                x_fno = self.norm[self.n_norms * index](x_fno)

            x = x_fno + x_skip_fno

            if (self.mlp is not None) or (index < (self.n_layers - 1)):
                x = self.non_linearity(x)

            if self.mlp is not None:
                x = self.mlp[index](x) + x_skip_mlp

                if self.norm is not None:
                    x = self.norm[self.n_norms * index + 1](x)

                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)

            return x
        with torch.no_grad():
            temp=mmm('in') if self.check_mem and not index else 0
            x_skip_fno = self.fno_skips[index](x)
            temp=mmm('skip_fno') if self.check_mem and not index else 0
            # """before norm., this checkpoint might be redundant"""
            # self.save_ls[index].append(x_skip_fno.to(self.blk_dvc_y3))###
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)#
            temp = mmm('skip_fno_trsfm') if self.check_mem and not index else 0
            x_fno = self.convs(x, index, output_shape=output_shape,test=0,dbg=0) if self.stabilizer!='tanh' else \
                self.convs(torch.tanh(x), index, output_shape=output_shape) #old: convs_s #dbg
            temp = mmm('x_fno') if self.check_mem and not index else 0
            if self.norm is not None:
                """before norm., this checkpoint might be redundant"""
                self.save_ls[index].append(x_fno.to(self.blk_dvc_y3[index]))###

                x_fno = self.norm[self.n_norms * index](x_fno)#
            temp = mmm('x_fno norm') if self.check_mem and not index else 0
            x_fno+=x_skip_fno
            del x_skip_fno
            temp = mmm('del skip_fno') if self.check_mem and not index else 0
            self.save_ls[index].append(x_fno.to(self.blk_dvc_y3[index]))
            #
            if self.mlp is not None:
                x_skip_mlp = self.mlp_skips[index](x)
                x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)
                temp = mmm('skip_mlp') if self.check_mem and not index else 0
            x=x_fno
            del x_fno
            temp = mmm('x=x_fno') if self.check_mem and not index else 0

            if (self.mlp is None) and (index < (self.n_layers - 1)):
                x = self.non_linearity(x)
                # temp = mmm('after nonlinear') if self.check_mem and not index else 0
            elif self.mlp is not None:
                x_skip_mlp+=self.mlp[index](x)#y1+y2 #0728: nonL is merged into self.mlp
                self.save_ls[index].append(x_skip_mlp.to(self.blk_dvc_y0[index]))# save y0
                temp = mmm('get y1;and y1+y2') if self.check_mem and not index else 0
                x=x_skip_mlp
                # x = self.mlp[index](x) + x_skip_mlp
                if self.norm is not None:
                    x = self.norm[self.n_norms * index + 1](x)
                if index < (self.n_layers - 1):
                    x = self.non_linearity(x)
                temp = mmm('before return') if self.check_mem and not index else 0

            return x

    def backward_with_postactivation(self,x,y_grad,index=0,output_shape=None):
        """Return grad_x; caution where to save grad_x
        Ideally, x is on cpu(dvc_chkpt); conduct in_place operations on the input y_grad (eventually becomes x_grad in this function)
        the grads are on dvc_cpt
        """

        x_grad=y_grad #grad_temp in fno.py, will conduct in-place operations on grad_outside. After bp, grad_outside is grad_x
        temp=mmm('bp:IN') if self.check_mem_bp and not(index-3) else 0
        if self.mlp is not None:
            y0=self.save_ls[index][-1].to(self.dvc_cpt)#g
            y0.requires_grad=True
            original_y0=y0

            if self.norm is not None:
                y0=self.norm[self.n_norms*index+1](y0)

            if index<(self.n_layers-1):
                y0=self.non_linearity(y0)

            y0.backward(y_grad.to(self.dvc_cpt))# Ideally, y_grad is on cuda:0

            original_y0.requires_grad=False
            del y0, self.save_ls[index][-1]#y0  This del step is needed

            y_grad=original_y0.grad.detach().clone()#on g. to cpu?
            del original_y0
        temp = mmm('get y0 grad') if self.check_mem_bp and not (index - 3) else 0



        if self.mlp is not None:
            if output_shape is not None or self.output_scaling_factor is not None:
                print('chkpt_BP not yet implemented for output scalings/output_shape!!')
                exit()
            x_grad[:]=self.mlp_skips[index].bp(x,y_grad,dvc_grad_x=self.dvc_cpt)#0728
            # x_grad=self.mlp_skips[index].bp(x,y_grad,dvc_grad_x=self.dvc_cpt)###0725 cahnge. err: folowing code x_grad+=A, A is on cpu
        temp = mmm('y2(x) grad:ok') if self.check_mem_bp and not (index - 3) else 0

        grad_y3=self.mlp[index].bp(self.save_ls[index][-1],y_grad,dvc_grad_x=self.dvc_cpt) # check where to save
        # grad_y3 = self.mlp[index].bp(self.save_ls[index][-1], y_grad).clone()
        del self.save_ls[index][-1],y_grad


        temp = mmm('del after get y3 grad') if self.check_mem_bp and not (index - 3) else 0
        """period end"""

        if output_shape is not None or self.output_scaling_factor is not None:
            print('chkpt_BP not yet implemented for output scalings/output_shape(x_skip_fno)!!')
            # exit()
        x_grad+=self.fno_skips[index].bp(x,grad_y3,dvc_grad_x=self.dvc_cpt)
        temp = mmm('y32 grad(fno skip):ok') if self.check_mem_bp and not (index - 3) else 0

        """BP through the norm. layer after FFT and before y3"""
        if self.norm is not None:
            y31=self.save_ls[index][-1].to(self.dvc_cpt)
            y31.requires_grad=True
            y3= self.norm[self.n_norms * index](y31)
            y3.backward(grad_y3.to(self.dvc_cpt))
            y31.requires_grad=False
            del y3,self.save_ls[index][-1],grad_y3#y31
            # grad_y3_=y31.grad.detach().to(self.blk_dvc_y3[index]).clone()##previously: var name=grad_y3, wrong grad. WHY?
            grad_y3_=y31.grad.detach().clone()##previously: var name=grad_y3, wrong grad. WHY?
            del y31
        """end"""
        temp = mmm('bp: norm between y31 and y3:ok') if self.check_mem_bp and not (index - 3) else 0

        '''case with tanh not implemented'''

        self.convs.bp(x,grad_y3_,index,output_shape=output_shape)#.to(x_grad.device)
        x_grad +=grad_y3_
        temp = mmm('bp fft:ok') if self.check_mem_bp and not (index - 3) else 0




        del grad_y3_


        return 0#x_grad




    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
