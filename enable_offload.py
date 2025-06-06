import math

from ..tool.my_tool import *

import torch

from typing import List, Optional, Tuple, Union

def enable_activation_offload_for_FNO(FNO):
    """Enable activation offloading for the FNO module.
    This function modifies the FNO class to support activation offloading.
    """
    FNO.check_mem=True
    FNO.chkpt=True

    FNO.ly_dvc='cuda:0'
    FNO.ly_dvc=['cuda:0' for x in FNO.ly_dvc]
    FNO.blk_dvc='cuda:0'
    FNO.blk_dvc_y3='cuda:0'
    FNO.blk_dvc_y0='cuda:0'
    FNO.save_ls=[] # intermediates after lifting, block,1,2,3,4, proj (i.e. output)

    FNO.blk_dvc = ['cuda:0' for x in FNO.blk_dvc]
    FNO.blk_dvc_y3 = ['cuda:0' for x in FNO.blk_dvc_y3]
    FNO.blk_dvc_y0 = ['cuda:0' for x in FNO.blk_dvc_y0]

    FNO.lifting=sinc(FNO.lifting,chkpt=1,splt=FNO.ly_splt[0],dim=2,dvc_chkpt=FNO.ly_dvc[0],dvc_cpt=FNO.dvc_cpt,no_splt=FNO.no_splt)
    FNO.projection=sinc(FNO.projection,chkpt=1,splt=FNO.ly_splt[1],dim=2,dvc_chkpt=FNO.ly_dvc[1],dvc_cpt=FNO.dvc_cpt,no_splt=FNO.no_splt)

    enable_activation_offload_for_FNOBlocks(FNO.FNOBlocks)

    FNO.forward = customized_forward

    def customized_forward(self, x, output_shape=None, test=False,**kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        temp=mmm('FNO model start') if self.check_mem else None
        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        # if not self.grad_chkpt[0]: 2407
        if not self.chkpt or test:
            x = self.lifting(x,test=test)
            temp = mmm('after lifting layer') if self.check_mem else None

            if self.domain_padding is not None:
                x = self.domain_padding.pad(x)
            for layer_idx in range(self.n_layers):
                x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx],test=test)
                temp = mmm(f"layer_id={layer_idx}") if self.check_mem else None

            if self.domain_padding is not None:
                x = self.domain_padding.unpad(x)

            x = self.projection(x,test=test)
            temp = mmm('fno after proj') if self.check_mem else None
            return x
        else:
            with torch.no_grad():
                x = self.lifting(x)
                self.save_ls.append(x.to(self.ly_dvc[0]))
                temp = mmm('after lifting layer') if self.check_mem else None

                if self.domain_padding is not None:
                    x = self.domain_padding.pad(x)
                temp = mmm('after dom pad') if self.check_mem else None
                for layer_idx in range(self.n_layers):
                    x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])
                    temp = mmm(f"layer_id={layer_idx}") if self.check_mem else None
                    if layer_idx<(self.n_layers-1):
                        self.save_ls.append(x.to(self.blk_dvc[layer_idx]))


                if self.domain_padding is not None:
                    x = self.domain_padding.unpad(x)
                self.save_ls.append(x.to(self.blk_dvc[self.n_layers-1]))

                x = self.projection(x)
                temp = mmm('fno after proj') if self.check_mem else None

            """c is on dvc_cpt (cuda:0)"""
            self.save_ls.append(x.to(self.ly_dvc[-1]))

            self.save_ls[-1].requires_grad=True



            return self.save_ls[-1]



def enable_activation_offload_for_FNOBlocks(FNOBlocks):
    """Enable activation offloading for the FNOBlocks module.
    This function modifies the FNOBlocks class to support activation offloading.
    """
    FNOBlocks.chkpt = True
    FNOBlocks.dvc_cpt = 'cuda:0'

    FNOBlocks.blk_dvc = [1]*FNOBlocks.n_layers
    FNOBlocks.blk_dvc = ['cuda:0' for x in FNOBlocks.blk_dvc]
    FNOBlocks.blk_dvc_y3 = [1]*FNOBlocks.n_layers
    FNOBlocks.blk_dvc_y3 = ['cuda:0' for x in FNOBlocks.blk_dvc]
    FNOBlocks.blk_dvc_y0 = [1]*FNOBlocks.n_layers
    FNOBlocks.blk_dvc_y0 = ['cuda:0' for x in FNOBlocks.blk_dvc]

    def customized_forward_with_postactivation(self, x, index=0, output_shape=None,test=False):
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

    FNOBlocks.forward_with_postactivation = customized_forward_with_postactivation

    enable_activation_offload_for_FNOBlocks(FNOBlocks.convs)

def enable_activation_offload_for_SpectralConv(SpectralConv):
    """
    Enable activation offloading for the SpectralConv module.
    This function modifies the SpectralConv class to support activation offloading.
    """
    SpectralConv.chkpt=True
    SpectralConv.dvc_cpt='cuda:0'
    SpectralConv.dvc_chkpt=[SpectralConv.dvc_cpt]*SpectralConv.n_layers
    SpectralConv.splt=[True]*SpectralConv.n_layers
    SpectralConv.save_ls=[[]for i in range(SpectralConv.n_layers)]

    SpectralConv.split_len=[math.ceil(SpectralConv.in_channels/i)if i else SpectralConv.in_channels for i in SpectralConv.splt]
    SpectralConv.split=[SpectralConv.in_channels//i for i in SpectralConv.split_len]
    SpectralConv.remainder=[int(SpectralConv.in_channels-SpectralConv.split[i]*SpectralConv.split_len[i])for i in range(len(SpectralConv.split_len))]

    SpectralConv.split_len_out=[mt.ceil(SpectralConv.out_channels/i)if i else SpectralConv.out_channels for i in SpectralConv.splt]
    SpectralConv.split_out=[SpectralConv.out_channels//i for i in SpectralConv.split_len_out]
    SpectralConv.remainder_out=[int(SpectralConv.out_channels-SpectralConv.split_out[i]*SpectralConv.split_len_out[i])for i in range(len(SpectralConv.split_len_out))]

    SpectralConv.dim=1
    SpectralConv.slice_index = [slice(None)] * (SpectralConv.dim + 1)  # later: tupe(slice_index); split the channel dim

    SpectralConv.bp_fft_slice=[slice(None,i)for i in SpectralConv.n_modes]# [:8,:16,:9]
    SpectralConv.full_fft_slice=[slice(None)]*2+SpectralConv.bp_fft_slice #[:,:,:8,:16,:9]

    SpectralConv.forward = customized_forward

    def customized_forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        x_shape_last_dim=fft_size[-1]
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))
        if self.fno_block_precision == "half":
            x = x.half()
        '''step1: fft'''
        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft=torch.empty([batchsize,self.in_channels,*self.n_modes],device=x.device,dtype=out_dtype)

        weight = self._get_weight(indices)

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(fft_size, list(weight.shape[2:]))]

        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [
            slice(None, -starts[-1]) if starts[-1] else slice(None)]  # The last mode already has redundant half removed
        # print(slices_x)
        pad_mode=[0,starts[-1]]
        for i in range(len(starts)-2,-1,-1):
            pad_mode+=[starts[i]//2,starts[i]//2]

        for k in range(self.split[indices]):
            self.slice_index[-1] = slice(k * self.split_len[indices], (k + 1) * self.split_len[indices])
            yk=torch.fft.rfftn(x[self.slice_index],norm=self.fft_norm,dim=fft_dims)
            if self.order>1:
                yk=torch.fft.fftshift(yk,dim=fft_dims[:-1])
            if self.fno_block_precision=='mixed':
                yk=yk.half()
            out_fft[self.slice_index]=yk[slices_x]
            del yk
        if self.remainder[indices]:
            self.slice_index[-1] = slice(-self.remainder[indices],None)
            yk=torch.fft.rfftn(x[self.slice_index],norm=self.fft_norm,dim=fft_dims)
            if self.order>1:
                yk=torch.fft.fftshift(yk,dim=fft_dims[:-1])
            if self.fno_block_precision=='mixed':
                yk=yk.half()
            out_fft[self.slice_index]=yk[slices_x]
            del yk
        del x
        if self.chkpt and (not test):
            self.save_ls[indices].append(out_fft.to(self.dvc_chkpt[indices]))
        temp = mmm('before linear') if self.check_mem and not indices else 0
        '''step2: linear'''
        out_fft=self._contract(out_fft, weight, separable=False)  # multiply_chk
        temp = mmm('after linear') if self.check_mem and not indices else 0
        '''no change'''
        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])])
        if output_shape is not None:
            mode_sizes = output_shape
        # if self.order > 1:
        #     out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
        '''step3: IFFT'''
        fft_size[-1]=x_shape_last_dim
        out=torch.empty([batchsize,self.out_channels,*fft_size],device=self.dvc_cpt)

        for k in range(self.split_out[indices]):
            self.slice_index[-1] = slice(k * self.split_len_out[indices], (k + 1) * self.split_len_out[indices])
            #
            xk=out_fft[self.slice_index]
            xk=fft.fftshift(pad(xk,pad=pad_mode),dim=fft_dims[:-1])
            yk=fft.irfftn(xk,s=mode_sizes,dim=fft_dims,norm=self.fft_norm)
            if self.bias is not None:
                yk+=self.bias[indices,self.slice_index[-1],...]
            out[self.slice_index]=yk
            del yk,xk

        if self.remainder_out[indices]:
            self.slice_index[-1] = slice(-self.remainder_out[indices],None)
            xk = out_fft[self.slice_index]
            xk = fft.fftshift(pad(xk, pad=pad_mode), dim=fft_dims[:-1])
            yk = fft.irfftn(xk, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)
            if self.bias is not None:
                yk += self.bias[indices, self.slice_index[-1], ...]
            out[self.slice_index] = yk
            del yk, xk

        temp = mmm('before return') if self.check_mem and not indices else 0
        return out
