import numpy as np
import torch
import scipy.io

from pathlib import Path
import warnings 
import torch.nn as nn
import warnings


'''
legacy class
class FC2D(object):
    # todo: add docstring, and use python best practices

    def __init__(self, device, d=5, C=25):
    
        self.device = device
        self.d = d
        self.C = C
        self.A = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath( \
              "fc_data/A_d" + str(d) + "_C" + str(C) + ".mat"))['A']).double()
        self.Q = torch.from_numpy(scipy.io.loadmat(Path(__file__).resolve().parent.joinpath( \
              "fc_data/Q_d" + str(d) + "_C" + str(C) + ".mat"))['Q']).double()
        # if device == 'cuda':
        self.A = self.A.cuda()
        self.Q = self.Q.cuda()

    def diff_x(self, u, domain_length_x = 1):

        nx = u.size(2)
        hx = domain_length_x / (nx - 1)
        fourPtsx = nx + self.C
        prdx = fourPtsx * hx
        u = u.double()

        if fourPtsx % 2 == 0:
                k_max = int(fourPtsx / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0)
        else:
                k_max = int((fourPtsx - 1) / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0)
        der_coeffsx = 1j * 2.0 * np.pi / prdx * k_x

        # compute derivatives along the x-direction
        # First produce the periodic extension
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -self.d:], self.Q), self.A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :self.d], dims=(2,)), self.Q), self.A), dims=(2,))
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real	
        ux = uder[:, :, :nx].float()

        return ux

    def diff_xx(self, u, domain_length_x = 1):

        nx = u.size(2)
        hx = domain_length_x / (nx - 1)
        fourPtsx = nx + self.C
        prdx = fourPtsx * hx
        u = u.double()

        if fourPtsx % 2 == 0:
                k_max = int(fourPtsx / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0)
        else:
                k_max = int((fourPtsx - 1) / 2)
                k_x = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0)
        der_coeffsx = - 4.0 * np.pi * np.pi / prdx / prdx * k_x * k_x

        # compute derivatives along the x-direction
        # First produce the periodic extension
        y1 = torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", u[:, :, -self.d:], self.Q), self.A)
        y2 = torch.flip(torch.einsum("hik,jk->hij", torch.einsum("hik,kj->hij", torch.flip(u[:, :, :self.d], dims=(2,)), self.Q), self.A), dims=(2,))
        # Compute the derivative of the extension
        ucont = torch.cat([u,y1+y2], dim=2)
        uhat = torch.fft.fft(ucont, dim=2)
        uder = torch.fft.ifft(uhat * der_coeffsx).real
        # Restrict to the original interval
        ux = uder[:, :, :nx].float()

        return ux        

    def diff_y(self, u, domain_length_y = 1):

        ny = u.size(1)
        nx = u.size(2)
        hy = domain_length_y / (ny - 1)
        fourPtsy = ny + self.C
        prdy = fourPtsy * hy
        u = u.double()

        if fourPtsy % 2 == 0:
                k_max = int(fourPtsy / 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max + 1, end = 0, step = 1, device = self.device)), 0).reshape(fourPtsy, 1).repeat(1, nx)
        else:
                k_max = int((fourPtsy - 1) / 2)
                k_y = torch.cat((torch.arange(start = 0, end = k_max + 1, step = 1, device = self.device),
                    torch.arange(start = - k_max, end = 0, step = 1, device = self.device)), 0).reshape(fourPtsy, 1).repeat(1, nx)	         

        der_coeffsy = 1j * 2.0 * np.pi / prdy * k_y

        # compute derivatives along the y-direction
        # First produce the periodic extension
        y1 = torch.einsum("ikl,jk->ijl", torch.einsum("ikl,kj->ijl", u[:, -self.d:, :], self.Q), self.A)
        y2 = torch.flip(torch.einsum("ikl,jk->ijl",torch.einsum("ikl,kj->ijl",torch.flip(u[:, :self.d, :], dims=(1,)), self.Q), self.A), dims=(1,))
        # Compute the derivative of the extension
        ucont = torch.cat([u,y1+y2], dim=1)
        uhat = torch.fft.fft(ucont, dim=1)
        uder = torch.fft.ifft(uhat * der_coeffsy, dim=1).real
        # Restrict to the original interval
        uy = uder[:, :ny, :].float()

        return uy
'''
class FCGram(nn.Module):
    def __init__(self, d=5, n_additional_pts=50, matrices_path=None):
        super().__init__()
        
        self.d=d
        self.n_additional_pts=n_additional_pts 
        
        if self.n_additional_pts%2==1:
            warnings.warn("n_additional_pts must be even, rounding down.", UserWarning)
            self.n_additional_pts -= 1
        self.C = int(self.n_additional_pts//2)
        
        if matrices_path is None:
            self.matrices_path = Path("~/pino-closure-models/neuraloperator/neuraloperator/neuralop/layers/FCGram_matrices")
        else:
            self.matrices_path = Path(matrices_path)
        
        self.load_matrices()

    
    def load_matrices(self):
        filepath = '/global/homes/y/ypincha/pino-closure-models/ks/fcgram_matrices/FCGram_data_d5_c25.npz'
        
        # if not filepath.exists():
        #     raise FileNotFoundError(
        #         f"FCGram matrices not found at {filepath}. \n"
        #         f"Please ensure the .npz file exists with d={self.d}, C={self.C}."
        #     )
        
        npz_data = np.load(str(filepath))
        
        self.register_buffer('ArQr', torch.from_numpy(npz_data['ArQr']))
        self.register_buffer('AlQl', torch.from_numpy(npz_data['AlQl']))
        
    
    def extend_left_right(self, x):
        # extract boundary values for continuation, use d points from each boundary
        left_bnd = x[..., :self.d]      
        right_bnd = x[..., -self.d:]    
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        # apply fc-gram continuation using ArQr matrix
        if x.is_complex():
            left_continuation = torch.matmul(left_bnd, AlQl.T + 0j)
            right_continuation = torch.matmul(right_bnd, ArQr.T + 0j)
        else:
            left_continuation = torch.matmul(left_bnd, AlQl.T)
            right_continuation = torch.matmul(right_bnd, ArQr.T)
        
        return torch.cat((left_continuation, x, right_continuation), dim=-1)
    
    def extend_top_bottom(self, x):
        top_bnd = x[..., :self.d, :]     
        bottom_bnd = x[..., -self.d:, :] 
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        if x.is_complex():
            bottom_continuation = torch.matmul(ArQr, bottom_bnd + 0j)
            top_continuation = torch.matmul(AlQl, top_bnd + 0j)
        else:
            bottom_continuation = torch.matmul(ArQr, bottom_bnd)
            top_continuation = torch.matmul(AlQl, top_bnd)
        
        return torch.cat((top_continuation, x, bottom_continuation), dim=-2)
    
    
    def extend_front_back(self, x):
        front_bnd = x[..., :self.d, :, :]     
        back_bnd = x[..., -self.d:, :, :] 
        
        AlQl = self.AlQl.to(dtype=x.dtype)
        ArQr = self.ArQr.to(dtype=x.dtype)
        
        y_shape = x.shape
        front_bnd_reshaped = front_bnd.reshape(*y_shape[:-3], self.d, -1)
        back_bnd_reshaped = back_bnd.reshape(*y_shape[:-3], self.d, -1)
        
        if x.is_complex():
            front_continuation_reshaped = torch.matmul(AlQl, front_bnd_reshaped + 0j)
            back_continuation_reshaped = torch.matmul(ArQr, back_bnd_reshaped + 0j)
        else:
            front_continuation_reshaped = torch.matmul(AlQl, front_bnd_reshaped)
            back_continuation_reshaped = torch.matmul(ArQr, back_bnd_reshaped)
        front_continuation = front_continuation_reshaped.reshape(*y_shape[:-3], self.C, y_shape[-2], y_shape[-1])
        back_continuation = back_continuation_reshaped.reshape(*y_shape[:-3], self.C, y_shape[-2], y_shape[-1])
        
        return torch.cat((front_continuation, x, back_continuation), dim=-3)

    def extend1d(self, x):
        return self.extend_left_right(x)
    
    def extend2d(self, x):
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        return x

    def extend3d(self, x):
        x = self.extend_left_right(x)
        x = self.extend_top_bottom(x)
        x = self.extend_front_back(x)
        return x
    
    def forward(self, x, dim=2):
        if dim == 1:
            return self.extend1d(x)
        if dim == 2:
            return self.extend2d(x)
        if dim == 3:
            return self.extend3d(x)

    def restrict(self, x, dim):
        c=self.n_additional_pts // 2
        return x[(Ellipsis,) + (slice(c, -c),)*dim]
    
class FC2D(nn.Module):
    def __init__(self, d=5, n_additional_pts=50, Lx=1.0, Ly=1.0, matrices_path=None, device="cpu"):
        super().__init__()
        self.d = d
        self.n_additional_pts = n_additional_pts
        self.Lx = Lx
        self.Ly = Ly
        self.device = device
        self.fcgram = FCGram(d=d, n_additional_pts=n_additional_pts, matrices_path=matrices_path).to(device)

    def _fft_derivative(self, u, order_x=0, order_y=0, Lx=None, Ly=None):
        if Lx is None: Lx = self.Lx
        if Ly is None: Ly = self.Ly

        ny, nx = u.shape[-2], u.shape[-1]
        dx, dy = Lx/nx, Ly/ny

        u_hat = torch.fft.fft2(u, dim=(-2, -1))

        kx = torch.fft.fftfreq(nx, d=dx, device=u.device)*(2*torch.pi)
        ky = torch.fft.fftfreq(ny, d=dy, device=u.device)*(2*torch.pi)
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")

        # differentiate in fourier space
        u_hat_diff = ((1j * KX) ** order_x) * ((1j * KY) ** order_y) * u_hat
        return torch.fft.ifft2(u_hat_diff, dim=(-2, -1)).real

    def diff_x(self, u, order=1):
        u = u.to(self.device)
        u_ext = self.fcgram.extend2d(u)

        ny, nx = u_ext.shape[-2], u_ext.shape[-1]
        Lx = self.Lx * nx / (nx - self.n_additional_pts)
        Ly = self.Ly * ny / (ny - self.n_additional_pts)

        u_diff_ext = self._fft_derivative(u_ext, order_x=order, order_y=0, Lx=Lx, Ly=Ly)
        return self.fcgram.restrict(u_diff_ext, dim=2)

    def diff_xx(self, u):
        return self.diff_x(u, order=2)

    def diff_y(self, u, order=1):
        u = u.to(self.device)
        u_ext = self.fcgram.extend2d(u)

        ny, nx = u_ext.shape[-2], u_ext.shape[-1]
        Lx = self.Lx * nx/(nx - self.n_additional_pts)
        Ly = self.Ly * ny/(ny - self.n_additional_pts)

        u_diff_ext = self._fft_derivative(u_ext, order_x=0, order_y=order, Lx=Lx, Ly=Ly)
        return self.fcgram.restrict(u_diff_ext, dim=2)

    def gradient(self, u):
        return torch.stack([self.diff_x(u), self.diff_y(u)], dim=0)

    def laplacian(self, u):
        return self.diff_x(u, order=2)+self.diff_y(u, order=2)
