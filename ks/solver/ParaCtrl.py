# solution of Kuramoto-Sivashinsky equation
#
# u_t = -u*u_x - u_xx - \niu u_xxxx, periodic boundary conditions on [0,2*pi*half_period]
# computation is based on v = fft(u), so linear term is diagonal
#
# Using this program:
# u is the initial condition
# h is the time step
# N is the number of points calculated along x
# x is used when using a periodic boundary condition, to set up in terms of
#   pi

# Initial condition and grid setup
import torch

proj_name="KS_solver"
norm='forward'#FFT norm

'''Basic Setting for generating dataset'''
niu=0.01
Nsum=1# number of trajectories to generate

tmax=150#00 # Simulations on t\in [0,tmax]
N=1024 #space grid size
N_proj=1024 # no use in the experiment
timegrid=0.001

half_period=3 #domain=2pi*half_period, basis =e^{k x/half_period}
space_scaling=half_period*2*torch.pi # original=1,correct=32pi or 32k pi


'''Other Settings (no need to change)'''
#initial_condition_choose
choose_ic=5

#scheme
choose_coef=0

M=64# number of points in complex unit circle to do average


'''Solve the equation on t\in [0,tmax]. Save num_plot snapshots in total.'''

nmax=round(tmax/timegrid)#epoch
num_plot=1500 #num of plot capture
nplt=int((tmax/num_plot)/timegrid)# save for plot, every nplt iter 1.5/0.25=6
dt_save=nplt*timegrid

eta_stb_change=0.10



file_name=f"T={tmax},niu={niu},N={N},dt={timegrid},6pi,dtsave={dt_save},sample={Nsum}"


