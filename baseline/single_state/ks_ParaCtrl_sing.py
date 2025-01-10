# KSequ.m - solution of Kuramoto-Sivashinsky equation
#
# u_t = -u*u_x - u_xx - \niu u_xxxx, periodic boundary conditions on [0,32*pi]
# computation is based on v = fft(u), so linear term is diagonal
#
# Using this program:
# u is the initial condition
# h is the time step
# N is the number of points calculated along x
# a is the max value in the initial condition
# b is the min value in the initial condition
# x is used when using a periodic boundary condition, to set up in terms of
#   pi
#
# Initial condition and grid setup
import torch
import neuralop.wcw.tool_wcw as wcw
counter_file = "counter.txt"
file_id=wcw.id_filename(counter_file)
proj_name="KS_solver"

# torch.manual_seed(0)

norm='forward'#FFT norm
niu=0.01
# N=1024 #space grid size
N=128 #space grid size
N_proj=128

# timegrid=0.25
timegrid=0.01

#!!!!
half_period=3 #domain=2pi*half_period, basis =e^{k x/half_period}
# half_period=16 #domain=2pi*half_period, basis =e^{k x/half_period}
space_scaling=half_period*2*torch.pi # original=1,correct=32pi or 32k pi

#initial_condition_choose
'''    functions = {
        0:period_original,1: nonperiod_original,
        2:single_steady,3:single_worst,
        4:nonL_period,5:random_gen
    }'''
choose_ic=5

#scheme
choose_coef=0
'''    functions={
        0:ff,1:ff,10:ff_ffexact,30:exact_taylor()
    }
    '''
M=64# number of points in complex unit circle to do average

#150,100,0.25
#5w, 10w
tmax=150#00
nmax=round(tmax/timegrid)#epoch
num_plot=150 #num of plot capture
nplt=int((tmax/num_plot)/timegrid)# save for plot, every nplt iter 1.5/0.25=6
dt_save=nplt*timegrid

eta_stb_change=0.10

Nsum=1 #400

file_name=f"T={tmax},niu={niu},N={N},dt={timegrid},6pi,dtsave={dt_save},sample={Nsum}_sgs_({file_id})."
# file_name=f"__T={tmax} ic={choose_ic},niu={niu},N={N},tmgd={timegrid},evel={choose_coef},eta={eta_stb_change}({file_id})."


'''sgs related'''
cs=0.005j # 0.1-1
clos_d=1
x_grid=space_scaling/N

start_plot_time=20
pre_check=0 # check before save
