# Baseline Methods
`sgs/`: Smagorinsky closure model.
`smag_lilly_model`: Dynamical Smagorinsky model.
`single_state/`: Learning-based single-state closure model.
Experiments related to multi-fidelity FNO is implemented similar to the main code.

## Smagorinsky closure model.
For KS equation,
```
python ks_sgs.py
```

Select the coefficient of eddy-viscosity term by tuning *cs* in `ks_ParaCtrl`.

For Navier-Stokes equation,
```
python kf_sgs.py
```
modify 'linknm' and 're' in Line 21 and 22 to switch between two test cases with different Reynolds number. Note that 're' corresponds to the reciprocal of the coefficient to Laplacian term in the equation, which is different from the Reynolds number of the system. The simulation is recorded for $t\in [start\_time, t\_traj\_phy]$ and saved every $dt\_save$ time period. $N$ is the number of trajectories. $s$ is the spatial resolution for coarse-grid simulation.



## Dynamical Smagorinsky model.
For Navier Stokes with small Reynolds number,
```
cd baseline/smag_lilly_model
python kf_sgs.py
```

For Navier Stokes with high Reynolds number,
```
cd baseline/smag_lilly_model
python re1w_lilly.py
```

By running these codes, they will generate a *.pt* file saving all the statistics information and an *xlsx* file saving the errors.

## Learning-based single-state closure model.


# ViT baselines
##  Dataset
The related files are `KS_load_1.py`, `KF_load_1.py` and `ns1w_load_1.py`. Make sure the relative paths towards dataset `.pt` files are correct. These datasets consist of snapshots of the fully-resolved trajectories saved when training PINO model.

## Training

For KS equation,
```
python trainer_vitks.py EXPNAME
```

For KF equations,
```
python trainer_vitkf.py EXPNAME
```

Modify the variables in Line 25-26 when for test cases with different Reynolds number.

## Prediction of Statistics

For KS, run `ks_sgs_sing.py`.

For NS , run `kf_sing.py`.

In Line 7
```
from ns1k_sing_pdes_periodic import NavierStokes2d
```
corresponds to high Reynolds number case.
```
from kf_sing_pdes_periodic import NavierStokes2d
```
corresponds to low Reynolds number case.