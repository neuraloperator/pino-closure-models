# ViT baselines

## Training

To train KS ViT model
```
python trainer_vitks.py EXPNAME
```

To train KF ViT model
```
python trainer_vitkf.py EXPNAME
```

## Pre-trained Models
How do you evaluate the trained model for [ks](./ks_final.pt) or  [kf](./kf_final.pt)? Taking `KS` as an example, Set [EVALUATE flag to True](./trainer_vitks.py#L15), which, in the corresponding file, uncomment [these lines](./trainer_vitks.py#L50-L51) and only evaluate the model. 