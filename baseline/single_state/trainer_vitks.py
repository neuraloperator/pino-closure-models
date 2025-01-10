import torch.nn.functional as F
from timeit import default_timer
from matplotlib import pyplot as plt
from pdb import set_trace as st
import cv2
from pytorch_msssim import  ms_ssim, ssim
import wandb
import sys
import json
import torch
import wandb
from utilities3 import *
from torch import nn
import pytorch_warmup as warmup
EVALUATE = False

learning_rate = 1e-4
IS_KF =1- True
USE_WANDB = True

if USE_WANDB:
    wandb.init(
        dir="./wandb_files",
        entity="peterwg",
        project='ks_closure',
        name=str(sys.argv[1]),
    )

if IS_KF:
    from vision_transformer import vit_b_kf
    from KF_load_1 import train_loader, test_loader, y_normalizer
    model =  vit_b_kf(num_classes=1024).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)
else:
    from vision_transformer import vit_b_ks
    from KS_load_1 import train_loader, test_loader, y_normalizer
    model =  vit_b_ks(num_classes=128).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)


epochs = 900*4 #100
iterations = epochs * len(train_loader)#(ntrain // batch_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
warmup_period=epochs//3
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=epochs//3)
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

if EVALUATE:
    model.load_state_dict(torch.load('ks_final.pt'), strict=True)

for ep in range(epochs):
    # training
    train_loss = 0
    test_loss = 0
    if not EVALUATE:
        model.train()
        for index_, data in enumerate(train_loader):
            if IS_KF:
                x = data['x'].permute(0, 3, 1, 2).cuda()
            else:
                x = data['x'].unsqueeze(1).unsqueeze(1).float().cuda()
            #x += (torch.rand(x.shape)-0.5).cuda()*x.abs().max()*1e-1
            y = data['y'].cuda().float()
            output = model(x)

            if IS_KF:
                output = output.reshape(-1, 4, 16, 16)
            #print(torch.unique(output))
            #st()
            loss = l1loss(y, output) + l2loss(y, output)
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()
            train_loss += loss
        train_loss /= len(train_loader)
        if USE_WANDB and ep %50==0:
            wandb.log({"epoch": ep, "train_loss": train_loss})

    # test
    model.eval()
    for index_, data in enumerate(test_loader):
        with torch.no_grad():
            if IS_KF:
                x = data['x'].permute(0, 3, 1, 2).cuda()
            else:
                x = data['x'].unsqueeze(1).unsqueeze(1).float().cuda()
            y = data['y'].cuda()
            output = model(x)

            if IS_KF:
                output = output.reshape(-1, 4, 16, 16)
            loss = l1loss(y, output) + l2loss(y, output)
            
            test_loss += loss
            # unnormalize
            #st()
            output = y_normalizer.decode(output)
    test_loss /= len(test_loader)
    if USE_WANDB and ep %50==0:
        wandb.log({"epoch": ep, "test_loss": test_loss})

torch.save(model.state_dict(), 'ks_final.pt') 