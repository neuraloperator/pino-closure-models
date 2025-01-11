
import wandb
from utilities3 import *
from torch import nn
import pytorch_warmup as warmup #

import time as tm
import neuralop_advance

from neuralop_advance.utils import get_wandb_api_key, count_model_params



load_per=50 #pritn and load
epochs = 8000 #100
warm_ep=epochs//4

learning_rate = 2.5e-5
IS_KF = True
USE_WANDB = 1
EVALUATE=0


'''les resolution'''
res=48
re=1000
wandb.login(key=get_wandb_api_key(api_key_file="../../config/wandb_api_key.txt"))
print("success!!!!")
wandb_name = "_".join(
    f"{var}"
    for var in [

        learning_rate,
        epochs,
        warm_ep
    ]
)
wandb_args = dict(
    # config=config,
    name=wandb_name,
    group='',
    project="add your project name here",
    entity="add your username here",
)

if USE_WANDB:
    wandb.init(**wandb_args)

if IS_KF:
    if re==100:
        from vision_transformer import vit_b_kf
        from KF_load_1 import train_loader, test_loader, y_normalizer
        model =  vit_b_kf(num_classes=1024).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)
    else:
        print('go this way')
        from vision_transformer_ns1k import vit_b_ns1k
        from ns1k_load_1 import train_loader, test_loader, y_normalizer
        model =  vit_b_ns1k(num_classes=9216).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)
else:
    from vision_transformer import vit_b_ks
    from KS_load_1 import train_loader, test_loader, y_normalizer
    model =  vit_b_ks(num_classes=128).cuda() #nn.Transformer(nhead=16, num_encoder_layers=12)



iterations = epochs * len(train_loader)#(ntrain // batch_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
warmup_period=epochs//3
warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=warm_ep)  #epochs//3
l1loss = nn.L1Loss()
l2loss = nn.MSELoss()

if EVALUATE:
    model.load_state_dict(torch.load('kf_final.pt'), strict=True)

t1=tm.time()
t2=0
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


            output = model(x) #btz, 9216 (=4,48,48)

            if IS_KF:
                output = output.reshape(-1, 4, res, res)  #(-1,4,16,16) for Re100

            loss = l1loss(y, output) + l2loss(y, output)
            loss.backward()
            optimizer.step()
            with warmup_scheduler.dampening():
                if warmup_scheduler.last_step + 1 >= warmup_period:
                    scheduler.step()
            train_loss += loss
        train_loss /= len(train_loader)
        if USE_WANDB and ep %load_per==0:
            wandb.log({"epoch": ep, "train_loss": train_loss})
            print(f'epoch: {ep}, train_loss: {train_loss}',end=' ')



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
                output = output.reshape(-1, 4, res, res)
            loss = l1loss(y, output) + l2loss(y, output)
            
            test_loss += loss
            # unnormalize
            #st()
            output = y_normalizer.decode(output)
    test_loss /= len(test_loader)
    if USE_WANDB and ep %load_per==0:
        t2=tm.time()
        wandb.log({"epoch": ep, "test_loss": test_loss})
        print(f'test_loss: {test_loss}; time={t2-t1}')
        t1=t2


torch.save(model.state_dict(), f'kf_{re}_final.pt')