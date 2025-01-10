from pathlib import Path
import torch

import neuralop_base
import sys
sys.modules['neuralop'] = neuralop_base


from neuralop.utils import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from torch.utils.data.sampler import SubsetRandomSampler

from .transforms import PositionalEmbedding


def load_data_small(
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    train_resolution,#int
    test_resolutions=[16, 32], #only used as wandb plot suffix
    grid_boundaries=[[0, 1], [0, 1]],# need to change
    positional_encoding=True,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    channel_dim=1,#
    dim_pde=1,#
    in_data=None,#
    K=None,
):
    """Loads a small Darcy-Flow dataset

    Training contains 1000 samples in resolution 16x16.
    Testing contains 100 samples at resolution 16x16 and
    50 samples at resolution 32x32.

    Parameters
    ----------
    n_train : int
    n_tests : int
    batch_size : int
    test_batch_sizes : int list
    test_resolutions : int list, default is [16, 32],
    grid_boundaries : int list, default is [[0,1],[0,1]],
    positional_encoding : bool, default is True
    encode_input : bool, default is False
    encode_output : bool, default is True
    encoding : 'channel-wise'
    channel_dim : int, default is 1
        where to put the channel dimension, defaults size is 1
        i.e: batch, channel, height, width

    Returns
    -------
    training_dataloader, testing_dataloaders

    training_dataloader : torch DataLoader
    testing_dataloaders : dict (key: DataLoader)
    """

    path = Path(__file__).resolve()
    return load_data_pt(
        str(path),
        n_train=n_train,
        n_tests=n_tests,
        batch_size=batch_size,
        test_batch_sizes=test_batch_sizes,
        test_resolutions=test_resolutions,
        train_resolution=train_resolution,
        grid_boundaries=grid_boundaries,
        positional_encoding=positional_encoding,
        encode_input=encode_input,
        encode_output=encode_output,
        encoding=encoding,
        dim_pde=dim_pde,
        in_data=in_data,
        K=K
    )


def load_data_pt(
    data_path,
    n_train,
    n_tests,
    batch_size,
    test_batch_sizes,
    test_resolutions=[32],
    train_resolution=32,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    channel_dim=1,
    dim_pde=2,
    in_data=None,
    K=None
):

    if in_data is not None:
        data=in_data['train'][0]

    data_devise=data['x'].device

    train_loaders={}
    for i in range(len(in_data['train'])):  # data: dict,'x','y','t_val' for loss
        data=in_data['train'][i]
        t_val=data['t_val']

        if data['x'].dim()==dim_pde+1: #N*gx*gy... , 'g':grid number
            x_train = (
                data["x"][0:n_train[i]].unsqueeze(channel_dim).type(torch.float32).clone()
            )
        elif data['x'].dim()==dim_pde+2: #N*feature*gx*gy...
            x_train = (
                data["x"][0:n_train[i]].type(torch.float32).clone()
            )
        else:
            print("data_load.py: load_darcy_pt: new cases not yet implemented_train x")
            exit()
        #x_train: N*f*gx*gy..
        y_train = data["y"][0:n_train[i]].unsqueeze(channel_dim).clone()#N*1*gx*gy..
        del data

        train_db = TensorDataset(#input of transform_x: x[i] i.e. f*gx*gy or 1*gx*gy
            x_train,
            y_train,
            transform_x=PositionalEmbedding(grid_boundaries, 0,dim_pde=dim_pde)
            if positional_encoding
            else None,
        )
        if i==-1:
            train_loader = torch.utils.data.DataLoader(
                train_db,
                batch_size=batch_size,
                # shuffle=True,
                num_workers=0,
                pin_memory=(data_devise=='cpu'),
                persistent_workers=False,
                sampler=SubsetRandomSampler(range(K))
            )
        else:
            train_loader = torch.utils.data.DataLoader(
                train_db,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=(data_devise == 'cpu'),
                persistent_workers=False,
                # sampler=SubsetRandomSampler(range(K))
            )
        train_loaders[i]=[train_loader,t_val]


    test_loaders = {}

    ii=0
    for (res, n_test, test_batch_size) in zip(
        test_resolutions, n_tests, test_batch_sizes
    ):
        print(
            f"Loading test db at resolution {res} with {n_test} samples "
            f"and batch-size={test_batch_size}"
        )
        '''change path here'''

        '''change path block end here'''
        if in_data is not None:
            data = in_data['test'][ii]
            t_val = data['t_val']
            ii+=1


        print("------------------------------------")
        print(data['x'].shape)
        if data['x'].dim() == dim_pde + 1:  # N*gx*gy...
            x_test = (
                data["x"][0:n_test].unsqueeze(channel_dim).type(torch.float32).clone()
            )
        elif data['x'].dim() == dim_pde + 2:  # N*feature*gx*gy...
            x_test = (
                data["x"][0:n_test].type(torch.float32).clone()
            )
        else:
            print("data_load.py: load_darcy_pt: new cases not yet implemented_test x")
            exit()
        y_test = data["y"][:n_test, :].unsqueeze(channel_dim).clone()
        del data
        # if input_encoder is not None:
        #     x_test = input_encoder.encode(x_test)

        test_db = TensorDataset(
            x_test,
            y_test,
            transform_x=PositionalEmbedding(grid_boundaries, 0,dim_pde=dim_pde)
            if positional_encoding
            else None,
        )
        test_loader = torch.utils.data.DataLoader(
            test_db,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=(data_devise=='cpu'),
            persistent_workers=False,
        )
        test_loaders[res] = [test_loader,t_val]

    return train_loaders, test_loaders,None #, output_encoder
