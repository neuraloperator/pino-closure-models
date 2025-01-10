import torch
import my_tools as wcw

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def append_2d_grid_positional_encoding(input_tensor, grid_boundaries=[[0,1],[0,1]], channel_dim=1):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    shape = list(input_tensor.shape)
    shape.pop(channel_dim)
    n_samples, height, width = shape
    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1], height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1], width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    input_tensor = torch.cat((input_tensor,
                             grid_x.repeat(n_samples, 1, 1).unsqueeze(channel_dim),
                             grid_y.repeat(n_samples, 1, 1).unsqueeze(channel_dim)),
                             dim=1)
    return input_tensor

def get_grid_positional_encoding(input_tensor, grid_boundaries=[[0,1],[0,1]], channel_dim=0,dim_pde=1):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
        """
    if len(grid_boundaries)!=dim_pde:
        print("position_embedding.py: L22: dim_pde should match that of grid_boundaries!")
        exit()
    shape=list(input_tensor.shape[-dim_pde:])

    gridd=[]
    for i in range(dim_pde):
        gridd.append(torch.linspace(grid_boundaries[i][0],grid_boundaries[i][1],shape[i]+1)[:-1].to(device))
        gridd[i]=wcw.span_view(gridd[i],dim_pde,i)
        aa=shape[i]
        shape[i]=1
        gridd[i]=gridd[i].repeat(shape)
        shape[i]=aa
        if len(input_tensor.shape)==dim_pde:
            gridd[i]=gridd[i].unsqueeze(channel_dim)
        else:
            gridd[i]=gridd[i].unsqueeze(channel_dim).unsqueeze(0)



    return gridd


