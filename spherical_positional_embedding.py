import torch

def rope_spherical_for_images(x, base=10000.0):
    '''
    inputs:
    x, shaped (B batches, H heads, rows, cols, C channels) or just (B, rows, cols, C)
    we gonna use the following rotation matrix with 2 angles. this was from SPHERICAL POSITION ENCODING FOR TRANSFORMERS (Oct 4 2023) by Eren Unlu
    [[cos(θ), −cos(ϕ)sin(θ), sin(ϕ)sin(θ)],
    [sin(θ), cos(ϕ)cos(θ), −sin(ϕ)cos(θ)],
    [0, sin(ϕ), cos(ϕ)]]
    '''
    rows, cols, C = x.shape[-3:]
    assert C % 3 == 0

    #these are used for reshaping tensors
    dim_sizes=list(x.shape)
    flattened_dim_sizes = dim_sizes[:-3] + [dim_sizes[-3]*dim_sizes[-2]] + [dim_sizes[-1]] #..., rows*cols, C
    expanded_dim_sizes = flattened_dim_sizes+[3]
    expanded_dim_sizes[-2]=C//3 #...rows, cols, C//3, C

    #these are used to make the row and column indices for each token
    rows_pos = torch.arange(rows,device=x.device,dtype=torch.float32) #shape (rows,)
    cols_pos = torch.arange(cols,device=x.device,dtype=torch.float32) #shape (cols,)
    grid_y, grid_x = torch.meshgrid(rows_pos, cols_pos, indexing='ij')  # Ensure 'ij' indexing
    coords = torch.stack([grid_y, grid_x], dim=-1)  # Shape: (rows, cols, 2)
    coords = coords.reshape(-1, 2) #shape (rows*cols,2)
    rows_pos = coords[:,:-1] #shape (rows*cols,1)
    cols_pos = coords[:,-1:] #shape (rows*cols,1)

    # 3) Create frequency scalings for each pair index
    freq_range = torch.arange(C // 3, device=x.device, dtype=torch.float32)  # indices [0..(C/3 - 1)]
    alpha = base ** (-3.0 * freq_range / C)  # shape (C/3,)
    theta = cols_pos * alpha  # shape (rows*cols,C/3)
    phi = rows_pos * alpha  # shape (rows*cols,C/3)

    # 5) Compute cos and sin of these angles
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    x = x.view(flattened_dim_sizes) #shape (..., rows*cols, C)
    feats_3d = x.view(expanded_dim_sizes) #shape (..., rows*cols, C//3, 3)
    #each of these 3 tensors below is shaped (...,rows*cols, C//3)
    x0 = feats_3d[..., 0]
    x1 = feats_3d[..., 1]
    x2 = feats_3d[..., 2]

    x0_rot = cos_theta * x0 - cos_phi * sin_theta * x1 + sin_phi * sin_theta * x2
    x1_rot = sin_theta * x0 + cos_phi * cos_theta * x1 - sin_phi * cos_theta * x2
    x2_rot = sin_phi * x1 + cos_phi * x2

    # 7) Reshape back and re-concatenate the position channel
    rotated_feats = torch.stack([x0_rot, x1_rot, x2_rot], dim=-1)  # (..., C/3, 3)
    rotated_feats = rotated_feats.view(dim_sizes) #original x shape
    return rotated_feats

if __name__=='__main__':
    x=torch.randn((1,2,4,4,18))
    x_flattened=x.reshape(-1,x.shape[-1])
    x_norms_before=torch.sqrt(torch.sum(x_flattened**2,dim=-1))
    print(x_norms_before)

    y=rope_spherical_for_images(x)
    x_flattened2=x.reshape(-1,x.shape[-1])
    x_norms_after=torch.sqrt(torch.sum(x_flattened**2,dim=-1))
    print(x_norms_after)