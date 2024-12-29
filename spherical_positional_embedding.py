import torch
def rope_spherical(x, positions, base=10000.0):
    '''
    inputs:
    x, shaped (batches, heads, tokens, channels) or just (batches, tokens, channels)
    positions, shaped (batches, heads, tokens, 2), or just (tokens, 2). the last dim contains the x and y positions of the tokens
    use the following rotation matrix with 2 angles, taken from SPHERICAL POSITION ENCODING FOR TRANSFORMERS (Oct 4 2023) by Eren Unlu:
    [[cos(θ), −cos(ϕ)sin(θ), sin(ϕ)sin(θ)],
    [sin(θ), cos(ϕ)cos(θ), −sin(ϕ)cos(θ)],
    [0, sin(ϕ), cos(ϕ)]]
    '''
    C = x.shape[-1]
    assert C%3==0
    x_pos = positions[...,:1] #(...,1)
    y_pos = positions[...,1:] #(...,1)

    freq_range = torch.arange(C // 3, device=x.device, dtype=torch.float32)  # indices [0..(C/3 - 1)]
    alpha = base ** (-3.0 * freq_range / C)  # shape (C/3,)
    theta = x_pos * alpha #shape (...,C/3)
    phi = y_pos * alpha #shape (...,C/3)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    dim_sizes = list(x.shape) + [3]
    dim_sizes[-2] = C//3
    feats_3d = x.view(dim_sizes)
    x0 = feats_3d[...,0]
    x1 = feats_3d[...,1]
    x2 = feats_3d[...,2]
    x0_rot = cos_theta * x0 - cos_phi * sin_theta* x1 + sin_phi*sin_theta *x2
    x1_rot = sin_theta * x0 + cos_phi * cos_theta * x1 - sin_phi*cos_theta * x2
    x2_rot = sin_phi*x1 + cos_phi*x2

    rotated_feats = torch.stack([x0_rot, x1_rot,x2_rot], dim=-1)  # (..., C/3, 3)
    rotated_feats = rotated_feats.view(x.shape)  #back to original x shape

    return rotated_feats
def rope_spherical_for_images(x, base=10000.0):
    '''
    inputs:
    x, shaped (batches, heads, rows, cols, channels) or just (batches, rows, cols, channels)
    use the following rotation matrix with 2 angles, taken from SPHERICAL POSITION ENCODING FOR TRANSFORMERS (Oct 4 2023) by Eren Unlu:
    [[cos(θ), −cos(ϕ)sin(θ), sin(ϕ)sin(θ)],
    [sin(θ), cos(ϕ)cos(θ), −sin(ϕ)cos(θ)],
    [0, sin(ϕ), cos(ϕ)]]
    '''
    rows, cols, C = x.shape[-3:]
    assert C % 3 == 0

    dim_sizes=list(x.shape)
    flattened_dim_sizes = dim_sizes[:-3] + [dim_sizes[-3]*dim_sizes[-2]] + [dim_sizes[-1]] #..., rows*cols, C
    expanded_dim_sizes = flattened_dim_sizes+[3]
    expanded_dim_sizes[-2]=C//3 #...rows, cols, C//3, C

    rows_pos = torch.arange(rows,device=x.device,dtype=torch.float32) #shape (rows,)
    cols_pos = torch.arange(cols,device=x.device,dtype=torch.float32) #shape (cols,)
    grid_y, grid_x = torch.meshgrid(rows_pos, cols_pos, indexing='ij')
    coords = torch.stack([grid_y, grid_x], dim=-1)  # Shape: (rows, cols, 2)
    coords = coords.reshape(-1, 2) #shape (rows*cols,2)
    rows_pos = coords[:,:-1] #shape (rows*cols,1)
    cols_pos = coords[:,-1:] #shape (rows*cols,1)

    freq_range = torch.arange(C // 3, device=x.device, dtype=torch.float32)  # indices [0..(C/3 - 1)]
    alpha = base ** (-3.0 * freq_range / C)  # shape (C/3,)
    theta = cols_pos * alpha  # shape (rows*cols,C/3)
    phi = rows_pos * alpha  # shape (rows*cols,C/3)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    x = x.view(flattened_dim_sizes) #shape (..., rows*cols, C)
    feats_3d = x.view(expanded_dim_sizes) #shaped (..., rows*cols, C//3, 3)
    #each these 3 below is shaped (...,rows*cols, C//3)
    x0 = feats_3d[..., 0]
    x1 = feats_3d[..., 1]
    x2 = feats_3d[..., 2]
    x0_rot = cos_theta * x0 - cos_phi * sin_theta * x1 + sin_phi * sin_theta * x2
    x1_rot = sin_theta * x0 + cos_phi * cos_theta * x1 - sin_phi * cos_theta * x2
    x2_rot = sin_phi * x1 + cos_phi * x2
    rotated_feats = torch.stack([x0_rot, x1_rot, x2_rot], dim=-1)  # (..., C/3, 3)
    rotated_feats = rotated_feats.view(dim_sizes) #original x shape
    return rotated_feats

if __name__=='__main__':
    x = torch.randn((2, 2, 10, 18))
    pos = torch.randn((10, 2)) ** 2

    xflat0 = x.reshape(-1, x.shape[-1])
    x_norms0 = torch.sqrt(torch.sum(xflat0**2, dim=-1))
    print(x_norms0)

    x = rope_spherical(x, pos)

    xflat1 = x.reshape(-1, x.shape[-1])
    x_norms1 = torch.sqrt(torch.sum(xflat1**2, dim=-1))
    print(x_norms1)

    print('\n for images')

    x=torch.randn((2,2,4,4,18))
    x_flat0=x.reshape(-1,x.shape[-1])
    x_norms0 = torch.sqrt(torch.sum(x_flat0**2, dim=-1))
    print(x_norms0)
    x = rope_spherical_for_images(x)
    xflat1 = x.reshape(-1, x.shape[-1])
    x_norms1 = torch.sqrt(torch.sum(xflat1**2, dim=-1))
    print(x_norms1)
