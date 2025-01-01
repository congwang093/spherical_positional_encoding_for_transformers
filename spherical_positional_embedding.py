import torch
import torch.nn as nn

class Rope_Spherical(nn.Module):
    '''applies this rotation matrix to Query or Key tensors for self attention
            [[cos(θ), −cos(ϕ)sin(θ), sin(ϕ)sin(θ)],
            [sin(θ), cos(ϕ)cos(θ), −sin(ϕ)cos(θ)],
            [0, sin(ϕ), cos(ϕ)]]'''
    
    def __init__(self,dim, rows=32,cols=32, positions=None):
        super(Rope_Spherical, self).__init__()
        assert dim % 3 == 0

        if type(positions)==type(None):
            positions=self.generate_positions(rows,cols)

        x_pos = positions[..., :1]  # (...,N,1)
        y_pos = positions[..., 1:]  # (...,N,1)
        freq_range = torch.arange(dim // 3, dtype=torch.float32)
        alpha = 10000 ** (-3.0 * freq_range / dim)  # shape (C/3,)
        # shape (...N,C/3) for these below
        theta = x_pos * alpha
        phi = y_pos * alpha
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)
        
        w = torch.zeros((3,3,positions.shape[0],dim//3))
        w[0,0,:,:]=cos_theta
        w[0,1,:,:]=-cos_phi*sin_theta
        w[0,2,:,:]=sin_phi*sin_theta

        w[1,0,:,:]=sin_theta
        w[1,1,:,:]=cos_phi*cos_theta
        w[1,2,:,:]=-sin_phi*cos_theta

        w[2,1,:,:]=sin_phi
        w[2,2,:,:]=cos_phi
        self.dim=dim
        self.register_buffer('w',w.permute(2,3,0,1)) #(N, dim//3, 3, 3)

    @staticmethod
    def generate_positions(rows, cols):
        rows_pos = torch.arange(rows, dtype=torch.float32)  #(rows,)
        cols_pos = torch.arange(cols, dtype=torch.float32)  #(cols,)
        grid_y, grid_x = torch.meshgrid(rows_pos, cols_pos, indexing='ij')
        coords = torch.stack([grid_y, grid_x], dim=-1)  #(rows, cols, 2)
        positions = coords.reshape(-1, 2)  # shape (rows*cols,2)
        return positions

    def forward(self,x):
        '''x should be a shape of (B batches, H heads, N tokens, C channels), returns same shape'''
        B,H,N,C=x.shape
        x=x.view(B, H, N, C//3, 3).unsqueeze(4) #B, H, N, C//3, 1, 3
        x = torch.sum(x*self.w,dim=-1) #(B, H, N, C//3, 1, 3) & (N, C//3, 3, 3) - > (B, H, N, C//3, 3)
        x=x.reshape(B,H,N,C) #(B, H, N, C//3, 3) - > (B,H,N,C)
        return x

if __name__ == "__main__":
    Q = torch.randn((2, 2, 100, 30))  #flattened image tensor (Batches,Heads,N_total_tokens,Channels)
    qnorm = torch.sqrt(torch.sum(Q ** 2, dim=-1))

    rope_spherical = Rope_Spherical(Q.shape[-1], rows=10, cols=10)
    Q = rope_spherical(Q)
    qnorm2 = torch.sqrt(torch.sum(Q ** 2, dim=-1))

    are_close = torch.allclose(qnorm, qnorm2, atol=1e-7)
    print(are_close)  # True
