import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange

class CL(nn.Module):
    def __init__(self, n, w=[1.0, 1.0], s=64.0):
        super(CL, self).__init__()
        self.s = s
        self.w = w
        self.n = n

    def loss_c(self, x):
        n = x.shape[1]
        x = F.normalize(x, dim=-1)
        
        x_col = x.unsqueeze(1)
        x_row = x.unsqueeze(2)

        dist = (x_col - x_row).norm(2, dim=-1).pow(2)
        valid_dist = dist[:, torch.tril(torch.ones(n, n), -1).to(bool)]
        
        return valid_dist.mean()

    def loss_d(self, x, p):
        x = F.normalize(x, dim=-1)
        p = F.normalize(p, dim=-1)

        cosine = torch.einsum('ik,jk->ij', x, p)
        cosine = cosine.clamp(-1.0, 1.0)
        cosine[cosine < 0] = 0.0

        return torch.log(cosine.pow(2).mul(self.s).exp().mean(dim=1)).mean()

    def forward(self, x, p):
        x = rearrange(x, '(k m) d -> k m d', m = self.n + 1)
        easy = x[:, :self.n, :]
        hard = x[:, -1, :]

        loss = self.w[0] * self.loss_c(easy) + self.w[1] * self.loss_d(hard, p)

        return loss
