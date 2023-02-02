import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ArcFace']

class FC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC, self).__init__()
        self.weights = nn.Parameter(torch.FloatTensor(in_dim, out_dim))
        nn.init.normal_(self.weights, std=0.01)

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.weights, dim=0)
        cosine = torch.einsum('bi,ic->bc', x_norm, w_norm)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        return cosine

class ArcFace(nn.Module):
    def __init__(self, in_dim, out_dim, s=64.0, m=0.5):
        super(ArcFace, self).__init__()
        self.fc = FC(in_dim, out_dim)
        self.s = s
        self.m = m
        self.out_dim = out_dim

    def get_hard_classes(self, indexes, labels):
        masks = torch.zeros(self.out_dim, dtype=bool).to(indexes.device)
        masks[indexes.reshape(-1)] = True
        masks[labels] = False
        return self.fc.weights[:, masks].data.T

    def forward(self, x, y, nc):
        cosine = self.fc(x)
        theta = cosine[torch.arange(0, x.shape[0]), y].acos()

        with torch.no_grad():
            intra_scores = cosine[torch.arange(0, x.shape[0]), y].clone()

            neg_cosine = cosine.clone()
            neg_cosine[torch.arange(0, x.shape[0]), y] = -1.0
            _, hard_indexes = neg_cosine.topk(nc, dim=-1)
            dict_indexes = {k:v for k,v in zip(y.cpu().numpy(), hard_indexes)}

        # assert theta + m < pi
        mask = (theta < math.pi - self.m)
        theta[mask] += self.m
        margin_cosine = theta.cos()
        margin_cosine[~mask] -= math.sin(math.pi - self.m) * self.m

        cosine[torch.arange(0, x.shape[0]), y] = margin_cosine
        logits = self.s * cosine
        return logits, intra_scores, dict_indexes
        