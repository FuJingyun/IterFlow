import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from scripts.raflow_utils.lib import pointnet2_utils as pointutils
from scripts.network.models.basic.PT_util.PT_layer import  PT_corr_Block



class BQ_CorrBlock(nn.Module):
    def __init__(self):
        super(BQ_CorrBlock, self).__init__()
        self.radius = [1]   
        self.nsamples = [8]  
        self.ball_num = len(self.radius)

        self.conv = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.PReLU()
        )
        self.out = nn.Conv1d(64, 64, 1)

    def init_module(self, fmap1, fmap2):
        self.corr = self.calculate_corr(fmap1, fmap2)

    def __call__(self, coords, xyz2):
        return self.best_multiscale_feature(coords, xyz2)


    def best_multiscale_feature(self, coords, xyz2):
        b, n_p, _ = coords.size()
        all_corr = None

        for ball_id in range(self.ball_num):
            radius = self.radius[ball_id]
            nsample = self.nsamples[ball_id]
            cur_queryandgroup = pointutils.my_QAG(radius, nsample)
            # idx ：torch.Size([B, N, nsample])
            # grouped_xyz ： torch.Size([B, 3, N, nsample])
            idx, grouped_xyz = cur_queryandgroup(xyz2, coords)
            assert idx.is_contiguous()
            idx = idx.long()
            # torch.Size([B, 1, N, nsample])
            cur_corr = torch.gather(self.corr.view(b * n_p, n_p), dim=1,
                                    index=idx.reshape(b * n_p, nsample)).reshape(b, 1, n_p, nsample)
            
            cur_feature = self.conv(torch.cat([cur_corr, grouped_xyz], dim=1))
            # B 64 N
            cur_feature = torch.max(cur_feature, dim=3)[0]
            if all_corr is None:
                all_corr = self.out(cur_feature)
            else:
                all_corr = all_corr + self.out(cur_feature)
        return all_corr
            

    @staticmethod
    def calculate_corr(fmap1, fmap2):
        _, dim, _ = fmap1.shape
        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr / torch.sqrt(torch.tensor(dim).float())
        return corr
    


