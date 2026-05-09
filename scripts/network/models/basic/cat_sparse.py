import torch
import torch.nn as nn

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True
import spconv.pytorch as spconv

class Cat_sp_tensor(nn.Module):
    def __init__(self, voxel_size, pseudo_image_dims, point_cloud_range,
                 feat_channels: int) -> None:
        super().__init__()
        self.pseudo_image_dims = pseudo_image_dims
        self.sparse_3d_dims=[1, pseudo_image_dims[0], pseudo_image_dims[1]]
        

    def forward(self, sp1, sp2) -> torch.Tensor:
        indices1 = sp1.indices
        features1 = sp1.features

        indices2 = sp2.indices
        features2 = sp2.features


        cat_indices = torch.cat((indices1, indices2), dim=0)
        unq_no_zero, unq_inv_no_zero = torch.unique(cat_indices, return_inverse=True, return_counts=False, dim=0)
        
        # print("unq_no_zero")
        # print(unq_no_zero.size())
        # print("unq_inv_no_zero")
        # print(unq_inv_no_zero.size())
        out_fea = torch.zeros((unq_no_zero.shape[0], features1.shape[1] + features2.shape[1]), dtype=features1.dtype, device=features1.get_device())

        
        for i in range(features1.shape[0] + features2.shape[0]):
            if i<features1.shape[0] :
                # lidar fea
                out_fea[unq_inv_no_zero[i], :features1.shape[1]] = features1[i]
            else:
                # radar fea
                out_fea[unq_inv_no_zero[i], features1.shape[1]:(features1.shape[1] + features2.shape[1])] = features2[i-features1.shape[0]]

        # sparse_tensor_3d = spconv.SparseConvTensor(out_fea.contiguous(), unq_no_zero.contiguous(), self.sparse_3d_dims, sp1.batch_size)
        sparse_tensor_2d = spconv.SparseConvTensor(out_fea.contiguous(), unq_no_zero.contiguous(), self.pseudo_image_dims, sp1.batch_size)


        # return sparse_tensor_3d
        return sparse_tensor_2d