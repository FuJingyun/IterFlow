import torch
import torch.nn as nn

# from typing import Tuple

import spconv as spconv_core
spconv_core.constants.SPCONV_ALLOW_TF32 = True

import spconv.pytorch as spconv
# from spconv.utils import Point2VoxelCPU3d as VoxelGenerator


class Cat_sp_tensor(nn.Module):
    def __init__(self,  pseudo_image_dims) :
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


class ConvWithNorms(nn.Module):

    def __init__(self, in_num_channels: int, out_num_channels: int,
                 kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.conv = nn.Conv2d(in_num_channels, out_num_channels, kernel_size,
                              stride, padding)
        self.batchnorm = nn.BatchNorm2d(out_num_channels)
        self.nonlinearity = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_res = self.conv(x)
        if conv_res.shape[2] == 1 and conv_res.shape[3] == 1:
            # This is a hack to get around the fact that batchnorm doesn't support
            # 1x1 convolutions
            batchnorm_res = conv_res
        else:
            batchnorm_res = self.batchnorm(conv_res)
        return self.nonlinearity(batchnorm_res)


class BilinearDecoder(nn.Module):

    def __init__(self, scale_factor: int):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x,
                                         scale_factor=self.scale_factor,
                                         mode="bilinear",
                                         align_corners=False)

class UpsampleSkip(nn.Module):
    # (512, 256, 256)
    def __init__(self, skip_channels: int, latent_channels: int,
                 out_channels: int):
        super().__init__()
        self.u1_u2 = nn.Sequential(
            nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
            BilinearDecoder(2))
        self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        self.u4_u5 = nn.Sequential(
            nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        u2_res = self.u1_u2(a)
        u3_res = self.u3(b)
        u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        return u5_res


class sp_Upsample(nn.Module):
    # (512, 256, 256)
    def __init__(self, skip_channels, latent_channels,
                 out_channels, pseudo_image_dims, up_key):
        super().__init__()
        # # nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.conv1 = spconv.SubMConv2d(skip_channels, latent_channels, kernel_size = 1, stride = 1, padding = 0)
        self.up = spconv.SparseInverseConv2d(latent_channels, latent_channels, kernel_size=(2,2), bias=False, indice_key = up_key)

        self.conv2 = spconv.SubMConv2d(latent_channels, latent_channels, kernel_size = 1, stride = 1, padding = 0)

        self.conv3 = spconv.SubMConv2d(2*latent_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = spconv.SubMConv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)

        self.cat_sp = Cat_sp_tensor(pseudo_image_dims)
        


        # self.u1_u2 = nn.Sequential(
        #     nn.Conv2d(skip_channels, latent_channels, 1, 1, 0),
        #     BilinearDecoder(2))
        # self.u3 = nn.Conv2d(latent_channels, latent_channels, 1, 1, 0)
        # self.u4_u5 = nn.Sequential(
        #     nn.Conv2d(2 * latent_channels, out_channels, 3, 1, 1),
        #     nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, a, b):
        fea_a = self.conv1(a)
        fea_a = self.up(fea_a)

        fea_b = self.conv2(b)

        cat_ab = Cat_sp_tensor(fea_a, fea_b)

        cat_ab = self.conv3(cat_ab)
        cat_ab = self.conv4(cat_ab)

        # u2_res = self.u1_u2(a)
        # u3_res = self.u3(b)
        # u5_res = self.u4_u5(torch.cat([u2_res, u3_res], dim=1))
        # return u5_res

        return cat_ab
 



class sp_conv2d_with_norm5(nn.Module):
    def __init__(self, in_planes, out_planes, mid_filters, kernel_size, key):
        super().__init__()
        # self.encoder_step_1 = nn.Sequential(ConvWithNorms(self.input_size, 64, 3, 2, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1))
        

        self.conv1 = spconv.SubMConv2d(in_planes, mid_filters, kernel_size, stride = 2, padding = 1, indice_key = key)
        self.bn1 = nn.BatchNorm1d(mid_filters)
        self.act1 = nn.LeakyReLU()

        self.conv2 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm1d(mid_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm1d(mid_filters)
        self.act3 = nn.LeakyReLU()

        self.conv4 = spconv.SubMConv2d(mid_filters, out_planes, kernel_size, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm1d(mid_filters)
        self.act4 = nn.LeakyReLU()

        self.conv5 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm1d(mid_filters)
        self.act5 = nn.LeakyReLU()


    def forward(self, sp) :
        feat = self.conv1(sp)
        feat = feat.replace_feature(self.bn1(feat.features))
        feat = feat.replace_feature(self.act1(feat.features))

        feat = self.conv2(feat)
        feat = feat.replace_feature(self.bn2(feat.features))
        feat = feat.replace_feature(self.act2(feat.features))

        feat = self.conv3(feat)
        feat = feat.replace_feature(self.bn3(feat.features))
        feat = feat.replace_feature(self.act3(feat.features))

        feat = self.conv4(feat)
        feat = feat.replace_feature(self.bn4(feat.features))
        feat = feat.replace_feature(self.act4(feat.features))

        feat = self.conv5(feat)
        feat = feat.replace_feature(self.bn5(feat.features))
        feat = feat.replace_feature(self.act5(feat.features))

        return feat



class sp_conv2d_with_norm6(nn.Module):
    def __init__(self, in_planes, out_planes, mid_filters, kernel_size, key):
        super().__init__()
        # self.encoder_step_1 = nn.Sequential(ConvWithNorms(self.input_size, 64, 3, 2, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1))

        self.conv1 = spconv.SubMConv2d(in_planes, mid_filters, kernel_size, stride = 2, padding = 1, indice_key = key)
        self.bn1 = nn.BatchNorm1d(mid_filters)
        self.act1 = nn.LeakyReLU()

        self.conv2 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn2 = nn.BatchNorm1d(mid_filters)
        self.act2 = nn.LeakyReLU()

        self.conv3 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn3 = nn.BatchNorm1d(mid_filters)
        self.act3 = nn.LeakyReLU()

        self.conv4 = spconv.SubMConv2d(mid_filters, out_planes, kernel_size, stride = 1, padding = 1)
        self.bn4 = nn.BatchNorm1d(mid_filters)
        self.act4 = nn.LeakyReLU()

        self.conv5 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn5 = nn.BatchNorm1d(mid_filters)
        self.act5 = nn.LeakyReLU()

        self.conv6 = spconv.SubMConv2d(mid_filters, mid_filters, kernel_size, stride = 1, padding = 1)
        self.bn6 = nn.BatchNorm1d(mid_filters)
        self.act6 = nn.LeakyReLU()


    def forward(self, sp) :
        feat = self.conv1(sp)
        feat = feat.replace_feature(self.bn1(feat.features))
        feat = feat.replace_feature(self.act1(feat.features))

        feat = self.conv2(feat)
        feat = feat.replace_feature(self.bn2(feat.features))
        feat = feat.replace_feature(self.act2(feat.features))

        feat = self.conv3(feat)
        feat = feat.replace_feature(self.bn3(feat.features))
        feat = feat.replace_feature(self.act3(feat.features))

        feat = self.conv4(feat)
        feat = feat.replace_feature(self.bn4(feat.features))
        feat = feat.replace_feature(self.act4(feat.features))

        feat = self.conv5(feat)
        feat = feat.replace_feature(self.bn5(feat.features))
        feat = feat.replace_feature(self.act5(feat.features))

        feat = self.conv6(feat)
        feat = feat.replace_feature(self.bn6(feat.features))
        feat = feat.replace_feature(self.act6(feat.features))

        return feat


class SP_UNet(nn.Module):
    """
    Standard UNet with a few modifications:
     - Uses Bilinear interpolation instead of transposed convolutions
    """
    def __init__(self, pseudo_image_dims):
        super().__init__()
        # self.encoder_step_0 = nn.Sequential(ConvWithNorms(32, 64, 3, 2, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1))
        self.input_size = 64 # 64
        self.cat_sp = Cat_sp_tensor(pseudo_image_dims)

        # spconv.SubMConv4d(in_planes, out_planes, kernel_size=(1,1,1,3), stride=stride,
        #                      padding=(0,0,0,1), bias=False, indice_key=indice_key)

        self.encoder_step_1 = sp_conv2d_with_norm5(self.input_size, 64, 64, 3,'down1')
        self.encoder_step_2 = sp_conv2d_with_norm6(64, 128, 128, 3,'down2')
        self.encoder_step_3 = sp_conv2d_with_norm6(128, 256, 256, 3,'down3')
        
        
        
        
        # nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        # self.encoder_step_1 = nn.Sequential(ConvWithNorms(self.input_size, 64, 3, 2, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1),
        #                                     ConvWithNorms(64, 64, 3, 1, 1))
        # self.encoder_step_2 = nn.Sequential(ConvWithNorms(64, 128, 3, 2, 1),
        #                                     ConvWithNorms(128, 128, 3, 1, 1),
        #                                     ConvWithNorms(128, 128, 3, 1, 1),
        #                                     ConvWithNorms(128, 128, 3, 1, 1),
        #                                     ConvWithNorms(128, 128, 3, 1, 1),
        #                                     ConvWithNorms(128, 128, 3, 1, 1))
        # self.encoder_step_3 = nn.Sequential(ConvWithNorms(128, 256, 3, 2, 1),
        #                                     ConvWithNorms(256, 256, 3, 1, 1),
        #                                     ConvWithNorms(256, 256, 3, 1, 1),
        #                                     ConvWithNorms(256, 256, 3, 1, 1),
        #                                     ConvWithNorms(256, 256, 3, 1, 1),
        #                                     ConvWithNorms(256, 256, 3, 1, 1))

        self.decoder_step1 = sp_Upsample(512, 256, 256, pseudo_image_dims,'down3')
        self.decoder_step2 = sp_Upsample(256, 128, 128, pseudo_image_dims,'down2')
        self.decoder_step3 = sp_Upsample(128, 2*self.input_size, 64, pseudo_image_dims,'down1')

        # self.decoder_step1 = UpsampleSkip(512, 256, 256)
        # self.decoder_step2 = UpsampleSkip(256, 128, 128)
        # # self.decoder_step3 = UpsampleSkip(128, 64, 64)
        # # self.decoder_step3 = UpsampleSkip(128, 128, 64)
        # self.decoder_step3 = UpsampleSkip(128, 2*self.input_size, 64)
        # self.decoder_step4 = nn.Conv2d(64, 64, 3, 1, 1)

        self.decoder_step4 = self.conv1 = spconv.SubMConv2d(64, 64, kernel_size=3, stride = 2, padding = 1)

    def forward(self, pc0_B, pc1_B) :

        pc0_F = self.encoder_step_1(pc0_B)
        pc0_L = self.encoder_step_2(pc0_F)
        pc0_R = self.encoder_step_3(pc0_L)

        pc1_F = self.encoder_step_1(pc1_B)
        pc1_L = self.encoder_step_2(pc1_F)
        pc1_R = self.encoder_step_3(pc1_L)

        Rstar = self.cat_sp(pc0_R, pc1_R)
        Lstar = self.cat_sp(pc0_L, pc1_L)
        Fstar = self.cat_sp(pc0_F, pc1_F)
        Bstar = self.cat_sp(pc0_B, pc1_B)

        # Rstar = torch.cat([pc0_R, pc1_R],
        #                   dim=1)  # torch.Size([1, 512, 64, 64])
        # Lstar = torch.cat([pc0_L, pc1_L],
        #                   dim=1)  # torch.Size([1, 256, 128, 128])
        # Fstar = torch.cat([pc0_F, pc1_F],
        #                   dim=1)  # torch.Size([1, 128, 256, 256])
        # Bstar = torch.cat([pc0_B, pc1_B],
        #                   dim=1)  # to become torch.Size([1, 64, 512, 512]) -> torch.Size([1, 128, 512, 512])

        S = self.decoder_step1(Rstar, Lstar)
        T = self.decoder_step2(S, Fstar)
        U = self.decoder_step3(T, Bstar)
        V = self.decoder_step4(U)

        return V








