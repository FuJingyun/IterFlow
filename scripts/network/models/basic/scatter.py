import torch
import torch.nn as nn

class PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        # output_shape [128 512]
        self.output_shape = output_shape
        self.ny = output_shape[0] # 128
        self.nx = output_shape[1] # 512
        self.in_channels = in_channels
        self.fp16_enabled = False

    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)

    # # voxel_coors : [  0, 460,  84] Z Y X
    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)

        # self.ny = 128
        # self.nx = 512
        indices = coors[:, 1] * self.nx + coors[:, 2]

        indices = indices.long()
        voxels = voxel_features.t()
        # Now scatter the blob back to the canvas.
        canvas[:, indices] = voxels
        # Undo the column stacking to final 4-dim tensor
        # torch.Size([1, 32, 128, 512])
        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny,
                                         self.nx)

        return batch_canvas



class my_PointPillarsScatter(nn.Module):
    """Point Pillar's Scatter.

    Converts learned features from dense tensor to sparse pseudo image.

    Args:
        in_channels (int): Channels of input features.
        output_shape (list[int]): Required output shape of features.
    """

    def __init__(self, in_channels, output_shape):
        super().__init__()
        # output_shape [128 512]
        # voxel_size: [0.1, 0.4, 6] # X Y Z
        # point_cloud_range: [0, -25.6, -3, 51.2, 25.6, 3]
        # grid_feature_size = [abs(int((point_cloud_range[0] - point_cloud_range[3]) / voxel_size[0])),
        #  abs(int((point_cloud_range[1] - point_cloud_range[4]) / voxel_size[1]))]
        self.output_shape = output_shape
        self.nx = output_shape[0] # 512
        self.ny = output_shape[1] # 128
        
        self.in_channels = in_channels
        self.fp16_enabled = False

    def forward(self, voxel_features, coors, batch_size=None):
        """Foraward function to scatter features."""
        # TODO: rewrite the function in a batch manner
        # no need to deal with different batch cases
        if batch_size is not None:
            return self.forward_batch(voxel_features, coors, batch_size)
        else:
            return self.forward_single(voxel_features, coors)


    def forward_single(self, voxel_features, coors):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel.
        """
        # Create the canvas for this sample
        canvas = torch.zeros(
            self.in_channels,
            self.nx * self.ny,
            dtype=voxel_features.dtype,
            device=voxel_features.device)


        # my EDIT
        canvas = canvas.view(self.in_channels, self.ny, self.nx)

        voxels = voxel_features.t()
        # point : Z Y X
        canvas[:, coors[:, 1].long(), coors[:, 2].long()] = voxels

        canvas = canvas.view(1, self.in_channels, self.ny, self.nx)
        return canvas

    def forward_batch(self, voxel_features, coors, batch_size):
        """Scatter features of single sample.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, M, C).
            coors (torch.Tensor): Coordinates of each voxel in shape (N, 4).
                The first column indicates the sample ID.
            batch_size (int): Number of samples in the current batch.
        """
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            # indices = this_coors[:, 2] * self.nx + this_coors[:, 3]
            # indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            # canvas[:, indices] = voxels
            # point : Z Y X
            canvas = canvas.view(self.in_channels, self.ny, self.nx)

            canvas[:, this_coors[:, 1].long(), this_coors[:, 2].long()] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, in_channels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)

        return batch_canvas

