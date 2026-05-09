import torch
import torch.nn as nn

from scripts.network.models.basic.PT_util.flot.gconv import DMR_SetConv, SetConv
from scripts.network.models.basic.PT_util.flot.graph import Graph


class FlotEncoder(nn.Module):
    def __init__(self, input_dim, num_neighbors = 8):
        super(FlotEncoder, self).__init__()
        n = 32
        self.num_neighbors = num_neighbors

        self.feat_conv1 = SetConv(input_dim, n) # input_dim=3
        self.feat_conv2 = SetConv(n, 2 * n)
        self.feat_conv3 = SetConv(2 * n, 4 * n)

    def forward(self, pc, fea):
        graph = Graph.construct_graph(pc, self.num_neighbors)
        x = self.feat_conv1(torch.cat((pc,fea),dim=-1), graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        x = x.transpose(1, 2).contiguous()

        return x, graph
    

class DMR_FlotRefine(nn.Module):
    def __init__(self):
        super(DMR_FlotRefine, self).__init__()
        n = 16 # 32
        self.num_neighbors = 4 

        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = nn.Linear(4 * n, 3)

    def forward(self, pc, flow):
        graph = Graph.construct_graph(pc, self.num_neighbors)
        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x
    


class DMR_FlotEncoder(nn.Module):
    def __init__(self, input_dim, num_neighbors = 8):
        super(DMR_FlotEncoder, self).__init__()
        n = 32
        self.num_neighbors = num_neighbors

        self.feat_conv1 = DMR_SetConv(input_dim, n) # input_dim=3
        self.feat_conv2 = DMR_SetConv(n, 2 * n)
        self.feat_conv3 = DMR_SetConv(2 * n, 4 * n)

    def forward(self, pc, fea):
        graph = Graph.construct_graph(pc, self.num_neighbors)
        # x = self.feat_conv1(torch.cat((pc,fea),dim=-1), graph)
        x = self.feat_conv1(fea, graph)
        x = self.feat_conv2(x, graph)
        x = self.feat_conv3(x, graph)
        x = x.transpose(1, 2).contiguous()

        return x
    


def sinkhorn(feature1, feature2, pcloud1, pcloud2, epsilon, gamma, max_iter):
    """
    Sinkhorn algorithm

    Parameters
    ----------
    feature1 : torch.Tensor
        Feature for points cloud 1. Used to computed transport cost. 
        Size B x N x C.
    feature2 : torch.Tensor
        Feature for points cloud 2. Used to computed transport cost. 
        Size B x M x C.
    pcloud1 : torch.Tensor
        Point cloud 1. Size B x N x 3.
    pcloud2 : torch.Tensor
        Point cloud 2. Size B x M x 3.
    epsilon : torch.Tensor
        Entropic regularisation. Scalar.
    gamma : torch.Tensor
        Mass regularisation. Scalar.
    max_iter : int
        Number of unrolled iteration of the Sinkhorn algorithm.

    Returns
    -------
    torch.Tensor
        Transport plan between point cloud 1 and 2. Size B x N x M.
    """

    # Squared l2 distance between points points of both point clouds
    distance_matrix = torch.sum(pcloud1 ** 2, -1, keepdim=True)
    distance_matrix = distance_matrix + torch.sum(
        pcloud2 ** 2, -1, keepdim=True
    ).transpose(1, 2)
    distance_matrix = distance_matrix - 2 * torch.bmm(pcloud1, pcloud2.transpose(1, 2))
    # Force transport to be zero for points further than 10 m apart
    support = (distance_matrix < 10 ** 2).float()

    # Transport cost matrix
    feature1 = feature1 / torch.sqrt(torch.sum(feature1 ** 2, -1, keepdim=True) + 1e-8)
    feature2 = feature2 / torch.sqrt(torch.sum(feature2 ** 2, -1, keepdim=True) + 1e-8)
    C = 1.0 - torch.bmm(feature1, feature2.transpose(1, 2))

    # Entropic regularisation
    K = torch.exp(-C / epsilon) * support

    # Early return if no iteration (FLOT_0)
    if max_iter == 0:
        return K

    # Init. of Sinkhorn algorithm
    power = gamma / (gamma + epsilon)
    a = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob1 = (
        torch.ones(
            (K.shape[0], K.shape[1], 1), device=feature1.device, dtype=feature1.dtype
        )
        / K.shape[1]
    )
    prob2 = (
        torch.ones(
            (K.shape[0], K.shape[2], 1), device=feature2.device, dtype=feature2.dtype
        )
        / K.shape[2]
    )

    # Sinkhorn algorithm
    for _ in range(max_iter):
        # Update b
        KTa = torch.bmm(K.transpose(1, 2), a)
        b = torch.pow(prob2 / (KTa + 1e-8), power)
        # Update a
        Kb = torch.bmm(K, b)
        a = torch.pow(prob1 / (Kb + 1e-8), power)

    # Transportation map
    T = torch.mul(torch.mul(a, K), b.transpose(1, 2))

    return T