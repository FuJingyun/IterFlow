import torch.nn as nn

from scripts.network.models.basic.Bi_util.gconv import SetConv
from scripts.network.models.basic.Bi_util.pointtransformer_seg import new_PTRefine


def pt_flow_trans(input, max_points):
    p_all, x = input.size()
    b_num = p_all // max_points
    # torch.Size([128, 8192*n])
    f = input.transpose(0,1)
    # torch.Size([128, n, 8192])
    f = f.view( x, b_num, p_all // b_num)
    f = f.permute(1,2,0)
    return f 


class FlotRefine(nn.Module):
    def __init__(self):
        super(FlotRefine, self).__init__()
        n = 32

        self.ref_conv1 = SetConv(3, n)
        self.ref_conv2 = SetConv(n, 2 * n)
        self.ref_conv3 = SetConv(2 * n, 4 * n)
        self.fc = nn.Linear(4 * n, 3)

    def forward(self, flow, graph):
        x = self.ref_conv1(flow, graph)
        x = self.ref_conv2(x, graph)
        x = self.ref_conv3(x, graph)
        x = self.fc(x)

        return flow + x
    


class new_pt_FlotRefine(nn.Module):
    def __init__(self):
        super(new_pt_FlotRefine, self).__init__()
        n = 32
        self.ref_pt = new_PTRefine()
        self.fc = nn.Linear(4 * n, 3)
        # self.fc = nn.Linear(n, 3)

    def forward(self, pxo, max_points):
        # logger = get_logger("pt_refine")
        x = self.ref_pt(pxo)
        x = pt_flow_trans(x, max_points)
        # logger.info('x:{}'.format(x.shape))
        x = self.fc(x)
        return  x
