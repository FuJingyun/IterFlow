import torch
import torch_scatter



def shift_bev_grids(bev_grids, shift_tensor, grid_size, cur_device, fix_dtype):
    """
    bev_coors : param voxel_grids: [N, 2], where N represents the number of non-sparse pillars
    shift_tensor: param shift_tensor: [9, 2]
    grid_size: param grid_size: [2]  
    cur_device : param current device
    return: shifted_voxel_grids: [9, N, 2]
    """
    shifted_bev_tensors = []
    fixed_shift_offset_ten = []
    # grid_size: [512,128] 
    # voxel_size:[0.4, 0.1, 6]
    # bs_grid_size = torch.tensor([1] + grid_size, device=cur_device)
    bs_grid_size = torch.tensor([grid_size[0]+1, grid_size[1]+1], device=cur_device)
    for shift in shift_tensor:
        shifted_bev_grids = (bev_grids + torch.tensor(shift, device=cur_device)) % bs_grid_size
        shifted_bev_tensors.append(shifted_bev_grids)

        cur_shift = torch.tensor(shift, dtype= fix_dtype, device=cur_device)
        fixed_shift_offset_ten.append(cur_shift)
    return shifted_bev_tensors, fixed_shift_offset_ten



def return_tensor_index(value, t):
    """
    Prerequisite: each value in tensor t is unique. That is, each value has at most 1 time appearance in t.
    Check if value is in tensor t. If True, return the index of value in tensor t; else, return -1
    :param value: Tensor(Q,M) a Q-dimension vector of M elements
    :param t: Tensor(N,M) a two-dimension tensor of N vectors, each vector has M elements
    :return: Tensor(Q,) a one-dimension vector of Q elements, each element is the  index of value in t 
    at the first dimension
    """
    Q = value.shape[0]
    N = t.shape[0]
    t_exp = t.unsqueeze(0).expand(Q, t.shape[0], t.shape[1]) # (Q, N, M)
    # value = value.unsqueeze(1) # (Q, 1, M)
    value = value.unsqueeze(1).expand(Q, t.shape[0], t.shape[1]) # (Q, 1, M)
    # value维度每一个元素都相等才判定为 True, N 个里最多匹配上一个
    res = torch.all(t_exp == value, dim=2) # (Q, N)
    ones = torch.ones((Q, 1), dtype=torch.bool, device=value.device)
    res_with_ones = torch.concat((res, ones), dim=1) # (Q, N+1)
    res_index = torch.nonzero(res_with_ones) # (Q + Q_num_in_t, 2)
    unq_res_index, unq_inv_res_index =torch.unique(res_index[:,0], return_inverse=True, dim=0)
    res_index = torch.masked_fill(res_index, res_index == N, -1) # (Q + Q_num_in_t, 2)  
    select_ind, _ = torch_scatter.scatter_max(res_index[:, 1], unq_inv_res_index, dim=0) 
    
    return select_ind
