"""
# Created: 2023-07-17 00:00
# Copyright (C) 2023-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# Description: Define the loss function for training.
"""
import torch
from assets.cuda.chamfer3D import nnChamferDis
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import knn_points
MyCUDAChamferDis = nnChamferDis()
import pointops_cuda
from scripts.raflow_utils.raflow_util import index_points_group

# NOTE(Qingwen 24/07/06): squared, so it's sqrt(4) = 2m, in 10Hz the vel = 20m/s ~ 72km/h
# If your scenario is different, may need adjust this TRUNCATED to 80-120km/h vel.
TRUNCATED_DIST = 4 # 4

def seflowLoss(res_dict, timer=None):
    pc0_label = res_dict['pc0_labels']
    pc1_label = res_dict['pc1_labels']

    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow

    unique_labels = torch.unique(pc0_label)
    pc0_dynamic = pc0[pc0_label > 0]
    pc1_dynamic = pc1[pc1_label > 0]
    # fpc1_dynamic = pseudo_pc1from0[pc0_label > 0]
    # NOTE(Qingwen): since we set THREADS_PER_BLOCK is 256
    have_dynamic_cluster = (pc0_dynamic.shape[0] > 256) & (pc1_dynamic.shape[0] > 256)

    # first item loss: chamfer distance
    # timer[5][1].start("MyCUDAChamferDis")
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    est_dist0, est_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc1from0, pc1)
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    raw_idx0 = raw_idx0.type(torch.long)
    chamfer_dis = torch.mean(est_dist0[est_dist0 <= TRUNCATED_DIST]) + torch.mean(est_dist1[est_dist1 <= TRUNCATED_DIST])
    # timer[5][1].stop()
    
    # second item loss: dynamic chamfer distance
    # timer[5][2].start("DynamicChamferDistance")
    dynamic_chamfer_dis = torch.tensor(0.0, device=est_flow.device)
    if have_dynamic_cluster:
        dynamic_chamfer_dis += MyCUDAChamferDis(pseudo_pc1from0[pc0_label>0], pc1_dynamic, truncate_dist=TRUNCATED_DIST)
    # timer[5][2].stop()

    # third item loss: exclude static points' flow
    # NOTE(Qingwen): add in the later part on label==0
    static_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    
    # fourth item loss: same label points' flow should be the same
    # timer[5][3].start("SameClusterLoss")
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)
    for label in unique_labels:
        mask = (pc0_label == label)
        if label == 0:
            # Eq. 6 in the paper
            static_cluster_loss += torch.linalg.vector_norm(est_flow[mask, :], dim=-1).mean()
        elif label > 0 and have_dynamic_cluster:
            cluster_id_flow = est_flow[mask, :]
            cluster_nnd = raw_dist0[mask]
            if cluster_nnd.shape[0] <= 0:
                continue

            # Eq. 8 in the paper
            sorted_idxs = torch.argsort(cluster_nnd, descending=True)
            sorted_idxs = sorted_idxs.type(torch.long)
            nearby_label = pc1_label[raw_idx0[mask][sorted_idxs]] # nonzero means dynamic in label
            non_zero_valid_indices = torch.nonzero(nearby_label > 0)
            # 找到pc1中最近的非0即非背景点
            if non_zero_valid_indices.shape[0] <= 0:
                continue
            max_idx = sorted_idxs[non_zero_valid_indices.squeeze(1)[0]]
            
            # Eq. 9 in the paper
            max_flow = pc1[raw_idx0[mask][max_idx]] - pc0[mask][max_idx]

            # Eq. 10 in the paper
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_id_flow - max_flow), dim=-1)))
    
    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean() # Eq. 11 in the paper
    elif have_dynamic_cluster:
        moved_cluster_loss = torch.mean(raw_dist0[raw_dist0 <= TRUNCATED_DIST]) + torch.mean(raw_dist1[raw_dist1 <= TRUNCATED_DIST])
    # timer[5][3].stop()

    res_loss = {
        'chamfer_dis': chamfer_dis,
        'dynamic_chamfer_dis': dynamic_chamfer_dis,
        'static_flow_loss': static_cluster_loss,
        'cluster_based_pc0pc1': moved_cluster_loss,
    }
    return res_loss





def PT_raLoss(res_dict):
    ra_pc0_dy = res_dict['ra_pc0_dy']
    pose_flow = res_dict['pose_flow']
    est_flow = res_dict['est_flow']
   
    all_loss = torch.tensor(0.0, device=est_flow.device)

    static_loss = torch.linalg.vector_norm((est_flow[~ra_pc0_dy, :] - pose_flow[~ra_pc0_dy, :]), dim=-1).mean()
    all_loss += static_loss
    
    return all_loss


def ra_chamferLoss(res_dict):

    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']
    ra_pc0_dy = res_dict['ra_pc0_dy']
    # pc0_vel = res_dict['ra_pc0_vel']
    # l2r = res_dict['l2r']
    # ra_pc1_dy = res_dict['ra_pc1_dy']

    est_flow = res_dict['est_flow']

    pseudo_pc1from0 = pc0 + est_flow
    all_loss = torch.tensor(0.0, device=est_flow.device)


    # first loss: chamfer distance
    # raw: pc0 to pc1, est: pseudo_pc1from0 to pc1, idx means the nearest index
    est_dist0, est_dist1, _, _ = MyCUDAChamferDis.disid_res(pseudo_pc1from0, pc1)
    raw_dist0, raw_dist1, raw_idx0, _ = MyCUDAChamferDis.disid_res(pc0, pc1)
    raw_idx0 = raw_idx0.type(torch.long)
    chamfer_dis = torch.mean(est_dist0[est_dist0 <= TRUNCATED_DIST]) + torch.mean(est_dist1[est_dist1 <= TRUNCATED_DIST])
    all_loss += chamfer_dis
    # timer[5][1].stop()
    

    # second loss: exclude static points' flow 
    static_loss = torch.linalg.vector_norm(est_flow[~ra_pc0_dy, :], dim=-1).mean()
    all_loss += static_loss

    # # third loss: vr loss
    # cur_dev = pc0.get_device()
    # ones = torch.ones((pc0.shape[0], 1), dtype=torch.float32, device=cur_dev)
    # homo_pc0 =  torch.cat([pc0,ones], dim=1) # N 4
    # homo_pc0_warp = torch.cat([pseudo_pc1from0,ones], dim=1) # N 4

    # homo_in_ra_pc0 = torch.mm(homo_pc0, l2r) # N 4   4 4
    # in_ra_pc0 = homo_in_ra_pc0[:,:3] 
    # homo_in_ra_pc0_warp = torch.mm(homo_pc0_warp, l2r)
    # in_ra_pc0_warp = homo_in_ra_pc0_warp[:,:3]
    # radar_flow_in_radar = in_ra_pc0_warp - in_ra_pc0 # N 3

    # pred_fr_a=torch.sum(radar_flow_in_radar*in_ra_pc0,dim=1)/(torch.norm(in_ra_pc0,dim=1))
    # diff_vel_a = torch.abs(pc0_vel * 0.1 - pred_fr_a)
    # veloLoss = torch.mean(diff_vel_a)
    # all_loss += veloLoss
 

    # # fourth item loss: same label points' flow should be the same
    # # timer[5][3].start("SameClusterLoss")
    # dy_pc0_chamfer = est_dist0[ra_pc0_dy]
    # dy_pc1_chamfer = est_dist1[ra_pc1_dy]

    # moved_loss = torch.mean(dy_pc0_chamfer[dy_pc0_chamfer <= TRUNCATED_DIST]) + torch.mean(dy_pc1_chamfer[dy_pc1_chamfer <= TRUNCATED_DIST])


    
    return all_loss






def dy_seg_Loss(res_dict):
    est_flow = res_dict['est_flow']
    scale_est_flow = torch.linalg.vector_norm(est_flow, dim=-1)
    # gt_flow = res_dict['gt_flow']
    track_id = res_dict['track_id']

    unique_labels = torch.unique(track_id)
    # have_dynamic_cluster = (est_flow.shape[0] > 256) 
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)

    for label in unique_labels:
        mask = (track_id == label)
        if label != 0 :
            cluster_scale_est_flow = scale_est_flow[mask]
            cluster_est_flow = est_flow[mask,:]
            # mean_flow = cluster_est_flow.mean(dim=0)
            # cluster_gt = gt_flow[mask,:]
            sorted_idxs = torch.argsort(cluster_scale_est_flow, descending=True)
            sorted_idxs = sorted_idxs.type(torch.long)
            max_flow = cluster_est_flow[sorted_idxs[0]]
            # max_flow = cluster_gt[sorted_idxs[0]]
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_est_flow - max_flow), dim=-1)))
            # moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_est_flow - mean_flow), dim=-1)))

    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean()

    return moved_cluster_loss


def dy_seg_Loss_mean(res_dict):
    est_flow = res_dict['est_flow']
    # scale_est_flow = torch.linalg.vector_norm(est_flow, dim=-1)
    # gt_flow = res_dict['gt_flow']
    track_id = res_dict['track_id']

    unique_labels = torch.unique(track_id)
    # have_dynamic_cluster = (est_flow.shape[0] > 256) 
    moved_cluster_loss = torch.tensor(0.0, device=est_flow.device)
    moved_cluster_norms = torch.tensor([], device=est_flow.device)

    for label in unique_labels:
        mask = (track_id == label)
        if label != 0 :
            # cluster_scale_est_flow = scale_est_flow[mask]
            cluster_est_flow = est_flow[mask,:]
            mean_flow = cluster_est_flow.mean(dim=0)
            # cluster_gt = gt_flow[mask,:]
            # sorted_idxs = torch.argsort(cluster_scale_est_flow, descending=True)
            # sorted_idxs = sorted_idxs.type(torch.long)
            # max_flow = cluster_est_flow[sorted_idxs[0]]
            # max_flow = cluster_gt[sorted_idxs[0]]
            # moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_est_flow - max_flow), dim=-1)))
            moved_cluster_norms = torch.cat((moved_cluster_norms, torch.linalg.vector_norm((cluster_est_flow - mean_flow), dim=-1)))

    if moved_cluster_norms.shape[0] > 0:
        moved_cluster_loss = moved_cluster_norms.mean()

    return moved_cluster_loss



def pc_distance(src, dst):

    N, _ = src.shape
    M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(1, 0))
    dist += torch.sum(src ** 2, -1).view(N, 1)
    dist += torch.sum(dst ** 2, -1).view(1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist




def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    dist = torch.maximum(dist,torch.zeros(dist.size()).cuda())
    return dist





def deflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']

    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    speed = gt.norm(dim=1, p=2) / 0.1
    # pts_loss = torch.norm(pred - gt, dim=1, p=2)
    pts_loss = torch.linalg.vector_norm(pred - gt, dim=-1)

    weight_loss = 0.0
    speed_0_4 = pts_loss[speed < 0.4].mean()
    speed_mid = pts_loss[(speed >= 0.4) & (speed <= 1.0)].mean()
    speed_1_0 = pts_loss[speed > 1.0].mean()

    if ~speed_1_0.isnan():
        weight_loss += speed_1_0
    if ~speed_0_4.isnan():
        weight_loss += speed_0_4
    if ~speed_mid.isnan():
        weight_loss += speed_mid
    return weight_loss

# ref from zeroflow loss class FastFlow3DDistillationLoss()
def zeroflowLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    mask_no_nan = (~gt.isnan() & ~pred.isnan() & ~gt.isinf() & ~pred.isinf())
    
    pred = pred[mask_no_nan].reshape(-1, 3)
    gt = gt[mask_no_nan].reshape(-1, 3)

    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    # gt_speed = torch.norm(gt, dim=1, p=2) * 10.0
    gt_speed = torch.linalg.vector_norm(gt, dim=-1) * 10.0
    
    mins = torch.ones_like(gt_speed) * 0.1
    maxs = torch.ones_like(gt_speed)
    importance_scale = torch.max(mins, torch.min(1.8 * gt_speed - 0.8, maxs))
    # error = torch.norm(pred - gt, dim=1, p=2) * importance_scale
    # 0.1 if s(p) < 0.4 m/s ; 1.0 if s(p) > 1.0 m/s ; 1.8s − 0.8 o.w.
    error = error * importance_scale
    return error.mean()

# ref from zeroflow loss class FastFlow3DSupervisedLoss()
def ff3dLoss(res_dict):
    pred = res_dict['est_flow']
    gt = res_dict['gt_flow']
    classes = res_dict['gt_classes']
    # error = torch.norm(pred - gt, dim=1, p=2)
    error = torch.linalg.vector_norm(pred - gt, dim=-1)
    is_foreground_class = (classes > 0) # 0 is background, ref: FOREGROUND_BACKGROUND_BREAKDOWN
    # 1 if p in Foreground ; 0.1 if p in Background
    background_scalar = is_foreground_class.float() * 0.9 + 0.1
    error = error * background_scalar
    return error.mean()


def maskLoss(est_score, gt_mask):
    loss = 0.0
    float_label = (gt_mask).type(torch.float32)
    BCEloss = torch.nn.BCELoss()

    loss += BCEloss(est_score[gt_mask], float_label[gt_mask])
    loss += BCEloss(est_score[~gt_mask], float_label[~gt_mask])

    loss = loss/2
    #loss = self.loss_obj(mseg_pre.squeeze(1), mseg_gt)
    return loss



def warpedLoss(res_dict, dist_threshold=4):
    pred = res_dict['est_flow']
    
    pc0 = res_dict['pc0']
    pc1 = res_dict['pc1']
    
    warped_pc = pc0 + pred
    target_pc = pc1
    
    # assert warped_pc.ndim == 3, f"warped_pc.ndim = {warped_pc.ndim}, not 3; shape = {warped_pc.shape}"
    # assert target_pc.ndim == 3, f"target_pc.ndim = {target_pc.ndim}, not 3; shape = {target_pc.shape}"
    if warped_pc.ndim == 2:
        warped_pc = warped_pc.unsqueeze(0)
        target_pc = target_pc.unsqueeze(0)
    loss = 0

    if dist_threshold is None:
        loss += chamfer_distance(warped_pc, target_pc,
                                 point_reduction="mean")[0].sum()
        loss += chamfer_distance(target_pc, warped_pc,
                                 point_reduction="mean")[0].sum()
        return loss
    
    
    # Compute min distance between warped point cloud and point cloud at t+1.
    warped_to_target_knn = knn_points(p1=warped_pc, p2=target_pc, K=1)
    warped_to_target_distances = warped_to_target_knn.dists[0]
    target_to_warped_knn = knn_points(p1=target_pc, p2=warped_pc, K=1)
    target_to_warped_distances = target_to_warped_knn.dists[0]
    # Throw out distances that are too large (beyond the dist threshold).
    loss += warped_to_target_distances[warped_to_target_distances < dist_threshold].mean()
    loss += target_to_warped_distances[target_to_warped_distances < dist_threshold].mean()

    return {'loss': loss}



def pt_loss(res_dict):
    est_flow = res_dict['est_flow']
    true_flow = res_dict['gt_flow']
    gamma=0.8
    n_predictions = len(est_flow)
    flow_loss = 0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = torch.mean(torch.abs(est_flow[i]-true_flow))
        flow_loss += i_weight * i_loss

    return flow_loss



def FLOT_loss(est_flow, true_flow):
    """
    Compute training loss.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    loss : torch.Tensor
        Training loss for current batch.

    """

    # mask = batch["ground_truth"][0][..., 0]
    # true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    # error = error[mask > 0]
    loss = torch.mean(torch.abs(error))

    return loss





