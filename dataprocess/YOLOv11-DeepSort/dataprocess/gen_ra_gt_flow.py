import numpy as np

import os
import torch
import numba

import multiprocessing
from multiprocessing import Pool, current_process
from typing import Optional   
from tqdm import tqdm
from scipy.spatial.transform import Rotation
from av2.geometry.se3 import SE3
import h5py
import pickle,time
from pathlib import Path
import copy
from sklearn.neighbors import BallTree

# import sys
from dataprocess.vod.configuration import ra_KittiLocations
from dataprocess.vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation, project_3d_to_2d
# from dataprocess.vod.frame.transformations import canvas_crop
from dataprocess.preprocess.radar_filter import  in_region, remove_height 


os.environ["OMP_NUM_THREADS"] = "1"
VALID_VOD_CLASS = ['Cyclist', 'rider', 'Car', 'moped_scooter', \
                   'Pedestrian', 'ride_other', 'truck', 'motor', \
                    'ride_uncertain', 'vehicle_other']
VOD_TO_ARGOVERSE = { 
'Cyclist': 4, # BICYCLIST 
'rider': 4, # "BICYCLIST'     
'Car': 19, # "REGULAR VEHICLE"  
'moped_scooter': 15, # "MOTORCYCLIST"  
'Pedestrian': 17,  #"PEDESTRIAN"  
'ride_other': 15,  # "MOTORCYCLE  
'truck': 25, # "TRUCK"   
'motor': 15,  # "MOTORCYCLE  
'ride_uncertain': 4, # "BICYCLIST'   
'vehicle_other': 19  #"REGULAR VEHICLE"   
 }

IMG_HEIGHT = 1216
IMG_WIDTH = 1936

# Based on training set statistics
MEAN_BOX_HALF_DIAGONAL = { 
'Cyclist': 1.37, # BICYCLIST 
'rider': 0.98, # "BICYCLIST'     
'Car': 2.42, # "REGULAR VEHICLE"  
'moped_scooter': 1.25, # "MOTORCYCLIST"  
'Pedestrian': 0.95,  #"PEDESTRIAN"  
'ride_other': 0.96,  # "MOTORCYCLE  
'truck': 3.79, # "TRUCK"   
'motor': 1.47,  # "MOTORCYCLE  
'ride_uncertain': 0.95, # "BICYCLIST'   
'vehicle_other': 7.53  #"REGULAR VEHICLE"   
 }

MAX_BOX_HALF_DIAGONAL = { 
'Cyclist': 1.72, # BICYCLIST 
'rider': 1.19, # "BICYCLIST'     
'Car': 3.48, # "REGULAR VEHICLE"  
'moped_scooter': 1.52, # "MOTORCYCLIST"  
'Pedestrian': 1.21,  #"PEDESTRIAN"  
'ride_other': 1.49,  # "MOTORCYCLE  
'truck': 4.20, # "TRUCK"   
'motor': 1.60,  # "MOTORCYCLE  
'ride_uncertain': 0.95, # "BICYCLIST'   
'vehicle_other': 9.06  #"REGULAR VEHICLE"   
 }
 

# TO DO 1 : mode
# TO DO 2 : The location of the files
mode = "train" # train val
clip_dir = "Dir to CLIP file/" + mode + "/" # my_new_clips my_clips cmflow_clips
dlo_pose_dir = "Dir to poses file/" + mode + "/" # my_new_dlo_poses my_dlo_poses cmflow_dlo_poses
kitti_locations = ra_KittiLocations(root_dir="Dir to DATASET")
SF_save_path = "Dir to output file"

BBOX_LIDAR_EXPANSION_XY = 0.2
BBOX_RADAR_EXPANSION_XY = 0.2
BBOX_RADAR_EXPANSION_Z = 0.2


def create_reading_index(data_dir: Path):
    start_time = time.time()
    data_index = []
    for file_name in tqdm(os.listdir(data_dir), ncols=100, desc='Create reading index'):
        if not file_name.endswith(".h5"):
            continue
        scene_id = file_name.split(".")[0]
        timestamps = []
        with h5py.File(data_dir + file_name, 'r') as f:
            timestamps.extend(f.keys())
        timestamps.sort(key=lambda x: int(x)) # make sure the timestamps are in order
        for timestamp in timestamps:
            data_index.append([scene_id, timestamp])
    with open(data_dir+'index_total.pkl', 'wb') as f:
        pickle.dump(data_index, f)
        print(f"Create reading index Successfully, cost: {time.time() - start_time:.2f} s")

def nearest_neighbor_search(x, data):
    tree = BallTree(data)
    dist, ind = tree.query(x.reshape(1,-1), k=1)
    # return ind[0][0], dist[0][0]
    return ind, dist

def rotation_3d_in_axis(
    points,
    angles,
    axis: int = 0,
    return_mat: bool = False,
    clockwise: bool = False
):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray or Tensor): Points with shape (N, M, 3).
        angles (np.ndarray or Tensor or float): Vector of angles with shape
            (N, ).
        axis (int): The axis to be rotated. Defaults to 0.
        return_mat (bool): Whether or not to return the rotation matrix
            (transposed). Defaults to False.
        clockwise (bool): Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: When the axis is not in range [-3, -2, -1, 0, 1, 2], it
            will raise ValueError.

    Returns:
        Tuple[np.ndarray, np.ndarray] or Tuple[Tensor, Tensor] or np.ndarray or
        Tensor: Rotated points with shape (N, M, 3) and rotation matrix with
        shape (N, 3, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 and \
        points.shape[0] == angles.shape[0], 'Incorrect shape of points ' \
        f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    points = torch.from_numpy(points)
    angles = torch.from_numpy(angles)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, zeros, -rot_sin]),
                torch.stack([zeros, ones, zeros]),
                torch.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = torch.stack([
                torch.stack([rot_cos, rot_sin, zeros]),
                torch.stack([-rot_sin, rot_cos, zeros]),
                torch.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = torch.stack([
                torch.stack([ones, zeros, zeros]),
                torch.stack([zeros, rot_cos, rot_sin]),
                torch.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(
                f'axis should in range [-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, rot_sin]),
            torch.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = torch.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = torch.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                    points[i, 0] * normal_vec[j, k, 0] +
                    points[i, 1] * normal_vec[j, k, 1] +
                    points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def surface_equ_3d(polygon_surfaces):
    """

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
        polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons, ), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)


def corner_to_surfaces_3d(corners):
    corners = corners.numpy()
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


def corners_nd(dims, origin=0.5):
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    corners = corners_nd(dims, origin=origin)
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


def limit_period(val,
                 offset: float = 0.5,
                 period: float = np.pi):
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val


def box_cam2lidar(cam_bbox_3d, cam2lidar):
    x, y, z, x_size, y_size, z_size, yaw = cam_bbox_3d
    lidar2cam = np.linalg.inv(cam2lidar)

    lidar_xyz = np.array([x, y, z, 1]) @ lidar2cam.T
    lidar_xyz = lidar_xyz[:3]

    yaw = -yaw - np.pi / 2
    yaw = limit_period(yaw, period=np.pi * 2)

    return np.array([*lidar_xyz, x_size, z_size, y_size, yaw])



def box_cam2radar(cam_bbox_3d, cam2radar, t_radar_lidar):
    x, y, z, x_size, y_size, z_size, yaw = cam_bbox_3d
    radar2cam = np.linalg.inv(cam2radar)

    radar_xyz = np.array([x, y, z, 1]) @ radar2cam.T
    radar_xyz = radar_xyz[:3]

    angle = [0, 0, -(yaw + np.pi / 2)]
    rot_m = Rotation.from_euler('XYZ', angle).as_matrix()
    rot_m = t_radar_lidar[:3,:3] @ rot_m
    yaw = Rotation.from_matrix(rot_m)
    final_yaw = yaw.as_euler('XYZ') # 转换回欧拉角

    return np.array([*radar_xyz, x_size, z_size, y_size, final_yaw[2]])


def pcl_to_uvs(point_cloud, t_camera_pcl, camera_projection_matrix):
    point_homo = np.hstack((point_cloud[:, :3],
                            np.ones((point_cloud.shape[0], 1),
                                    dtype=np.float32)))

    points_camera_frame = homogeneous_transformation(point_homo, transform=t_camera_pcl)

    uvw = camera_projection_matrix.dot(points_camera_frame.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T

    return uvs


def filt_points_in_fov(pc_data, transforms, sensor):

    pc_h = np.concatenate((pc_data[:,0:3],np.ones((pc_data.shape[0],1))),axis=1)
    if sensor == 'radar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_radar)
    if sensor == 'lidar':
        pc_cam = homogeneous_transformation(pc_h, transforms.t_camera_lidar)
    uvs = project_3d_to_2d(pc_cam,transforms.camera_projection_matrix)
    filt_uv = np.logical_and(np.logical_and(uvs[:,0]>0, uvs[:,0]<=IMG_WIDTH),\
         np.logical_and(uvs[:,1]>0, uvs[:,1]<=IMG_HEIGHT))
    indices = np.argwhere(filt_uv).flatten()

    return indices


def big_v_r_compensated_filter(radar):
    big_error_threshold = 15
    moving_threshold = 0.1

    v_r_compensated = radar[:,5]
    moving_v_r_compensated_mask = np.absolute(v_r_compensated) > moving_threshold

    if moving_v_r_compensated_mask.sum()>0:
        moving_v_r_compensated = v_r_compensated[moving_v_r_compensated_mask]
        moving_v_r_compensated = np.absolute(moving_v_r_compensated)
        
        big_v_r_compensated = np.percentile(moving_v_r_compensated, 50)
        big_v_r_compensated_mask = moving_v_r_compensated > big_v_r_compensated
        if big_v_r_compensated_mask.sum()>0:
            big_moving_v_r_compensated = moving_v_r_compensated[big_v_r_compensated_mask]
            mean_big_v_r_compensated = np.mean(big_moving_v_r_compensated)
            valid_v_r_compensated_mask = np.absolute(v_r_compensated) < mean_big_v_r_compensated + big_error_threshold
            output_radar = radar[valid_v_r_compensated_mask]
            return output_radar
        else:     
            return radar


    else:
        return radar


    

    



def process_log(data_dir: Path, log_id: str, output_dir: Path, n: Optional[int] = None) :
    def create_group_data(group, pose, radar_pc, lidar_pc,\
                        radar_id, radar_flow_category, \
                    
                        radar_flow=None, radar_flow_valid=None, \
                        ego_motion=None,  radar_motion_mask=None,\
                        clean_radar_mask=None):
        group.create_dataset('pose', data=pose.astype(np.float32))
        group.create_dataset('radar_pc', data=radar_pc.astype(np.float32))
        group.create_dataset('lidar_pc', data=lidar_pc.astype(np.float32))
        # Track id
        group.create_dataset('radar_id', data=radar_id.astype(np.int16))
        group.create_dataset('radar_flow_category', data=radar_flow_category.astype(np.uint8))
        
        if radar_flow is not None:
            # ground truth flow information
            group.create_dataset('radar_flow', data=radar_flow.astype(np.float32))
            group.create_dataset('radar_flow_valid', data=radar_flow_valid.astype(bool))
            group.create_dataset('ego_motion', data=ego_motion.astype(np.float32))
            group.create_dataset('radar_motion_mask', data=radar_motion_mask.astype(bool))
            group.create_dataset('clean_radar_mask', data=clean_radar_mask.astype(bool))


    # start for VOD
    clip = open(os.path.join(data_dir, log_id), 'r').readlines()
    start_idx = int(clip[0])
    end_idx = int(clip[-1])

    dlo_poses = open(os.path.join(dlo_pose_dir, log_id), 'r').readlines()
    offset = end_idx + 1 - start_idx - len(dlo_poses)

    timestamps = range(start_idx + offset, end_idx)

    with h5py.File(output_dir + log_id.replace('.txt', '')+ ".h5", 'a') as f:
        for cnt, ts0 in enumerate(timestamps):
            group = f.create_group(str(ts0))

            # Last Frame
            if cnt == len(timestamps) - 1:
                cur_frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number='%05d' % ts0)
                cur_frame_trans = FrameTransformMatrix(cur_frame_data)

                raw_pose_data = dlo_poses[ts0 - offset - start_idx]
                pose_data = [float(data) for data in raw_pose_data.split(' ')]
                ego_rotation = Rotation.from_quat([*pose_data[3:]]).as_matrix()
                ego_translation = np.array([*pose_data[:3]], dtype=np.float32)
                # pose0
                cur_pose = SE3(rotation=ego_rotation, translation=ego_translation)

                # Lidar [:,4]
                cur_lidar_pts = cur_frame_data.lidar_data
                
                cur_ground_label = cur_frame_data.ground_label
                cur_lidar_ground_mask = cur_ground_label[:cur_lidar_pts.shape[0]]                
                cur_lidar_valid = cur_lidar_pts[~cur_lidar_ground_mask,:]

                lidar_xyz = cur_lidar_valid[:,:3]
                lidar_xyz = np.hstack((lidar_xyz,
                        np.ones((lidar_xyz.shape[0], 1),
                                dtype=np.float32)))
                lidar_transformed_homo = homogeneous_transformation(lidar_xyz, transform=cur_frame_trans.t_radar_lidar)
                cur_lidar_valid[:,:3] = lidar_transformed_homo[:,:3]

                cur_lidar_valid = in_region(cur_lidar_valid)
                in_img_mask = filt_points_in_fov(cur_lidar_valid, cur_frame_trans, 'radar')
                cur_lidar_valid = cur_lidar_valid[in_img_mask]

                # Radar [:,7]
                cur_radar_pts = cur_frame_data.radar_data
                cur_radar_valid = cur_radar_pts               

                cur_radar_valid = remove_height(cur_radar_valid) # in_region
                cur_radar_valid = big_v_r_compensated_filter(cur_radar_valid)
                in_img_mask = filt_points_in_fov(cur_radar_valid, cur_frame_trans, 'radar')
                cur_radar_valid = cur_radar_valid[in_img_mask]


                cur_labels = []
                cur_box_centers = []
                cur_frame_data_raw_labels, cur_frame_data_lable_flag = cur_frame_data.raw_labels
                
                for raw_label in cur_frame_data_raw_labels:
                    act_line = raw_label.split()
                    if cur_frame_data_lable_flag:
                        label, obj_id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
                    else:
                        label, obj_id, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
                 
                    if (label in VALID_VOD_CLASS):
                        h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
                        y = y-0.5*h
                        obj_id = int(obj_id)
                        cur_labels.append({ 'box_class': label, 
                                            'label_class': int(VOD_TO_ARGOVERSE[label]), 
                                            'obj_id': obj_id,
                                            'h': h,
                                            'w': w,
                                            'l': l,
                                            'x': x,
                                            'y': y,
                                            'z': z,
                                            'rotation': rot,
                                            'score': score})


                cur_valid_radar_classes = np.zeros(cur_radar_valid.shape[0], dtype=np.uint8)
                cur_radar_id = np.zeros(cur_radar_valid.shape[0], dtype=np.int16)

                for cur_label in cur_labels:
                    obj_id = cur_label['obj_id']
                    cur_cam_bbox_3d = np.array([cur_label['x'],
                                                cur_label['y'], 
                                                cur_label['z'],
                                                cur_label['l'],
                                                cur_label['h'],
                                                cur_label['w'],
                                                cur_label['rotation']])
                    
                    # To Radar cooridinate                   
                    cur_radar_bbox_3d = box_cam2radar(cur_cam_bbox_3d, cur_frame_trans.t_camera_radar, cur_frame_trans.t_radar_lidar)
                    # ([*radar_xyz, l , w , h , yaw]) 
                    
                    find_radar_pts_in_bbox_3d = copy.deepcopy(cur_radar_bbox_3d) 
                    find_radar_pts_in_bbox_3d[2] -= 0.5 * cur_label['h']  
                    find_radar_pts_in_bbox_3d[3] += BBOX_RADAR_EXPANSION_XY
                    find_radar_pts_in_bbox_3d[4] += BBOX_RADAR_EXPANSION_XY
                    find_radar_pts_in_bbox_3d[5] += BBOX_RADAR_EXPANSION_Z

                    in_bbox_radar_indices = points_in_rbbox(cur_radar_valid, find_radar_pts_in_bbox_3d[None])
                    radar_pts_in_box = cur_radar_valid[in_bbox_radar_indices[:, 0]]
                    cur_valid_radar_classes[in_bbox_radar_indices[:, 0]] = cur_label['label_class']
                    cur_radar_id[in_bbox_radar_indices[:, 0]] = obj_id

                create_group_data(group, cur_pose.transform_matrix.astype(np.float32), 
                                    cur_radar_valid, cur_lidar_valid, cur_radar_id, cur_valid_radar_classes)
                
            else:          
                cur_frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number='%05d' % ts0)
                cur_frame_trans = FrameTransformMatrix(cur_frame_data)

                next_frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number='%05d' % (ts0 + 1))
                next_frame_trans = FrameTransformMatrix(next_frame_data)

                raw_pose_data = dlo_poses[ts0 - offset - start_idx]
                pose_data = [float(data) for data in raw_pose_data.split(' ')]
                ego_rotation = Rotation.from_quat([*pose_data[3:]]).as_matrix()
                ego_translation = np.array([*pose_data[:3]], dtype=np.float32)
                # pose0
                cur_pose = SE3(rotation=ego_rotation, translation=ego_translation)

                # Source Lidar [:,4]
                cur_lidar_pts = cur_frame_data.lidar_data
                
                cur_ground_label = cur_frame_data.ground_label
                cur_lidar_ground_mask = cur_ground_label[:cur_lidar_pts.shape[0]]                
                cur_lidar_valid = cur_lidar_pts[~cur_lidar_ground_mask,:]

                lidar_xyz = cur_lidar_valid[:,:3]
                lidar_xyz = np.hstack((lidar_xyz,
                        np.ones((lidar_xyz.shape[0], 1),
                                dtype=np.float32)))
                lidar_transformed_homo = homogeneous_transformation(lidar_xyz, transform=cur_frame_trans.t_radar_lidar)
                cur_lidar_valid[:,:3] = lidar_transformed_homo[:,:3]

                cur_lidar_valid = in_region(cur_lidar_valid)
                in_img_mask = filt_points_in_fov(cur_lidar_valid, cur_frame_trans, 'radar')
                cur_lidar_valid = cur_lidar_valid[in_img_mask]
              

                # Source Radar [:,7]
                cur_radar_pts = cur_frame_data.radar_data
                cur_radar_valid = cur_radar_pts

                cur_radar_valid = remove_height(cur_radar_valid) # in_region
                cur_radar_valid = big_v_r_compensated_filter(cur_radar_valid)
                in_img_mask = filt_points_in_fov(cur_radar_valid, cur_frame_trans, 'radar')
                cur_radar_valid = cur_radar_valid[in_img_mask]


                # Next Pose
                raw_pose_data = dlo_poses[ts0 - offset - start_idx + 1]
                pose_data = [float(data) for data in raw_pose_data.split(' ')]
                ego_rotation = Rotation.from_quat([*pose_data[3:]]).as_matrix()
                ego_translation = np.array([*pose_data[:3]], dtype=np.float32)
                next_pose = SE3(rotation=ego_rotation, translation=ego_translation)

                # Next Radar [:,7]
                next_radar_pts = next_frame_data.radar_data
                next_radar_valid = next_radar_pts

                next_radar_valid = remove_height(next_radar_valid) 
                next_radar_valid = big_v_r_compensated_filter(next_radar_valid)
                in_img_mask = filt_points_in_fov(next_radar_valid, next_frame_trans, 'radar')
                next_radar_valid = next_radar_valid[in_img_mask]

                # Convert to float32s
                ego1_SE3_ego0 = next_pose.inverse().compose(cur_pose)
                ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
                ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)

                # Rigid flow
                radar_flow = ego1_SE3_ego0.transform_point_cloud(cur_radar_valid[:, :3]) - cur_radar_valid[:, :3]
                radar_flow = radar_flow.astype(np.float32)
                rigid_radar_flow = copy.deepcopy(radar_flow)


                cur_labels = []
                cur_box_centers = []
                cur_frame_data_raw_labels, cur_frame_data_lable_flag = cur_frame_data.raw_labels
                
                for raw_label in cur_frame_data_raw_labels:
                    act_line = raw_label.split()
                    if cur_frame_data_lable_flag:
                        label, obj_id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
                    else:
                        label, obj_id, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line

                    if (label in VALID_VOD_CLASS):
                        h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
                        y = y-0.5*h
                        obj_id = int(obj_id)

                        cur_labels.append({ 'box_class': label, 
                                            'label_class': int(VOD_TO_ARGOVERSE[label]), # label
                                            'obj_id': obj_id,
                                            'h': h,
                                            'w': w,
                                            'l': l,
                                            'x': x,
                                            'y': y,
                                            'z': z,
                                            'rotation': rot,
                                            'score': score})
                        cur_cam_bbox_3d = np.array([x,y,z,l,h,w,rot])              
                        cur_radar_bbox_3d = box_cam2radar(cur_cam_bbox_3d, cur_frame_trans.t_camera_radar, cur_frame_trans.t_radar_lidar)
                        cur_box_centers.append([cur_radar_bbox_3d[0], cur_radar_bbox_3d[1], cur_radar_bbox_3d[2], obj_id, MEAN_BOX_HALF_DIAGONAL[label]])

                cur_box_centers = np.array(cur_box_centers) # [numbox, 5]


                next_labels = []
                next_box_centers = []
                next_frame_data_raw_labels, next_frame_data_lable_flag = next_frame_data.raw_labels
                
                for raw_label in next_frame_data_raw_labels:
                    act_line = raw_label.split()
                    if next_frame_data_lable_flag:
                        label, obj_id, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
                    else:
                        label, obj_id, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
                 
                    if (label in VALID_VOD_CLASS):
                        h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])
                        y = y-0.5*h
                        obj_id = int(obj_id)

                        next_labels.append({'box_class': label,
                                            'label_class': int(VOD_TO_ARGOVERSE[label]), # label
                                            'obj_id': obj_id,
                                            'h': h,
                                            'w': w,
                                            'l': l,
                                            'x': x,
                                            'y': y,
                                            'z': z,
                                            'rotation': rot,
                                            'score': score})
                        next_cam_bbox_3d = np.array([x,y,z,l,h,w,rot])                 
                        next_radar_bbox_3d = box_cam2radar(next_cam_bbox_3d, next_frame_trans.t_camera_radar, next_frame_trans.t_radar_lidar)
                        next_box_centers.append([next_radar_bbox_3d[0], next_radar_bbox_3d[1], next_radar_bbox_3d[2], obj_id, MEAN_BOX_HALF_DIAGONAL[label]])

                next_box_centers = np.array(next_box_centers) # [numbox, 5]   


                
                # Find corresponding boxes
                cur_valid_radar_mask = np.ones(cur_radar_valid.shape[0], dtype=np.bool_)
                cur_valid_radar_classes = np.zeros(cur_radar_valid.shape[0], dtype=np.uint8)
                cur_radar_id = np.zeros(cur_radar_valid.shape[0], dtype=np.int16)
                cur_radar_motion_mask = np.zeros(cur_radar_valid.shape[0], dtype=np.bool_)

                for cur_label in cur_labels:
                    obj_id = cur_label['obj_id']
                    cur_cam_bbox_3d = np.array([cur_label['x'],
                                                cur_label['y'],  
                                                cur_label['z'],
                                                cur_label['l'],
                                                cur_label['h'],
                                                cur_label['w'],
                                                cur_label['rotation']])
               
                    cur_radar_bbox_3d = box_cam2radar(cur_cam_bbox_3d,cur_frame_trans.t_camera_radar, cur_frame_trans.t_radar_lidar)
                    find_radar_pts_in_bbox_3d = copy.deepcopy(cur_radar_bbox_3d) 

                    find_radar_pts_in_bbox_3d[2] -= 0.5 * cur_label['h']  
                    find_radar_pts_in_bbox_3d[3] += BBOX_RADAR_EXPANSION_XY
                    find_radar_pts_in_bbox_3d[4] += BBOX_RADAR_EXPANSION_XY
                    find_radar_pts_in_bbox_3d[5] += BBOX_RADAR_EXPANSION_Z

                    origin_in_bbox_radar_indices = points_in_rbbox(cur_radar_valid, find_radar_pts_in_bbox_3d[None])
                    in_bbox_radar_indices = origin_in_bbox_radar_indices[:, 0]
 
                    radar_pts_in_box = cur_radar_valid[in_bbox_radar_indices]
                    cur_valid_radar_classes[in_bbox_radar_indices] = cur_label['label_class']
                    cur_radar_id[in_bbox_radar_indices] = obj_id
                                

                    find_flag = False
                    for target_label in next_labels:
                        if obj_id == target_label['obj_id'] and ( cur_label['label_class'] == target_label['label_class']):
                            
                            target_cam_bbox_3d = np.array([target_label['x'],
                                            target_label['y'],
                                            target_label['z'],
                                            target_label['l'],
                                            target_label['h'],
                                            target_label['w'],
                                            target_label['rotation']])
                            # to Rada Cooridinate
                            target_radar_bbox_3d = box_cam2radar(target_cam_bbox_3d, next_frame_trans.t_camera_radar, cur_frame_trans.t_radar_lidar)
                            cur_rot = cur_radar_bbox_3d[-1]
                            cur_rot_mat = np.array([[np.cos(cur_rot), np.sin(cur_rot), 0],
                                                    [-np.sin(cur_rot), np.cos(cur_rot), 0],
                                                    [0, 0, 1]])
                            
                            target_rot = target_radar_bbox_3d[-1]
                            target_rot_mat = np.array([[np.cos(target_rot), np.sin(target_rot), 0],
                                                        [-np.sin(target_rot), np.cos(target_rot), 0],
                                                        [0, 0, 1]])
                           
                            # warp Radar to next frame
                            rel_pts = radar_pts_in_box[:, :3] - cur_radar_bbox_3d[:3]
                            rel_pts = rel_pts @ cur_rot_mat.T
                            
                            warped_obj_pts = rel_pts @ target_rot_mat + target_radar_bbox_3d[:3]
                            obj_flow = warped_obj_pts - radar_pts_in_box[:, :3]

                            ego_motion_radar_flow = rigid_radar_flow[in_bbox_radar_indices]
                            cur_radar_motion_mask[in_bbox_radar_indices] = np.linalg.norm(obj_flow - ego_motion_radar_flow, axis=1)>0.05
                            radar_flow[in_bbox_radar_indices] = obj_flow.astype(np.float32)

                            find_flag = True
                            break
                    
                    # No corresponding box
                    if not find_flag:
                        cur_valid_radar_mask[in_bbox_radar_indices] = 0

                # Filter noisy radar
                final_valid_radar_indices = np.full((cur_radar_valid.shape[0], ), True, dtype=bool)


                for ra_id in range(cur_radar_valid.shape[0]):
                    cur_radar_pt = cur_radar_valid[ra_id,:]  
                    # [x, y, z, RCS, v_r, v_r_compensated, time]
                    temp_vr_compensated = cur_radar_pt[5] 

                    if (np.absolute(temp_vr_compensated) > 0.1):
                        cur_radar_pt = cur_radar_pt.reshape(1,-1)
                        relative_radar_obj_flow = radar_flow[ra_id,:] - rigid_radar_flow[ra_id,:]

                        # radial projection of flow
                        radar_flow_radial = np.dot(relative_radar_obj_flow , cur_radar_pt[0,:3])/ np.linalg.norm(cur_radar_pt[0,:3]) * 10 # 注意这里10Hz转为速度
                        vr_valid_flag = (np.absolute(radar_flow_radial - cur_radar_valid[ra_id,5]) < 1)
                        if cur_valid_radar_mask[ra_id] :
                            final_valid_radar_indices[ra_id] = vr_valid_flag 
                        else:
                            final_valid_radar_indices[ra_id] = True
                    
                
                
                create_group_data(group, cur_pose.transform_matrix.astype(np.float32), 
                                  cur_radar_valid, cur_lidar_valid, cur_radar_id, cur_valid_radar_classes,
                                
                                radar_flow, cur_valid_radar_mask,
                                ego1_SE3_ego0.transform_matrix.astype(np.float32), 
                                cur_radar_motion_mask, final_valid_radar_indices)


                            
def proc(x, ignore_current_process=False):
    if not ignore_current_process:
        current=current_process()
        pos = current._identity[0]
    else:
        pos = 1
    process_log(*x, n=pos)


if __name__ == '__main__':
    sf_dir = SF_save_path
    os.makedirs(sf_dir, exist_ok=True)
    my_sf_dir = sf_dir + mode + "/"
    os.makedirs(my_sf_dir, exist_ok=True)
    nproc = int(multiprocessing.cpu_count() - 1)

    clip_files = os.listdir(clip_dir)
    args = sorted([(clip_dir, log, my_sf_dir) for log in clip_files])
    print(f'Using {nproc} processes to process data: {clip_dir} to .h5 format. (#scenes: {len(args)})')

    if nproc <= 1:
        for x in tqdm(args, ncols=120):
            proc(x, ignore_current_process=True)
    else:
        with Pool(processes=nproc) as p:
            res = list(tqdm(p.imap_unordered(proc, args), total=len(clip_files), ncols=120))
    create_reading_index(my_sf_dir)