import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import pickle
import glob
import os
import open3d as o3d
from sklearn.neighbors import BallTree
import time

# grid_size = 1.6
# mesh_size = int (51.2 / grid_size)


def cal_grid_index(row,col,grid_size):
    mesh_size = int (51.2 / grid_size)
    grid_index = int(row * mesh_size + col)
    return grid_index


def is_in_mesh(row,col,grid_size):
    mesh_size = int (51.2 / grid_size)
    check =  (row > (mesh_size-1) or row < 0 or col < 0 or col > (mesh_size-1))
    return ~check

def get_neighbor(grid_index,grid_size):
    mesh_size = int (51.2 / grid_size)
    neighbor = [grid_index]
    row = int(grid_index / mesh_size)
    col = int(grid_index % mesh_size)
    # 1 左上
    temp_row = row-1
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 2 正上
    temp_row = row
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 3 右上
    temp_row = row+1
    temp_col = col-1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 4 正左
    temp_row = row-1
    temp_col = col
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 5 正右
    temp_row = row+1
    temp_col = col
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 6 左下
    temp_row = row-1
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 7 正下
    temp_row = row
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    # 8 右下
    temp_row = row+1
    temp_col = col+1
    if(is_in_mesh(temp_row,temp_col,grid_size)):
        neighbor.append(cal_grid_index(temp_row,temp_col,grid_size))
    return neighbor


# 移除不在有radar点范围内的lidar点
# 以及不在标注范围内的点
def remove_under_ground(xyz, ground_h):
    val_inds = (xyz[:, 2] > ground_h-1)
    velo_valid = xyz[val_inds, :]
    return velo_valid

def remove_height(xyz):
    val_inds = (xyz[:, 2] > -3)
    val_inds = val_inds &  (xyz[:, 2] < 3)
    velo_valid = xyz[val_inds, :]
    return velo_valid


def in_region(xyz):
    val_inds = (xyz[:, 0] > 0 )
    val_inds = val_inds & (xyz[:, 0] < 51.2 )
    val_inds = val_inds & (xyz[:, 1] > -25.6 )
    val_inds = val_inds & (xyz[:, 1] < 25.6 )
    val_inds = val_inds & (xyz[:, 2] < 3 )
    val_inds = val_inds & (xyz[:, 2] > -3 )
    # remove_near_car
    invalid_inds = (xyz[:, 0] <3) & (xyz[:, 0] >0) & (xyz[:, 1] <1.5) & (xyz[:, 1] >-1.5)
    val_inds = val_inds & (~invalid_inds)
    velo_valid = xyz[val_inds, :]
    return velo_valid



#cam00坐标系下的筛选
def remove_xy(xyz):
    value = 51.2
    val_inds = (xyz[:, 1] > -value )
    val_inds = val_inds & (xyz[:, 1] < value )
    val_inds = val_inds & (xyz[:, 0] > -value )
    val_inds = val_inds & (xyz[:, 0] < value )
    velo_valid = xyz[val_inds, :]
    velo_invalid = xyz[~val_inds, :]

    return velo_valid, velo_invalid


def remove_z(xyz):
    val_inds =  (xyz[:, 2] <0.5)
    velo_valid = xyz[val_inds, :]
    velo_invalid = xyz[~val_inds, :]

    return velo_valid, velo_invalid

def read_calib(file):
    with open(file, "r") as f:
        lines = f.readlines()
        intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
        extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
        extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

    return intrinsic, extrinsic

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
    This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    return transform.dot(points.T).T




def lidar_bin_filter(valid_radar, valid_lidar, grid_size):
    mesh={}
    for lidar_index in range(valid_lidar.shape[0]):
        row = int((valid_lidar[lidar_index,0]) / grid_size)
        col = int((25.6 + valid_lidar[lidar_index,1])/ grid_size)
        grid_index = cal_grid_index(row,col,grid_size)
        if (str(grid_index) not in mesh.keys()):
            mesh[str(grid_index)] = 1
        else:
            mesh[str(grid_index)]+=1
    filtered_by_lidar =  np.zeros(valid_radar.shape[0])
    for radar_index in range(valid_radar.shape[0]):
        row = int((valid_radar[radar_index,0]) / grid_size)
        col = int((25.6 + valid_radar[radar_index,1])/ grid_size)
        grid_index = cal_grid_index(row,col,grid_size)
        neighbor = get_neighbor(grid_index,grid_size)
        count_neighbor = 0
        for neighbor_grid in neighbor:
            if(str(neighbor_grid)in mesh.keys()):
                count_neighbor += mesh[str(neighbor_grid)]
        if count_neighbor>1:
            filtered_by_lidar[radar_index] = 1
    lidar_filtered_mask = np.array(filtered_by_lidar, dtype= bool)
    lidar_filtered_radar = valid_radar[lidar_filtered_mask,:]
    return lidar_filtered_radar




if __name__ == '__main__':
    i = 4800
    bin_file_name = "%05d.bin"%i
    txt_file_name = "%05d.txt"%i

    pcd_ground_path = "C:/Users/Alyson/Desktop/try_out/"
    lidar_path = "G:/zju_ilr_vis/data/vod/lidar/"
    radar_path = "G:/zju_ilr_vis/data/vod/radar/"
    calib_lidar = "G:/zju_ilr_vis/data/vod/lidar_calib/"
    calib_radar = "G:/zju_ilr_vis/data/vod/radar_calib/"
    

    # for visualziation 
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 3
    vis.get_render_option().background_color = np.array([0,0,0])
    PC1 = [0,0,1]
    PC2 = [1,0,0]
    PC3 = [0,1,0]
    PC4 = [1,1,0]

    lidar_read_path = lidar_path + bin_file_name
    radar_read_path = radar_path + bin_file_name
    ground_label_path = pcd_ground_path + txt_file_name
    calib_lidar_file = calib_lidar + txt_file_name
    calib_radar_file = calib_radar + txt_file_name


    # 计算 radar 至 lidar的投影变换
    calib_lidar_file = "G:/zju_ilr_vis/data/vod/lidar_calib/" + txt_file_name
    calib_radar_file = "G:/zju_ilr_vis/data/vod/radar_calib/" + txt_file_name
    _camera_projection_matrix, t_camera_radar = read_calib(calib_radar_file)
    _camera_projection_matrix, t_camera_lidar = read_calib(calib_lidar_file)
    # t_radar_camera = np.linalg.inv(t_camera_radar)
    t_lidar_camera = np.linalg.inv(t_camera_lidar)
    # _T_radar_lidar = np.dot(t_radar_camera, t_camera_lidar)
    _T_lidar_radar = np.dot(t_lidar_camera, t_camera_radar)
    # print(_T_radar_lidar)
    print(_T_lidar_radar)


    # 激光雷达点云输入
    lidar_data = np.fromfile(lidar_read_path, dtype=np.float32).reshape(-1,4)
    lidar_pc = lidar_data[:,:3]
    print("lidar_pc.shape")
    print(lidar_pc.shape)


    lidar = o3d.geometry.PointCloud()
    lidar.points = o3d.utility.Vector3dVector(in_region(lidar_pc))
    lidar.paint_uniform_color(PC1)


    # 毫米波雷达点云输入
    radar_data = np.fromfile(radar_read_path, dtype=np.float32).reshape(-1,7)
    radar_pc = radar_data[:,:3]
    radar_pc_homo = np.hstack((radar_pc,
                        np.ones((radar_pc.shape[0], 1),
                                dtype=np.float32)))
    radar_transformed_homo = homogeneous_transformation(radar_pc_homo, transform=_T_lidar_radar)
    # 与 lidar对齐后的radar点云
    radar_transformed = radar_transformed_homo[:,:3]
    print("radar_transformed.shape")
    print(radar_transformed.shape)

    # # 原始毫米波雷达点云输入
    # radar_origin = o3d.geometry.PointCloud()
    # radar_origin.points = o3d.utility.Vector3dVector(radar_pc)
    # radar_origin.paint_uniform_color(PC3)

    # Radar 对齐至Lidar 
    # radar = o3d.geometry.PointCloud()
    # radar.points = o3d.utility.Vector3dVector(radar_transformed)
    # radar.paint_uniform_color(PC2)

    # lidar & radar 点云
    # pcd_pts = np.append(lidar_pc,radar_transformed,axis=0)
    # pcd_out = o3d.geometry.PointCloud()
    # pcd_out.points = o3d.utility.Vector3dVector(pcd_pts)

    # 读取地面分割标签并着色
    seg_label = np.loadtxt(ground_label_path)
    print("seg_label.shape")
    print(seg_label.shape)

    lidar_seg_label = seg_label[:lidar_pc.shape[0]]
    radar_seg_label = seg_label[lidar_pc.shape[0]:]

    lidar_seg = np.array(lidar_seg_label, dtype= bool)
    radar_seg = np.array(radar_seg_label, dtype= bool)

    # Lidar 部分
    lidar_pc_ground = lidar_pc[lidar_seg,:]
    lidar_pc_rest = lidar_pc[~lidar_seg,:]
    valid_lidar = in_region(lidar_pc_rest)
    print("valid_lidar_in_region.shape")
    print(valid_lidar.shape)

    mean_ground = np.mean(lidar_pc_ground, axis=0)
    print("mean_ground")
    print(mean_ground)

    valid_lidar = remove_under_ground(valid_lidar, mean_ground[2])

    print("valid_lidar_no_underground.shape")
    print(valid_lidar.shape)

    # lidar_ground = o3d.geometry.PointCloud()
    # lidar_ground.points = o3d.utility.Vector3dVector(in_region(lidar_pc_ground))
    # lidar_ground.paint_uniform_color(PC1)
    
    
    lidar_filter = o3d.geometry.PointCloud()
    lidar_filter.points = o3d.utility.Vector3dVector(valid_lidar)
    lidar_filter.paint_uniform_color(PC1)


    # radar 部分
    radar_pc_ground = radar_transformed[radar_seg,:]
    radar_pc_rest = radar_transformed[~radar_seg,:]
    valid_radar = in_region(radar_pc_rest)
    print("valid_radar_in_region.shape")
    print(valid_radar.shape)

    # valid_radar , inverted_radar = inverted_ground_radar(valid_radar, mean_ground[2])

    valid_radar = remove_under_ground(valid_radar, mean_ground[2])
    # print("valid_radar_no_underground.shape")
    # print(valid_radar.shape)
    print("valid_radar_up_ground.shape")
    print(valid_radar.shape)

    # print("valid_radar_inverted.shape")
    # print(inverted_radar.shape)

    radar = o3d.geometry.PointCloud()
    radar.points = o3d.utility.Vector3dVector(valid_radar)
    radar.paint_uniform_color(PC2)

    # radar_inverted = o3d.geometry.PointCloud()
    # radar_inverted.points = o3d.utility.Vector3dVector(inverted_radar)
    # radar_inverted.paint_uniform_color(PC3)



    # # 方法一
    # t1 = time.time()
    # # Lidar 约束 radar
    # mesh={}

    # # for i in range( mesh_size * mesh_size ):
    # #     mesh[str(i)] = []

    # for lidar_index in range(valid_lidar.shape[0]):
    #     row = int((valid_lidar[lidar_index,0]) / grid_size)
    #     col = int((25.6 + valid_lidar[lidar_index,1])/ grid_size)
    #     grid_index = cal_grid_index(row,col)
    #     if (str(grid_index) not in mesh.keys()):
    #         mesh[str(grid_index)] = []
    #         mesh[str(grid_index)].append(valid_lidar[lidar_index,:])
    #     else:
    #         mesh[str(grid_index)].append(valid_lidar[lidar_index,:])

    

    
    # for key in mesh.keys():
    #     mesh[key] = np.array(mesh[key])
    #     # print(mesh[key].shape)


    # filtered_by_lidar =  np.zeros(valid_radar.shape[0])
            
    # for radar_index in range(valid_radar.shape[0]):
    #     row = int((valid_radar[radar_index,0]) / grid_size)
    #     col = int((25.6 + valid_radar[radar_index,1])/ grid_size)
    #     grid_index = cal_grid_index(row,col)
    #     neighbor = get_neighbor(grid_index)
    #     count_neighbor = 0
    #     for neighbor_grid in neighbor:
    #         if(str(neighbor_grid)in mesh.keys()):
    #             count_neighbor += mesh[str(neighbor_grid)].shape[0]  
    #     if count_neighbor>3:
    #         filtered_by_lidar[radar_index] = 1

    # print(f'方法一耗时:{time.time() - t1:.4f}s')

    # lidar_filtered_mask = np.array(filtered_by_lidar, dtype= bool)
    # lidar_filtered_radar = valid_radar[lidar_filtered_mask,:]

    # print("filtered_by_lidar_valid_radar_no_underground.shape")
    # print(lidar_filtered_radar.shape)



    # radar_filter = o3d.geometry.PointCloud()
    # radar_filter.points = o3d.utility.Vector3dVector(lidar_filtered_radar)
    # radar_filter.paint_uniform_color(PC2)





    # # 方法一 优化版本
    # # t3 = time.time()
    # # Lidar 约束 radar
    # mesh={}

    # # for i in range( mesh_size * mesh_size ):
    # #     mesh[str(i)] = []

    # for lidar_index in range(valid_lidar.shape[0]):
    #     row = int((valid_lidar[lidar_index,0]) / grid_size)
    #     col = int((25.6 + valid_lidar[lidar_index,1])/ grid_size)
    #     grid_index = cal_grid_index(row,col)
    #     if (str(grid_index) not in mesh.keys()):
    #         mesh[str(grid_index)] = 1
    #     else:
    #         mesh[str(grid_index)]+=1

    # filtered_by_lidar =  np.zeros(valid_radar.shape[0])
            
    # for radar_index in range(valid_radar.shape[0]):
    #     row = int((valid_radar[radar_index,0]) / grid_size)
    #     col = int((25.6 + valid_radar[radar_index,1])/ grid_size)
    #     grid_index = cal_grid_index(row,col)
    #     neighbor = get_neighbor(grid_index)
    #     count_neighbor = 0
    #     for neighbor_grid in neighbor:
    #         if(str(neighbor_grid)in mesh.keys()):
    #             count_neighbor += mesh[str(neighbor_grid)]
    #     if count_neighbor>3:
    #         filtered_by_lidar[radar_index] = 1

    # # print(f'方法一优化版本耗时:{time.time() - t3:.4f}s')

    # lidar_filtered_mask = np.array(filtered_by_lidar, dtype= bool)
    # lidar_filtered_radar = valid_radar[lidar_filtered_mask,:]


    lidar_filtered_radar = lidar_bin_filter(valid_radar, valid_lidar)

    print("filtered_by_lidar_valid_radar_no_underground.shape")
    print(lidar_filtered_radar.shape)
    radar_filter = o3d.geometry.PointCloud()
    radar_filter.points = o3d.utility.Vector3dVector(lidar_filtered_radar)
    radar_filter.paint_uniform_color(PC2)






    # # 方法二
    # t2 = time.time()
    # dist_by_lidar =  np.zeros(valid_radar.shape[0])

    # for pts_id in range(valid_radar.shape[0]):
    #     nearest_index,dist = nearest_neighbor_search(valid_radar[pts_id], valid_lidar)
    #     if (dist<2) :
    #         dist_by_lidar[pts_id] = 1
    
    # print(f'方法二耗时:{time.time() - t2:.4f}s')

    # dist_mask = np.array(dist_by_lidar, dtype= bool)
    # dist_filtered_radar = valid_radar[dist_mask,:]

    # print("dist_radar_no_underground.shape")
    # print(dist_filtered_radar.shape)

    # radar_dist = o3d.geometry.PointCloud()
    # radar_dist.points = o3d.utility.Vector3dVector(dist_filtered_radar)
    # radar_dist.paint_uniform_color(PC2)






    # # 全部点地面分割着色
    # points_colors = np.zeros([seg_label.shape[0],3])
    # for i in range(seg_label.shape[0]):
    #     # 地面点情形
    #     if seg_label[i]:
    #         points_colors[i] = PC2
    #     else:
    #         points_colors[i] = PC1
    # pcd_out.colors = o3d.utility.Vector3dVector(points_colors)

    # # lidar 部分的地面
    # lidar_seg_label = seg_label[:lidar_pc.shape[0]]
    # print("Ground points number for lidar:")
    # print(np.sum(lidar_seg_label))
    # lidar_points_colors = np.zeros([lidar_seg_label.shape[0],3])
    # for i in range(lidar_seg_label.shape[0]):
    #     # 地面点情形
    #     if lidar_seg_label[i]:
    #         lidar_points_colors[i] = PC3
    #     else:
    #         lidar_points_colors[i] = PC1
    # lidar.colors = o3d.utility.Vector3dVector(lidar_points_colors)

    # # radar 部分的地面
    # radar_seg_label = seg_label[lidar_pc.shape[0]:]
    # print("Ground points number for radar:")
    # print(np.sum(radar_seg_label))
    # radar_points_colors = np.zeros([radar_seg_label.shape[0],3])
    # for i in range(radar_seg_label.shape[0]):
    #     # 地面点情形
    #     if radar_seg_label[i]:
    #         radar_points_colors[i] = PC3
    #     else:
    #         radar_points_colors[i] = PC1
    # radar.colors = o3d.utility.Vector3dVector(radar_points_colors)



    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    vis.add_geometry(mesh_frame)

    # vis.add_geometry(pcd_out)
    vis.add_geometry(lidar_filter)
    # vis.add_geometry(lidar_rest)
    # vis.add_geometry(lidar)
    # vis.add_geometry(lidar_ground)
    vis.add_geometry(radar_filter)
    # vis.add_geometry(radar)
    # vis.add_geometry(radar_inverted)
    # vis.add_geometry(radar_dist)
    # vis.add_geometry(radar_origin)
    # vis.add_geometry(scan_remain)
    # vis.add_geometry(scan)
    # vis.add_geometry(ground)
    # vis.add_geometry(rest)

    vis.run()
    vis.destroy_window()

