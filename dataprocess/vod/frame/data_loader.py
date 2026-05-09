import os
import numpy as np
import cv2
import random
from typing import Optional, List
from matplotlib import pyplot as plt
import logging

from scipy.spatial import KDTree

# import sys
# sys.path.append('/home/fjy/DeFlow-master/dataprocess/')

from dataprocess.vod.configuration import KittiLocations
# from vod.configuration import KittiLocations


def canvas_crop(points, image_size, points_depth=None):
    """
This function filters points that lie outside a given frame size.
    :param points: Input points to be filtered.
    :param image_size: Size of the frame.
    :param points_depth: Filters also depths smaller than 0.
    :return: Filtered points.
    """
    idx = points[:, 0] > 0
    idx = np.logical_and(idx, points[:, 0] < image_size[1])
    idx = np.logical_and(idx, points[:, 1] > 0)
    idx = np.logical_and(idx, points[:, 1] < image_size[0])
    if points_depth is not None:
        idx = np.logical_and(idx, points_depth > 0)

    return idx


def project_3d_to_2d(points: np.ndarray, projection_matrix: np.ndarray):
    """
This function projects the input 3d ndarray to a 2d ndarray, given a projection matrix.
    :param points: Homogenous points to be projected.
    :param projection_matrix: 4x4 projection matrix.
    :return: 2d ndarray of the projected points.
    """
    if points.shape[-1] != 4:
        raise ValueError(f"{points.shape[-1]} must be 4!")

    uvw = projection_matrix.dot(points.T)
    uvw /= uvw[2]
    uvs = uvw[:2].T
    uvs = np.round(uvs).astype(np.int)

    return uvs


def diamond_kernel(size):
    assert size % 2 == 1
    kernel = np.zeros([size, size], dtype=np.uint8)
    for i in range(size):
        if i <= size // 2:
            start = size // 2 - i
            end = size // 2 + i
            kernel[i][start:end+1] = 1
        else:
            start = i - size // 2
            end = size - start
            kernel[i][start:end] = 1

    return kernel


def fit_plane(points, max_iter, threshold):
    points = np.concatenate([points, np.array([[0, 0, -0.5]])], axis=0)

    num_points = len(points)
    best_param = np.array([0, 0, 1, 0.5])
    best_count = 0

    if num_points < 15000:
        return best_param

    for i in range(max_iter):
        idx = random.sample(range(num_points), 3)
        sample_pts = points[idx]

        n = np.cross(sample_pts[1] - sample_pts[0], sample_pts[2] - sample_pts[0])
        n = n / np.linalg.norm(n)

        if n[2] < 0:
            n = -n

        d = -np.dot(n, sample_pts[0])
        param = np.array([*n, d])

        dist = np.abs(points[:, 0] * n[0] + points[:, 1] * n[1] + points[:, 2] * n[2] + d)
        count = np.array(dist <= threshold, dtype=np.int32).sum()

        if count > best_count:
            best_count = count
            best_param = param

    return best_param


class FrameDataLoader:
    """
    This class is responsible for loading any possible data from the dataset for a single specific frame.
    """
    def __init__(self,
                 kitti_locations: KittiLocations,
                 frame_number: str,
                 radar_region_reduce=False,
                 radar_density_reduce=False,
                 radar_penetration_reduce=False):
        """
Constructor which creates the backing fields for the properties which can load and store data from the dataset
upon request
        :param kitti_locations: KittiLocations object.
        :param frame_number: Specific frame number for which the data should be loaded.
        """

        # Assigning parameters
        self.kitti_locations: KittiLocations = kitti_locations
        self.frame_number: str = frame_number

        # radar reduction parameters
        self.radar_region_reduce = radar_region_reduce
        self.radar_density_reduce = radar_density_reduce
        self.radar_penetration_reduce = radar_penetration_reduce

        # Getting filed id from frame number
        self.file_id: str = str(self.frame_number).zfill(5)

        # Creating properties for possible data.
        # Data is only loaded upon request, then stored for future use.
        self._image: Optional[np.ndarray] = None
        self._lidar_data: Optional[np.ndarray] = None
        self._radar_data: Optional[np.ndarray] = None
        self._raw_labels: Optional[np.ndarray] = None
        self._prediction: Optional[np.ndarray] = None
        self._ground_label: Optional[np.ndarray] = None

    @property
    def image(self):
        """
Image information property in RGB format.
        :return: RGB image.
        """
        if self._image is not None:
            # When the data is already loaded.
            return self._image
        else:
            # Load data if it is not loaded yet.
            self._image = self.get_image()
            return self._image

    @property
    def lidar_data(self):
        """
Ego-motion compensated 360 degree lidar data of a Nx4 array including x,y,z,reflectance values.
        :return: Lidar data.
        """
        if self._lidar_data is not None:
            # When the data is already loaded.
            return self._lidar_data
        else:
            # Load data if it is not loaded yet.
            self._lidar_data = self.get_lidar_scan()
            return self._lidar_data
        
    @property
    def ground_label(self):
        """
Ego-motion compensated 360 degree lidar data of a Nx4 array including x,y,z,reflectance values.
        :return: Lidar data.
        """
        if self._ground_label is not None:
            # When the data is already loaded.
            return self._ground_label
        else:
            # Load data if it is not loaded yet.
            self._ground_label = self.get_ground_label()
            return self._ground_label

    @property
    def radar_data(self):
        if self._radar_data is not None:
            # When the data is already loaded.
            return self._radar_data
        else:
            # Load data if it is not loaded yet.
            radar_data = self.get_radar_scan()

            # 获取全点云的地面标签
            # radar_ground_file = os.path.join(self.kitti_locations.ground_dir, f'{self.file_id}.txt')
            # seg_label = np.loadtxt(radar_ground_file)

            # lidar_file = os.path.join(self.kitti_locations.lidar_dir, f'{self.file_id}.bin')
            # lidar_pc = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

            # radar_seg_label = seg_label[lidar_pc.shape[0]:]
            self._radar_data = radar_data

            return self._radar_data
    @property
    def old_radar_data(self):
        """
Ego-motion compensated radar data from the front bumper in the format of [x, y, z, RCS, v_r, v_r_compensated, time].

* V_r is the relative radial velocity
* v_r_compensated is the absolute (i.e. ego motion compensated) radial velocity of the point.
* Time is the time id of the point, indicating which scan it originates from.

        :return: Radar data.
        """
        if self._radar_data is not None:
            # When the data is already loaded.
            return self._radar_data
        else:
            # Load data if it is not loaded yet.
            radar_data = self.get_radar_scan()

            if self.radar_region_reduce:
                ground_filter = (radar_data[:, 0] <= 3) & (radar_data[:, 1] <= 1.5) & \
                                (radar_data[:, 1] >= -1.5) & (np.abs(radar_data[:, 2] + 0.5) < 0.2)
                ground_pts = radar_data[ground_filter]
                line_param = fit_plane(ground_pts[:, :3], 100, 0.05)

                pts_height = np.dot(radar_data[:, :3], line_param[:3]) + line_param[-1]
                pts_filter = (pts_height > -0.5) & (pts_height < 3.5) & (radar_data[:, 0] > 2)
                radar_data = radar_data[pts_filter]

            if self.radar_density_reduce:
                kd_tree = KDTree(radar_data[:, :3])
                valid_mask = []
                for pts in radar_data[:, :3]:
                    if len(kd_tree.query_ball_point(pts, 1.)) > 3:
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                radar_data = radar_data[valid_mask]

            if self.radar_penetration_reduce:
                point_homo = np.hstack((radar_data[:, :3],
                                        np.ones((radar_data.shape[0], 1),
                                                dtype=np.float32)))

                trans_mat = np.array([[-0.013857  , -0.9997468 ,  0.01772762,  0.05283124],
                                      [ 0.10934269, -0.01913807, -0.99381983,  0.98100483],
                                      [ 0.99390751, -0.01183297,  0.1095802 ,  1.44445002],
                                      [ 0.        ,  0.        ,  0.        ,  1.        ]])
                intrinsic = np.array([[1.4954686e+03, 0.0000000e+00, 9.6127246e+02, 0.0000000e+00],
                                      [0.0000000e+00, 1.4954686e+03, 6.2489594e+02, 0.0000000e+00],
                                      [0.0000000e+00, 0.0000000e+00, 1.0000000e+00, 0.0000000e+00]])
                image_shape = self.image.shape

                points_camera_frame = point_homo @ trans_mat.T

                point_depth = points_camera_frame[:, 2]

                uvs = project_3d_to_2d(points=points_camera_frame,
                                       projection_matrix=intrinsic)

                filtered_idx = canvas_crop(points=uvs,
                                           image_size=image_shape,
                                           points_depth=point_depth)

                point_depth = point_depth[filtered_idx]
                uvs = uvs[filtered_idx]
                radar_data = radar_data[filtered_idx]

                depth_map = np.zeros([*image_shape[:2]])
                attr_map = np.zeros([*image_shape[:2], radar_data.shape[-1] - 3])

                for uv, d, attr in zip(uvs, point_depth, radar_data[:, 3:]):
                    u = uv[0]
                    v = uv[1]

                    if depth_map[v, u] == 0:
                        depth_map[v, u] = d
                        attr_map[v, u] = attr
                    else:
                        if depth_map[v, u] > d:
                            depth_map[v, u] = d
                            attr_map[v, u] = attr

                # fast depth completion (using fixed kernel)
                fast_depth_map = depth_map.copy().astype(np.float32)
                valid_pixels = (fast_depth_map > 0)
                fast_depth_map[valid_pixels] = depth_map.max() - fast_depth_map[valid_pixels]
                fast_depth_map = cv2.dilate(fast_depth_map, diamond_kernel(51))
                valid_pixels = (fast_depth_map > 0)
                fast_depth_map[valid_pixels] = depth_map.max() - fast_depth_map[valid_pixels]
                fast_depth_map[~valid_pixels] = 0
                clean_mask = np.abs(fast_depth_map - depth_map) > 1.5
                fast_depth_map[clean_mask] = 0

                # fast depth completion (using unit cube)
                # fast_depth_map = depth_map.copy().astype(np.float32)
                # size = 0.2
                # corner_offset = np.array([[-size, size, size, 0],
                #                           [-size, size, -size, 0],
                #                           [-size, -size, size, 0],
                #                           [-size, -size, -size, 0],
                #                           [size, size, size, 0],
                #                           [size, size, -size, 0],
                #                           [size, -size, size, 0],
                #                           [size, -size, -size, 0]])
                # points_cube_corner = points_camera_frame[:, None] + corner_offset
                # corner_uv = project_3d_to_2d(points=points_cube_corner.reshape(-1, 4), projection_matrix=intrinsic)
                # filtered_idx = canvas_crop(points=corner_uv,
                #                            image_size=image_shape,
                #                            points_depth=points_cube_corner.reshape(-1, 4)[:, 2])
                # corner_mask = filtered_idx.astype(np.int32).reshape(-1, 8).sum(axis=-1) == 8
                # valid_corner_uv = corner_uv.reshape(-1, 8, 2)[corner_mask]
                # valid_center_uv = project_3d_to_2d(points=points_camera_frame, projection_matrix=intrinsic)[corner_mask]
                # image_corner_uv = np.concatenate([valid_corner_uv.min(axis=1), valid_corner_uv.max(axis=1)], axis=-1)
                # for center_uv, corner_uv in zip(valid_center_uv, image_corner_uv):
                #     depth_map_part = depth_map[corner_uv[1]:corner_uv[3], corner_uv[0]:corner_uv[2]]
                #     valid_v, valid_u = depth_map_part.nonzero()
                #     cur_depth = depth_map[center_uv[1], center_uv[0]]
                #     valid_v += corner_uv[1]
                #     valid_u += corner_uv[0]
                #     remove_flag = depth_map[valid_v, valid_u] > 6 + cur_depth
                #     fast_depth_map[valid_v[remove_flag],  valid_u[remove_flag]] = 0

                # solve penetration
                # cand_v, cand_u = depth_map.nonzero()
                # value_mask = depth_map[cand_v, cand_u] - fast_depth_map[cand_v, cand_u] <= 5
                # invalid_cand_v = cand_v[~value_mask]
                # invalid_cand_u = cand_u[~value_mask]
                # depth_map[invalid_cand_v, invalid_cand_u] = 0
                # depth_map[invalid_cand_v, invalid_cand_u] = fast_depth_map[invalid_cand_v, invalid_cand_u]
                depth_map = fast_depth_map

                us = np.arange(image_shape[1])
                vs = np.arange(image_shape[0])
                vs, us = np.meshgrid(vs, us, indexing='ij')
                temp = np.stack([us * depth_map, vs * depth_map, depth_map, np.ones([*image_shape[:2]])], axis=-1)
                temp = temp[depth_map > 0]
                intrinsic = np.concatenate([intrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                cam_points = temp @ np.linalg.inv(intrinsic.T)
                radar_points = cam_points @ np.linalg.inv(trans_mat).T

                radar_points = radar_points[:, :3]
                radar_attr = attr_map[depth_map > 0]
                radar_data = np.concatenate([radar_points, radar_attr], axis=-1)

            self._radar_data = radar_data

            return self._radar_data

    @property
    def raw_labels(self):
        """
The labels include the ground truth data for the frame in kitti format including:

* Class: Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
* Truncated: Not used, only there to be compatible with KITTI format.
* Occluded: Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
* Alpha: Observation angle of object, ranging [-pi..pi]
* Bbox: 2D bounding box of object in the image (0-based index) contains left, top, right, bottom pixel coordinates.
* Dimensions: 3D object dimensions: height, width, length (in meters)
* Location: 3D object location x,y,z in camera coordinates (in meters)
* Rotation: Rotation around -Z axis of the LiDAR sensor [-pi..pi]
        :return: Label data in string format
        """
        if self._raw_labels is not None:
            # When the data is already loaded.
            return self._raw_labels
        else:
            # Load data if it is not loaded yet.
            self._raw_labels, cur_lable_flag = self.get_labels()
            return self._raw_labels, cur_lable_flag

    @property
    def predictions(self):
        """
The predictions include the predicted information for the frame in kitti format including:

* Class: Describes the type of object: 'Car', 'Pedestrian', 'Cyclist', etc.
* Truncated: Not used, only there to be compatible with KITTI format.
* Occluded: Integer (0,1,2) indicating occlusion state 0 = fully visible, 1 = partly occluded 2 = largely occluded.
* Alpha: Observation angle of object, ranging [-pi..pi]
* Bbox: 2D bounding box of object in the image (0-based index) contains left, top, right, bottom pixel coordinates.
* Dimensions: 3D object dimensions: height, width, length (in meters)
* Location: 3D object location x,y,z in camera coordinates (in meters)
* Rotation: Rotation around -Z axis of the LiDAR sensor [-pi..pi]
        :return: Label data in string format
        """

        if self._prediction is not None:
            # When the data is already loaded.
            return self._prediction
        else:
            # Load data if it is not loaded yet.
            self._prediction = self.get_predictions()
            return self._prediction

    def get_image(self) -> Optional[np.ndarray]:
        """
This method obtains the image information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: Numpy array with image data.
        """
        try:
            img = plt.imread(
                os.path.join(self.kitti_locations.camera_dir, f'{self.frame_number}.jpg'))

        except FileNotFoundError:
            logging.error(f"{self.frame_number}.jpg does not exist at location: {self.kitti_locations.camera_dir}!")
            return None

        return img

    def get_radar_scan(self) -> Optional[np.ndarray]:
        """
        This method obtains the radar information from the location specified by the KittiLocations object. If the file
        does not exist, it returns None.

        :return: Numpy array with radar data.
        """

        try:
            radar_file = os.path.join(self.kitti_locations.radar_dir, f'{self.file_id}.bin')
            scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 7)

            # scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 11)
            # scan = scan[:, :7]

            # scan = np.fromfile(radar_file, dtype=np.float32).reshape(-1, 4)
            # scan = np.concatenate([scan, np.zeros([len(scan), 3])], axis=-1)

            assert scan.shape[-1] == 7

        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.kitti_locations.radar_dir}!")
            return None

        return scan
    

    def get_lidar_scan(self) -> Optional[np.ndarray]:
        """
This method obtains the lidar information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: Numpy array with lidar data.
        """

        try:
            # fjy TO DO
            lidar_file = os.path.join(self.kitti_locations.lidar_dir, f'{self.file_id}.bin')
            scan = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        except FileNotFoundError:
            logging.error(f"{self.file_id}.bin does not exist at location: {self.kitti_locations.lidar_dir}!")
            return None

        return scan
    

    def get_ground_label(self) -> Optional[np.ndarray]:
        try:
            # fjy TO DO
            ground_label_file = os.path.join(self.kitti_locations.ground_dir, f'{self.file_id}.txt')
            seg_label = np.loadtxt(ground_label_file)
            seg_label_out = np.array(seg_label, dtype= bool)       

        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.kitti_locations.ground_dir}!")
            return None

        return seg_label_out

    def get_labels(self) -> Optional[List[str]]:
        """
This method obtains the label information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: List of strings with label data.
        """

        lable_flag = True

        try:
            # My edit
            # label_file = os.path.join(self.kitti_locations.label_dir, f'{self.file_id}.txt')
            # with open(label_file, 'r') as text:
            #     labels = text.readlines()

            # For normal model
            label_file = os.path.join(self.kitti_locations.label_dir, f'{self.file_id}.txt')
            if os.path.exists(label_file):
                with open(label_file, 'r') as text:
                    labels = text.readlines()
            else:
                label_file = os.path.join(self.kitti_locations.cmflow_label_dir, f'{self.file_id}.txt')
                with open(label_file, 'r') as text:
                    labels = text.readlines()
                lable_flag = False

            # # For cmflow:
            # label_file = os.path.join(self.kitti_locations.cmflow_label_dir, f'{self.file_id}.txt')
            # if os.path.exists(label_file):
            #     with open(label_file, 'r') as text:
            #         labels = text.readlines()
            #     lable_flag = False
            # else:
            #     label_file = os.path.join(self.kitti_locations.label_dir, f'{self.file_id}.txt')
            #     with open(label_file, 'r') as text:
            #         labels = text.readlines()
           

        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.kitti_locations.label_dir}!")
            return None

        return labels, lable_flag

    def get_predictions(self) -> Optional[List[str]]:
        """
This method obtains the prediction information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: List of strings with prediction data.
        """

        try:
            label_file = os.path.join(self.kitti_locations.pred_dir, f'{self.file_id}.txt')
            with open(label_file, 'r') as text:
                labels = text.readlines()

        except FileNotFoundError:
            logging.error(f"{self.file_id}.txt does not exist at location: {self.kitti_locations.pred_dir}!")
            return None

        return labels

