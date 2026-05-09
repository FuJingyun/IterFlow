import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



# import sys
from dataprocess.vod.configuration import ra_KittiLocations
from dataprocess.vod.frame import FrameDataLoader, FrameTransformMatrix, homogeneous_transformation, project_3d_to_2d
from dataprocess.vod.frame.transformations import canvas_crop

import pickle,time
import h5py

import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt
import cv2
# from scripts.network.dataloader import preprocess_VODH5Dataset
import os

mode = "train" # val train
h5_directory = "Dir to h5 file" + mode + "/"

clip_dir = "Dir to CLIP file" + mode + "/" # my_new_clips my_clips cmflow_clips
dlo_pose_dir = "Dir to poses file/" + mode + "/" # my_new_dlo_poses my_dlo_poses cmflow_dlo_poses
kitti_locations = ra_KittiLocations(root_dir="Dir to DATASET")
TXT_save_path = "Dir to output file"

def draw_box_2d_pic(pic, bbox, color):
    pic[bbox[1]:bbox[3], bbox[0]:bbox[2], :] = color
    return pic


def int_box_2d_pic(pic, bbox, id):
    pic[bbox[1]:bbox[3], bbox[0]:bbox[2]] = id
    return pic


def draw_seg_pic(pic, mask, color):
    color_mask = mask.reshape(pic.shape[0], pic.shape[1], 1) * color  #  h, w, 1   1,1,3   -> h, w,3
    pic[mask,:] = color_mask[mask,:]
    return pic

def int_seg_pic(pic, mask, id):
    pic[mask] = id
    return pic


IMG_HEIGHT = 1216
IMG_WIDTH = 1936


if __name__ == "__main__":
    yolo_name = "yolo11"
    deepsort_txt_dir = TXT_save_path + yolo_name + "_vod_track/"  # yolo11x_vod_track yolo11_vod_track

    # Segmentation
    sam_checkpoint = "/data/fjy/SAG_ckpt/sam_vit_l_0b3195.pth"  # sam_vit_l_0b3195.pth  sam_vit_b_01ec64.pth  sam_vit_h_4b8939.pth
    model_type = "vit_l"  # "vit_h"  # "vit_b" vit_h vit_l
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    
    with open(os.path.join(h5_directory, 'index_total.pkl'), 'rb') as f:
        data_index = pickle.load(f)

    total_frame_num = len(data_index)
    relax = False

    for index in range(total_frame_num):
        scene_id, timestamp = data_index[index]
        ts0 = int(timestamp)
        frame1 = '%05d' % ts0
        print(str(index)+'/' +str(total_frame_num-1)+ ' '+ str(scene_id) + ' ' +str(timestamp))

        cur_frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number=frame1)
        cur_frame_trans = FrameTransformMatrix(cur_frame_data)

        # (1216, 1936, 3)
        image = cur_frame_data.image
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        with open(os.path.join(deepsort_txt_dir, frame1+'.txt'), 'r') as text:
            labels = text.readlines()
    
        cur_labels = []
        no_id = 8888 # defalut id
        for raw_label in labels:
            act_line = raw_label.split()
            LW, LH, RW, RH, id = act_line
            LW, LH, RW, RH = map(float, [LW, LH, RW, RH])
            if id =='obj':
                obj_id = int(no_id)
                no_id = no_id + 1
            else:
                obj_id = int(id)

            cur_labels.append({ "LW": LW,
                                "LH": LH,
                                "RW": RW,
                                "RH": RH,
                                'obj_id': obj_id})


        for cur_label in cur_labels:

            obj_id = cur_label['obj_id']
            cur_cam_bbox_2d = np.array([cur_label['LW'],
                                        cur_label['LH'],  # 高度已移动至目标框中心
                                        cur_label['RW'],
                                        cur_label['RH']])

            predictor.set_image(image)
            masks, _, _ = predictor.predict(
                box=cur_cam_bbox_2d[None, :],
                multimask_output=False,
                )
            mask = masks[0]
            # (1, 1216, 1936)
            cur_cam_bbox_2d = cur_cam_bbox_2d.astype(np.int16)
            seg_out_id = int_seg_pic(seg_out_id, mask, obj_id)

            
        with h5py.File(os.path.join(h5_directory, f'{scene_id}.h5'), 'r+') as f:
            key = str(ts0)
            radar_pc = f[key]['radar_pc'][:,:3]  
            t_camera_radar = cur_frame_trans.t_camera_radar
            camera_projection_matrix = cur_frame_trans.camera_projection_matrix

            radar_p = np.concatenate((radar_pc[:,0:3],np.ones((radar_pc.shape[0],1))),axis=1)
            radar_data_t = homogeneous_transformation(radar_p, t_camera_radar)
            uvs = project_3d_to_2d(radar_data_t, camera_projection_matrix)

            ra_id_seg = seg_out_id[uvs[:,1]-1, uvs[:,0]-1]
            seg_name = model_type + "_" + yolo_name + '_ds_id_seg'

            if seg_name in f[key]:
                del f[key][seg_name]
            f[key].create_dataset(seg_name, data=ra_id_seg.astype(np.int16))

