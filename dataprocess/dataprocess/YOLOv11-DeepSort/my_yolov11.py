# -.
from dataprocess.vod.configuration import ra_KittiLocations
from dataprocess.vod.frame import FrameDataLoader, FrameTransformMatrix  # , homogeneous_transformation, project_3d_to_2d
# from dataprocess.vod.frame.transformations import canvas_crop


import numpy as np
import cv2
from ultralytics import YOLO
from imutils import resize
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import torchvision.transforms as transforms
from PIL import Image
import time
from tqdm import tqdm
# from deep_sort.utils.parser import get_config
from deep_sort.deep_sort.deep_sort import DeepSort
import yaml
from easydict import EasyDict as edict
from yaml import Loader


# TO DO 1 : mode
# TO DO 2 : The location of the files
mode = "train" # train val
clip_dir = "Dir to CLIP file/" + mode + "/" # my_new_clips my_clips cmflow_clips
dlo_pose_dir = "Dir to poses file/" + mode + "/" # my_new_dlo_poses my_dlo_poses cmflow_dlo_poses
kitti_locations = ra_KittiLocations(root_dir="Dir to DATASET")
object_class=['car', 'bicycle', 'person']
save_path = "Dir to save file/" + "yolo11m_vod_track/"


class YamlParser(edict):
    """
    This is yaml parser based on EasyDict.
    """
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read(), Loader=Loader))

        super(YamlParser, self).__init__(cfg_dict)

    
    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            self.update(yaml.load(fo.read(), Loader=Loader))

    
    def merge_from_dict(self, config_dict):
        self.update(config_dict)



def get_config(config_file=None):
    return YamlParser(config_file=config_file)


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



if __name__ == '__main__':
    yamlPath = './deep_sort/configs/deep_sort.yaml'
    cfg = get_config()
    cfg.merge_from_file(yamlPath)

    vod_track_dir = save_path
    os.makedirs(vod_track_dir , exist_ok=True)

    clip_files = os.listdir(clip_dir)

    # Yolo11 Model
    model = YOLO('./model/yolo11l.pt')  # Replace with your YOLO model

    names = model.names

    for clip_file in clip_files:
        cur_clip = os.path.join(clip_dir, clip_file)
        DS = DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=True)
        
        clip = open(os.path.join(clip_dir, clip_file), 'r').readlines()
        start_idx = int(clip[0])
        end_idx = int(clip[-1])

        dlo_poses = open(os.path.join(dlo_pose_dir, clip_file), 'r').readlines()
        offset = end_idx + 1 - start_idx - len(dlo_poses)

        timestamps = range(start_idx + offset, end_idx)

        for cnt, ts0 in enumerate(timestamps):
            frame1 = '%05d' % ts0
            print(clip_file + " "+ frame1)
            temp_txt = os.path.join(vod_track_dir, frame1+'.txt')

            cur_frame_data = FrameDataLoader(kitti_locations=kitti_locations, frame_number='%05d' % ts0)
            cur_frame_trans = FrameTransformMatrix(cur_frame_data)

            # (1216, 1936, 3)
            image = cur_frame_data.image
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = model(image)

            bbox_xywh = []
            confs = []
            clss = []
            detect_out = []

            # Iterate over the results of object detection
            for result in results:

                # Iterate over each bounding box detected in the result
                for r in result.boxes.data.tolist():
                    # Extract the coordinates, score, and class ID from the bounding box
                    x1, y1, x2, y2, score, class_id = r

                    obj = [
                        int((x1+x2)/2), int((y1+y2)/2),
                        x2-x1, y2-y1
                    ]
                    bbox_xywh.append(obj)
                    confs.append(score)
                    clss.append(class_id)

                    # for no track case: from detection
                    xyxy = np.zeros(4)
                    xyxy[0] = x1
                    xyxy[1] = y1
                    xyxy[2] = x2
                    xyxy[3] = y2
                    # xyxy = xyxy.reshape(1,4)
                    detect_out.append(xyxy)

            xywhs = torch.Tensor(bbox_xywh)
            confss = torch.Tensor(confs)

            trackout =[]
            if len(bbox_xywh)>0:
                # N, 4
                trackout = DS.update(xywhs, confss, clss, image)
        
            with open(temp_txt,"w") as f:
                if  len(trackout) >0 : 
                    print(trackout)
                    for value in list(trackout):
                        x1, y1, x2, y2, cls_, track_id = value
                        f.write(str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+ str(track_id)+"\n")
                else:
                    print("Empty trackout")
                    if len(detect_out)>0:
                        for i in range(len(detect_out)):
                            obj = detect_out[i]
                            f.write(str(obj[0])+" "+str(obj[1])+" "+str(obj[2])+" "+str(obj[3])+" "+ "obj"+"\n")
                    