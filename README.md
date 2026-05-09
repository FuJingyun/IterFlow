# IterFlow
Source Code for ICML 2026 paper: "Weakly Supervised Cross-Modal Learning for 4D Radar Scene Flow Estimation". 
![](image/overall.png)

## 0. Setup
**Environment**: Clone the repo and build the environment. We use conda to manage the environment.
check [detail installation](https://github.com/KTH-RPL/OpenSceneFlow/assets/README.md) for more information. 

```bash
conda env create -f environment.yaml
```

CUDA package (need install nvcc compiler):
```bash
# CUDA already install in python environment.
cd assets/cuda/chamfer3D && python ./setup.py install && cd ../../..
```

## 1. Data Preparation
### A. Download The View-of-Delft dataset [(VoD)](https://github.com/tudelft-iv/view-of-delft-dataset)
The dataset is organized as follows:
```
PATH_TO_VOD_DATASET
    ├── image_2
    │   │── 00001.jpg
    |       ...
    ├── pose
    │   │── 00001.json
    |       ...
    |       ...
    ├── label_2_withid
    │   │── 00001.txt
    |       ...
    |
    ├── lidar
    │   │── training
    │       ├── velodyne
    │           ├── 00001.bin
    │       ...
    │   │── calib
    │       ├──00001.txt
    │       ...
    ├── radar_5frames
    │   │── training
    │       ├── velodyne
    │           ├── 00001.bin
    │       ...
    │   │── calib
    │       ├──00001.txt
    │       ...
    ├── seg_ground_5frames
    │   │── training
    │       ├──00001.txt
```

### B. Generate VoD Scene Flow Dataset in .h5 format
In each script that needs to be run, the parts where the **PATH** or **MODE** needs to be modified for Reproduction have been highlighted with **"TO DO"**.
```bash
# You need to change the paths in gen_ra_gt_flow.py.py
# change the val/train mode to generate the Training set and Validation set seperately.

cd ./dataprocess
python gen_ra_gt_flow.py.py
```

### C. Generate 2D Tracking boxes for VoD sequences with YOLOv11 model

we adopt the deepsort 2D tracking algorithm from [YOLOv11-DeepSort](https://github.com/Sharpiless/YOLOv11-DeepSort/tree/main)
And we use the official pretrained Yolov11L model weight:
[Yolov11-L](https://pan.baidu.com/s/13m_R_lYVWGHoGNYHEzAeHw?pwd=uvnh)


```bash
# You need to change the **PATH** or **MODE** in ./dataprocess/YOLOv11-DeepSort/my_yolov11.py
# change the val/train mode to generate 2D Tracking boxes for the Training set and Validation set seperately.

cd ./dataprocess/YOLOv11-DeepSort
python my_yolov11.py
```


### D. Generate 2D Segmentation Masks with SAM Model and Project to 3D Space

We use the pretrained SAM model to generate instance-level masks for each 2D tracking box from previous step. 
[SAM with ViT-H](https://pan.baidu.com/s/13m_R_lYVWGHoGNYHEzAeHw?pwd=uvnh)
Then per-point instance id is generated for radar point clouds base on 2D-3D projection.

```bash
# You need to change the **PATH** or **MODE** in ./dataprocess/yolo11_deepsort_segany.py
# change the val/train mode to generate per-point instance id for the Training set and Validation set seperately.

cd ./dataprocess
python yolo11_deepsort_segany.py
```

## 2. Training
```bash
# You need to change the **PATH** in ./conf/fusion_config.yaml
# Maybe also the GPU settings

cd ..
python train.py
```
Please check `/checkpoint` file for our trained model.

## 3. Evaluation
```bash
# You need to change the **PATH** in ./conf/fusion_eval.yaml

cd ..
python eval.py
```



## Cite & Acknowledgements
❤️: [OpenSceneFlow](https://github.com/KTH-RPL/OpenSceneFlow)
❤️: [DeFlow](https://github.com/KTH-RPL/DeFlow), [BucketedSceneFlowEval](https://github.com/kylevedder/BucketedSceneFlowEval)
❤️: [PV-RAFT](https://github.com/weiyithu/PV-RAFT) 

