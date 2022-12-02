# Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images

## Introduction

This is the release of the paper
[**Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images**](
https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Graph-Based_3D_Multi-Person_Pose_Estimation_Using_Multi-View_Images_ICCV_2021_paper.pdf) 
based on [MMPose](https://github.com/open-mmlab/mmpose). The code will be integrated into MMPose in the future.

## Results

### Multiview 3D Pose Estimation on [CMU Panoptic](http://domedb.perception.cs.cmu.edu/)

 | Refine Pose | mAP| mAR | MPJPE | Config | Download |
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | - | |  | |[config](configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py) | [model]();  [log]() |
 | + | |  | |[config](configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py) | [model]();  [log]() |


## Installation

### MMPose
Following this [installation guidance](https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md) to 
build the latest from source.

```bash
git clone https://github.com/open-mmlab/mmpose.git
cd mmpose
git checkout reform_multiviewpose     # to be discarded in future
pip install -r requirements.txt
pip install -v -e .
```

### PyTorch Geometric
We implement our GCN modules using [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).
```bash
conda install pyg -c pyg   # for PyTorch >= 1.8.0
```
If having problem installing PyTorch Geometric with the above command, try build it from source following the 
[official tutorial](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).


## Usage

### Data preparation

Prepare CMU Panoptic Dataset following 
[MMPose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/tasks/3d_body_keypoint.md#cmu-panoptic). 
The data structure looks like:

```
multiview_pose
├── multiview_pose
├── tools
├── configs
`── data
    ├── panoptic
        ├── 16060224_haggling1
        |   |   ├── hdImgs
        |   |   ├── hdvideos
        |   |   ├── hdPose3d_stage1_coco19
        |   |   ├── calibration_160224_haggling1.json
        ├── 160226_haggling1
            ├── ...
```

### Pre-trained 2D model
Download the pre-trained 2D bottom-up pose estimator from [model]() and put it under checkpoints
```
multiview_pose
├── checkpoints 
    ├── resnet_50_deconv.pth.tar
```

### Training and testing
For training and testing, use the exactly same commands as 
[MMPose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/get_started.md#train-a-model).
For example,

```bash
bash tools/dist_train.sh configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py 8
```
and 
```bash
bash tools/dist_test.sh configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py \
 path/to/checkpoint 8 --eval mpjpe
```
## Citation

```bibtex
@inproceedings{wu2021graph,
  title={Graph-based 3d multi-person pose estimation using multi-view images},
  author={Wu, Size and Jin, Sheng and Liu, Wentao and Bai, Lei and Qian, Chen and Liu, Dong and Ouyang, Wanli},
  booktitle={ICCV},
  year={2021}
}
```