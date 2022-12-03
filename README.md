# Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images (ICCV'2021)

\[[📜 Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Graph-Based_3D_Multi-Person_Pose_Estimation_Using_Multi-View_Images_ICCV_2021_paper.pdf))\]
\[[📰 Blog (商汤学术)](https://mp.weixin.qq.com/s/N-CQoefmPfSoafzzqGF77A)\]

## Introduction

This is the official release of our paper
[**Graph-Based 3D Multi-Person Pose Estimation Using Multi-View Images**](
https://openaccess.thecvf.com/content/ICCV2021/papers/Wu_Graph-Based_3D_Multi-Person_Pose_Estimation_Using_Multi-View_Images_ICCV_2021_paper.pdf) 
based on [MMPose](https://github.com/open-mmlab/mmpose). Codes will be integrated into MMPose in the future.

## TODO
- [x] Training/testing codes release.
- [x] Pre-trained model release.
- [ ] Integrate codes to [MMPose](https://github.com/open-mmlab/mmpose).

## Results

### Multiview 3D Pose Estimation on [CMU Panoptic](http://domedb.perception.cs.cmu.edu/)

 | Refine Pose | mAP| mAR | MPJPE | Config | Download |
 | :---: | :---: | :---: | :---: | :---: | :---: |
 | - | 97.25 | 98.24 | 17.18 |[config](configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end_test_without_refinement.py) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EUzgB7BmI9VEqSyPH9eW7mwBdc7xj74CrvFIJdwfo2ZcmA?e=8WSbqd);  [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/ETwew6qgVY1AgqwmsC-ZmFYB2eoQyJycVA9NpD2MXuQNIA?e=7YsiFS) |
 | + | 98.65 | 98.80 | 15.68 |[config](configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py) | [model](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EUzgB7BmI9VEqSyPH9eW7mwBdc7xj74CrvFIJdwfo2ZcmA?e=8WSbqd);  [log](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/ETwew6qgVY1AgqwmsC-ZmFYB2eoQyJycVA9NpD2MXuQNIA?e=7YsiFS) |


## Installation

### MMPose
Following this [installation guidance](https://github.com/open-mmlab/mmpose/blob/master/docs/en/install.md) to 
build the latest version of MMPose from source.

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
Download the pre-trained 2D bottom-up pose estimator this [link](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/js20_connect_hku_hk/EcNwzSz4jy5Kl7tu4S3LR_EBQIAAhs96EVCuwIBi4PNm9w?e=UaGexg) and put it under checkpoints
```
multiview_pose
├── checkpoints 
    ├── resnet_50_deconv.pth.tar
```

### Training and testing
For training and testing, use the exactly same commands as 
[MMPose](https://github.com/open-mmlab/mmpose/blob/master/docs/en/get_started.md#train-a-model).
For example, you can train the model with 8 GPUs by using 

```bash
bash tools/dist_train.sh configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py 8
```
To test the model with 8 GPUs, use the following
```bash
bash tools/dist_test.sh configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end.py \
path/to/checkpoint 8 --eval mpjpe mAP
bash tools/dist_test.sh configs/body/3d_kpt_mview_rgb_img/graph_pose/panoptic/gcn_cpn80x80x20_panoptic_cam5_end2end_test_without_refinement.py \
path/to/checkpoint 8 --eval mpjpe mAP   # test without pose refinement
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
