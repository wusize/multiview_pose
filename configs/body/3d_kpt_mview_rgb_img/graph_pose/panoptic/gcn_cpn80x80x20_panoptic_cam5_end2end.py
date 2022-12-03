_base_ = ['../../../../_base_/datasets/panoptic_body3d.py']
from configs._base_.datasets.panoptic_body3d import dataset_info
log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metric=['mAP', 'mpjpe'], save_best='mAP')

optimizer = dict(
    type='Adam',
    lr=0.0001,
)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 8])
total_epochs = 10
log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

space_size = [8000, 8000, 2000]
space_center = [0, -500, 800]
cube_size = [80, 80, 20]
sub_space_size = [2000, 2000, 2000]
sub_cube_size = [64, 64, 64]
image_size = [960, 512]
heatmap_size = [240, 128]
num_joints = 15
num_cameras = 5
train_data_cfg = dict(
    image_size=image_size,
    heatmap_size=[heatmap_size],
    num_joints=num_joints,
    seq_list=[
        # '160422_haggling1'
        '160422_ultimatum1', '160224_haggling1', '160226_haggling1',
        '161202_haggling1', '160906_ian1', '160906_ian2', '160906_ian3',
        '160906_band1', '160906_band2'
    ],
    cam_list=[(0, 12), (0, 6), (0, 23), (0, 13), (0, 3)],
    num_cameras=num_cameras,
    seq_frame_interval=3,
    subset='train_debug',
    root_id=2,
    max_num=10,
    space_size=space_size,
    space_center=space_center,
    cube_size=cube_size,
)

test_data_cfg = train_data_cfg.copy()
test_data_cfg.update(
    dict(
        seq_list=[
            '160906_pizza1',
            '160422_haggling1',
            '160906_ian5',
            '160906_band4',
        ],
        seq_frame_interval=12,
        subset='validation_debug'))

# model settings
backbone = dict(type='ResNet', depth=50)
keypoint_head = dict(
        type='CustomDeconvHead',
        feature_extract_layers=[5, 8],
        in_channels=2048,
        out_channels=num_joints,
        num_deconv_layers=3,
        num_deconv_filters=(256, 256, 256),
        num_deconv_kernels=(4, 4, 4),
        loss_keypoint=dict(
            type='CustomHeatmapLoss',
            loss_weight=1.0)
)

model = dict(
    type='GraphBasedModel',
    num_joints=15,
    backbone=backbone,
    freeze_keypoint_head=False,
    freeze_2d=True,
    keypoint_head=keypoint_head,
    pretrained='checkpoints/resnet_50_deconv.pth.tar',
    human_detector=dict(
        type='GraphCenterDetection',
        match_cfg=dict(type='MultiViewMatchModule',
                       feature_map_size=heatmap_size,
                       match_gcn=dict(type='EdgeConvLayers',
                                      node_channels=512+num_joints,
                                      edge_channels=1,
                                      node_out_channels=None,
                                      edge_out_channels=1,
                                      mid_channels=256),
                       num_cameras=num_cameras,
                       match_threshold=0.5,
                       cfg_2d=dict(center_index=2,
                                   nms_kernel=5,
                                   val_threshold=0.3,
                                   dist_threshold=5,
                                   max_persons=10,
                                   center_channel=512 + 2,
                                   dist_coef=10),
                       cfg_3d=dict(space_size=space_size,
                                   space_center=space_center,
                                   cube_size=cube_size,
                                   dist_threshold=300)
                       ),
        refine_cfg=dict(type='CenterRefinementModule',
                        project_layer=dict(feature_map_size=heatmap_size),
                        center_gcn=dict(type='EdgeConvLayers',
                                        node_edge_merge='add',
                                        edge_channels=None,
                                        edge_out_channels=None,
                                        node_channels=512+num_joints,
                                        node_out_channels=None,
                                        mid_channels=256,
                                        ),
                        score_threshold=0.5,
                        max_persons=10, max_pool_kernel=5,
                        cfg_3d=dict(space_size=space_size,
                                    space_center=space_center,
                                    cube_size=cube_size,
                                    search_radiance=[200],
                                    search_step=[50])
                        ),
        train_cfg=dict(match_loss_weight=1.0,
                       center_loss_weight=1.0)),
    pose_regressor=dict(
        type='VoxelSinglePose',
        image_size=image_size,
        heatmap_size=heatmap_size,
        sub_space_size=sub_space_size,
        sub_cube_size=sub_cube_size,
        num_joints=15,
        pose_net=dict(type='V2VNet', input_channels=15, output_channels=15),
        pose_head=dict(type='CuboidPoseHead', beta=100.0)),
    pose_refiner=dict(type='PoseRegressionModule',
                      pose_noise=18.0,
                      pose_topology=dataset_info,
                      num_joints=num_joints,
                      num_cameras=num_cameras,
                      feature_map_size=heatmap_size,
                      mid_channels=512,
                      reg_loss=dict(type='L1Loss',
                                    use_target_weight=True, loss_weight=0.10),
                      cls_loss=dict(type='BCELoss',
                                    loss_weight=0.05),
                      multiview_gcn=dict(type='EdgeConvLayers',
                                         node_channels=512,
                                         mid_channels=512,
                                         node_out_channels=512,
                                         norm_type='BN1d',
                                         edge_channels=None,
                                         edge_out_channels=None,
                                         node_edge_merge='add',
                                         node_layers=[2, 2],
                                         edge_layers=[0, 0],
                                         residual=True
                                         ),
                      pose_gcn=dict(type='EdgeConvLayers',
                                    node_channels=512,
                                    mid_channels=512,
                                    node_out_channels=512,
                                    node_edge_merge='add',
                                    residual=True,
                                    norm_type='BN1d',
                                    edge_channels=None,
                                    edge_out_channels=None,
                                    edge_layers=[0, 0, 0, 0],
                                    node_layers=[2, 2, 2, 2],
                                    ),
                      test_cfg=dict(cls_thr=0.5),
                      space_size=space_size,
                      space_center=space_center
                      ))

train_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='BottomUpRandomAffine',
                rot_factor=0,
                scale_factor=[1.0, 1.0],
                scale_type='long',
                trans_factor=0),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
            dict(
                type='BottomUpGenerateHeatmapTarget',
                sigma=2),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
            'num_persons', 'sample_id'
        ]),
    dict(type='GenerateCenterPairs',
         center_index=2,
         disturbance=3,
         camera_type='CustomSimpleCamera',
         dist_coef=10),
    dict(type='GenerateCenterCandidates',
         sample_sigma=[400.0, 100.0, 800.0],
         sample_type=['gaussian']*2 + ['uniform'],
         dist_sigma=200,
         samples_per_person=[100, 100, 100],
         pos_to_neg_ratio=4,
         max_total_samples=1000
         ),
    dict(
        type='Collect',
        keys=['img', 'target', 'mask', 'center_candidates', 'match_graph'],
        meta_keys=[
            'num_persons', 'joints_3d', 'camera', 'center', 'scale',
            'joints_3d_visible', 'roots_3d', 'joints', 'person_ids'
        ]),
]

val_pipeline = [
    dict(
        type='MultiItemProcess',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='BottomUpRandomAffine',
                rot_factor=0,
                scale_factor=[1.0, 1.0],
                scale_type='long',
                trans_factor=0),
            dict(type='ToTensor'),
            dict(
                type='NormalizeTensor',
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        ]),
    dict(
        type='DiscardDuplicatedItems',
        keys_list=[
            'joints_3d', 'joints_3d_visible', 'ann_info', 'roots_3d',
            'num_persons', 'sample_id'
        ]),
    dict(
        type='Collect',
        keys=['img'],
        meta_keys=['sample_id', 'camera', 'center', 'scale', 'ann_info']),
]

test_pipeline = val_pipeline

data_root = 'data/panoptic/'
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    val_dataloader=dict(samples_per_gpu=2),
    test_dataloader=dict(samples_per_gpu=2),
    train=dict(
        type='CustomPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=train_data_cfg,
        pipeline=train_pipeline,
        dataset_info={{_base_.dataset_info}}),
    val=dict(
        type='CustomPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=val_pipeline,
        dataset_info={{_base_.dataset_info}}),
    test=dict(
        type='CustomPanopticDataset',
        ann_file=None,
        img_prefix=data_root,
        data_cfg=test_data_cfg,
        pipeline=test_pipeline,
        dataset_info={{_base_.dataset_info}}),
)
