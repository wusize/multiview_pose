_base_ = ['./gcn_cpn80x80x20_panoptic_cam5_end2end.py']

load_from = None
resume_from = None
optimizer = dict(
    type='Adam',
    lr=0.0001,
)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[7, 8])
total_epochs = 10


# model settings
model = dict(
    freeze_keypoint_head=False,
    freeze_2d=True,
    pretrained='checkpoints/resnet_50_deconv.pth.tar',
    human_detector=dict(
        match_cfg=dict(match_threshold=0.5,
                       cfg_2d=dict(val_threshold=0.3),
                       ),
        refine_cfg=dict(score_threshold=0.5)
    ),
    pose_refiner=dict(pose_noise=15.0,
                      reg_loss=dict(type='L1Loss',
                                    use_target_weight=True, loss_weight=0.10),
                      cls_loss=dict(type='BCELoss',
                                    loss_weight=0.05),
                      test_cfg=dict(cls_thr=0.5)
                      )
)