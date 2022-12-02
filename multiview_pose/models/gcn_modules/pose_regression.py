import torch
import torch.nn as nn
import torch.nn.functional as F
from .builder import GCNS
from itertools import combinations
from .utils import NonGridProjectLayer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmpose.models import LOSSES


@GCNS.register_module()
class PoseRegressionModule(nn.Module):
    def __init__(self,
                 pose_topology,
                 num_joints,
                 num_cameras,
                 feature_map_size,
                 mid_channels,
                 multiview_gcn,
                 pose_gcn,
                 space_size,
                 space_center,
                 reg_loss,
                 cls_loss,
                 pose_noise=0.0,
                 train_cfg=dict(),
                 test_cfg=dict(cls_thr=0.3),
                 detach_feature=False):
        super(PoseRegressionModule, self).__init__()
        self.multiview_gcn = GCNS.build(multiview_gcn)
        self.pose_gcn = GCNS.build(pose_gcn)
        self.num_joints = num_joints
        self.num_cameras = num_cameras
        self.pose_topology = pose_topology
        self.mid_channels = mid_channels
        self.project = NonGridProjectLayer(feature_map_size=feature_map_size)
        self.register_buffer('space_size', torch.tensor(space_size))
        self.register_buffer('space_center', torch.tensor(space_center))
        self.coordinates_proj = nn.Linear(3, mid_channels)
        self.joint_type_proj = nn.Linear(num_joints, mid_channels)
        self.coordinates_reg = nn.Linear(mid_channels, 3)
        self.cls_layer = nn.Sequential(nn.Linear(mid_channels, 1),
                                       nn.Sigmoid())
        self._init_graph(pose_topology)
        self.reg_loss = LOSSES.build(reg_loss)
        self.cls_loss = LOSSES.build(cls_loss)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.detach_feature = detach_feature
        self.pose_noise = pose_noise

    def clamp_xyzs(self, xyzs):
        return torch.max(torch.min(xyzs, self.space_center + 0.5 * self.space_size),
                         self.space_corner)

    def coordinate_normalize(self, tensor):
        if tensor.shape[-1] == 2:
            tensor = (tensor - self.space_corner[:2]) / self.space_size[:2]
        else:
            tensor = (tensor - self.space_corner) / self.space_size
        tensor = torch.clamp(tensor, max=1.0, min=0.0)

        return tensor

    def coordinate_denormalize(self, tensor):
        if tensor.shape[-1] == 2:
            return tensor * self.space_size[:2] + self.space_corner[:2]
        else:
            return tensor * self.space_size + self.space_corner

    @property
    def space_corner(self):
        return self.space_center - 0.5 * self.space_size

    def _init_graph(self, pose_topology):
        keypoint_info = pose_topology['keypoint_info']
        skeleton_info = pose_topology['skeleton_info']
        keypoint_name2id = {v['name']: k for k, v in keypoint_info.items()}

        joint_types_onehot = F.one_hot(torch.arange(self.num_joints)).float()
        self.register_buffer('joint_types_onehot', joint_types_onehot)

        edge_ids = []
        for skt in skeleton_info.values():
            link = skt['link']
            edge_ids.append([keypoint_name2id[link[0]],
                             keypoint_name2id[link[1]]])

        skeleton = torch.tensor(edge_ids, dtype=torch.long)   # num_links x 2
        self.register_buffer('single_skeleton', skeleton)
        multiview_skeleton = skeleton[:, None] * self.num_cameras + torch.arange(
            self.num_cameras).view(1, -1, 1)      # num_links x num_cameras x 2
        self.register_buffer('multiview_skeleton', multiview_skeleton)

        camera_pairs = torch.tensor(
            list(combinations(range(self.num_cameras), 2)), dtype=torch.long)
        crossview_connection = camera_pairs[None] + torch.arange(
            self.num_joints).view(-1, 1, 1) * self.num_cameras    # num_joints x num_cameras x 2

        self.register_buffer('crossview_connection', crossview_connection)

    def _build_multiview_pose_graph(self, num_persons, device):
        # num_links x 2
        num_nodes_per_person = self.num_joints * self.num_cameras
        crossview_connection = self.crossview_connection[None] + torch.arange(
            num_persons, device=device).view(-1, 1, 1, 1) * num_nodes_per_person
        multiview_skeleton = self.multiview_skeleton[None] + torch.arange(
            num_persons, device=device).view(-1, 1, 1, 1) * num_nodes_per_person

        return crossview_connection.flatten(0, 2), multiview_skeleton.flatten(0, 2)

    def _build_pooled_pose_graph(self, num_persons, device):
        skeleton = self.single_skeleton[None] + torch.arange(
            num_persons, device=device).view(-1, 1, 1) * self.num_joints

        return skeleton.flatten(0, 1)

    def forward_share(self, poses, feature_maps, img_metas):
        batch_size, num_persons, num_joints = poses.shape[:3]

        multiview_features, _, _ = self.project(feature_maps, img_metas, poses.flatten(1, 2), False)
        multiview_features = torch.cat([f.flatten(0, 1) for f in multiview_features], dim=0)
        device = multiview_features.device
        multiview_features = multiview_features.view(-1, self.num_joints, self.num_cameras, self.mid_channels)
        multiview_features = multiview_features \
                             + self.joint_type_proj(self.joint_types_onehot).view(1, self.num_joints, 1, -1)
        multiview_features = multiview_features.view(-1, self.mid_channels)
        # num_persons*num_joints*num_cameras x C
        crossview_connection, _ = self._build_multiview_pose_graph(batch_size * num_persons,
                                                                                    device=device)
        multiview_edge_indices = torch.cat([crossview_connection, crossview_connection.flip([1])]).T

        normed_coordinates = self.coordinate_normalize(poses[..., :3].view(-1, 3))
        positional_emb = self.coordinates_proj(normed_coordinates)[:, None].repeat(1, self.num_cameras, 1)
        multiview_features = multiview_features + positional_emb.flatten(0, 1)

        multiview_features, _ = self.multiview_gcn(multiview_features,
                                                   multiview_edge_indices,
                                                   None)

        multiview_features = multiview_features.view(-1, self.num_cameras, multiview_features.shape[-1])
        keypoint_features = multiview_features.sum(1)
        edge_indices = self._build_pooled_pose_graph(batch_size * num_persons, device=device)
        edge_indices = torch.cat([edge_indices, edge_indices.flip([1])]).T

        keypoint_features, _ = self.pose_gcn(keypoint_features,
                                             edge_indices, None)
        coordinate_logits = inverse_sigmoid(normed_coordinates, 1e-12)
        regression = self.coordinates_reg(keypoint_features)

        new_coordinates = (coordinate_logits + regression).sigmoid()

        cls_scores = self.cls_layer(
            keypoint_features)[..., 0].view(
            batch_size, num_persons, num_joints).mean(-1)

        return new_coordinates.view(batch_size, num_persons, num_joints, 3), cls_scores

    def forward_train(self, poses, feature_maps, img_metas):
        poses = poses.detach()
        # add noise
        poses_noise = (2 * torch.rand(poses[..., :3].shape) - 1) * self.pose_noise
        poses[..., :3] = self.clamp_xyzs(poses[..., :3] + poses_noise.to(poses.device))
        feature_maps = torch.cat([f[:, None, :self.mid_channels] for f in feature_maps], dim=1)
        if self.detach_feature:
            feature_maps = feature_maps.detach()
        refined_poses, cls_scores = self.forward_share(poses, feature_maps, img_metas)
        refined_poses = self.coordinate_denormalize(refined_poses)
        device = refined_poses.device

        gt_3d = torch.stack([
            torch.tensor(img_meta['joints_3d'], device=device)
            for img_meta in img_metas
        ])
        gt_3d_vis = torch.stack([
            torch.tensor(img_meta['joints_3d_visible'], device=device)
            for img_meta in img_metas
        ])
        # regression
        targets = []
        target_weights = []
        valid_poses = []
        batch_size = gt_3d.shape[0]
        for i in range(batch_size):
            matched_gt_ids = poses[i, :, 0, 3].long()
            valid_preds = matched_gt_ids >= 0
            matched_gt_ids = matched_gt_ids[valid_preds]
            targets.append(gt_3d[i, matched_gt_ids])
            target_weights.append(gt_3d_vis[i, matched_gt_ids])
            valid_poses.append(refined_poses[i, valid_preds])
        loss_pose_refine = self.reg_loss(torch.cat(valid_poses),
                                         torch.cat(targets),
                                         torch.cat(target_weights))

        # classification
        cls_targets = torch.zeros_like(cls_scores)
        cls_targets[poses[..., 0, 3].long() >= 0] = 1.0

        loss_pose_cls = self.cls_loss(cls_scores, cls_targets)

        return dict(loss_pose_refine=loss_pose_refine,
                    loss_pose_cls=loss_pose_cls)

    def forward_test(self, poses, feature_maps, img_metas):
        poses = poses.detach()
        feature_maps = torch.cat([f[:, None, :self.mid_channels] for f in feature_maps], dim=1)
        refined_poses, cls_scores = self.forward_share(poses, feature_maps, img_metas)
        refined_poses = self.coordinate_denormalize(refined_poses)

        poses[..., :3] = refined_poses
        new_pose_valid = torch.logical_and(poses[..., 0, 3] >= 0,
                                           cls_scores >= self.test_cfg['cls_thr'])

        new_pose_valid = new_pose_valid.float() - 1.0
        poses[..., 3] = new_pose_valid[..., None]
        poses[..., 4] = cls_scores[..., None]

        return poses
