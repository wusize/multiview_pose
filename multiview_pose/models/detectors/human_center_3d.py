import torch
import torch.nn as nn
import torch.nn.functional as F
from mmpose.models import builder
from multiview_pose.core.camera import CustomSimpleCameraTorch as SimpleCameraTorch
from mmpose.core.post_processing.post_transforms import (
    affine_transform_torch, get_affine_transform)
from mmpose.models.builder import MODELS


def compute_grid(space_size, space_center, cube_size):
    if isinstance(space_size, int) or isinstance(space_size, float):
        space_size = [space_size, space_size, space_size]
    if isinstance(space_center, int) or isinstance(space_center, float):
        space_center = [space_center, space_center, space_center]
    if isinstance(cube_size, int):
        cube_size = [cube_size, cube_size, cube_size]

    grid_1D_x = torch.linspace(
        -space_size[0] / 2, space_size[0] / 2, cube_size[0])
    grid_1D_y = torch.linspace(
        -space_size[1] / 2, space_size[1] / 2, cube_size[1])
    grid_1D_z = torch.linspace(
        -space_size[2] / 2, space_size[2] / 2, cube_size[2])
    grid_x, grid_y, grid_z = torch.meshgrid(
        grid_1D_x + space_center[0],
        grid_1D_y + space_center[1],
        grid_1D_z + space_center[2],
    )
    grid_x = grid_x.contiguous().view(-1, 1)
    grid_y = grid_y.contiguous().view(-1, 1)
    grid_z = grid_z.contiguous().view(-1, 1)
    grid = torch.cat([grid_x, grid_y, grid_z], dim=1)

    return grid


class NonGridProjectLayer(nn.Module):
    def __init__(self, feature_map_size):
        """Project layer to get multi-view features.
        Args:
            cfg (dict):
                image_size: input size of the 2D model
                feature_map_size: output size of the 2D model
        """
        super(NonGridProjectLayer, self).__init__()
        # image_size = cfg['image_size']
        # feature_map_size = cfg['feature_map_size']

        # if isinstance(image_size, int):
        #     image_size = [image_size, image_size]
        if isinstance(feature_map_size, int):
            feature_map_size = [feature_map_size, feature_map_size]
            
        # self.register_buffer('image_size', torch.tensor(image_size))
        self.register_buffer('feature_map_size', torch.tensor(feature_map_size))

    def forward(self, feature_maps, meta, multiview_sample_points, discard_nan=True):
        """

        Args:
            feature_maps: NxVxCxHxW
            meta:
            multiview_sample_points: [num_candidates_i x 5] i=0:N-1

        Returns:

        """
        device = feature_maps.device
        batch_size, num_cameras, num_channels = feature_maps.shape[:3]
        multiview_features = []
        bounding = []
        for sample_points in multiview_sample_points:
            # multiview_features.append(torch.zeros(num_cameras, num_channels,
            #                                       sample_points.shape[0], device=device))
            bounding.append(torch.ones(num_cameras, 1,
                                       sample_points.shape[0], device=device))
        # w, h = self.feature_map_size[0].item(), self.feature_map_size[1].item()
        h, w = feature_maps.shape[-2:]
        for i, sample_points in enumerate(multiview_sample_points):
            multiview_sample_points_norm = []
            for c in range(num_cameras):
                center = meta[i]['center'][c]
                scale = meta[i]['scale'][c]
                width, height = center * 2

                trans = torch.as_tensor(
                    get_affine_transform(center, scale / 200.0, 0,
                                         [w, h]),
                    dtype=torch.float,
                    device=device)

                cam_param = meta[i]['camera'][c].copy()

                single_view_camera = SimpleCameraTorch(
                    param=cam_param, device=device)
                xy = single_view_camera.world_to_pixel(sample_points[:, :3])

                bounding[i][c, 0] *= (xy[:, 0] >= 0
                                      ) & (xy[:, 1] >= 0
                                           ) & (xy[:, 0] < width) & (xy[:, 1] < height)
                sample_points_pixel_ = torch.clamp(xy, -1.0,
                                                  max(width, height))
                sample_points_pixel = affine_transform_torch(sample_points_pixel_, trans)
                # sample_points_pixel = sample_points_pixel * self.feature_map_size[
                #     None].float() / self.image_size[None].float()
                sample_points_norm = sample_points_pixel / (self.feature_map_size[
                    None].float() - 1) * 2.0 - 1.0
                sample_points_norm = torch.clamp(
                    sample_points_norm.view(1, 1, -1, 2), -1.1, 1.1)

                multiview_sample_points_norm.append(sample_points_norm)
                # multiview_features[i][c] = F.grid_sample(
                #     feature_maps[c][i:i + 1],
                #     sample_points_norm,
                #     align_corners=True)[0]
            multiview_sample_points_norm = torch.cat(multiview_sample_points_norm, dim=0)  # Vx1xPx2
            multiview_features.append(F.grid_sample(
                feature_maps[i],
                multiview_sample_points_norm,
                align_corners=True)[:, :, 0])   # [(VxCxP)]

        for i, multiview_feature in enumerate(multiview_features):
            multiview_features[i] = (multiview_feature *
                                     bounding[i]).permute(2, 0, 1).contiguous()
            is_not_nan = multiview_features[i].isnan().sum([1, 2]).ge(1).logical_not()
            # is_nan = multiview_features[i].isnan().sum([1, 2]).ge(1)
            if discard_nan:
                multiview_features[i] = multiview_features[i][is_not_nan]
                bounding[i] = bounding[i][:, 0, is_not_nan].sum(0) > 0
                multiview_sample_points[i] = multiview_sample_points[i][is_not_nan]
            else:
                bounding[i] = (bounding[i][:, 0].sum(0) > 0) * is_not_nan

        return multiview_features, bounding, multiview_sample_points


@MODELS.register_module()
class GraphCenterDetection(nn.Module):
    """
    Match and refinement. The matching module matches 2D human centers
    and generate coarse 3D human center candidates. The refinement
    module generate refined 3D human centers upon the coarse candidates.
    """
    def __init__(self, match_cfg, refine_cfg, train_cfg):
        super(GraphCenterDetection, self).__init__()
        self.match_module = builder.MODELS.build(match_cfg)
        self.refine_module = builder.MODELS.build(refine_cfg)
        self.match_loss_weight = train_cfg['match_loss_weight']
        self.center_loss_weight = train_cfg['center_loss_weight']

    def forward(self,
                img,
                img_metas,
                return_loss=True,
                feature_maps=None,
                targets_3d=None,
                match_graph=None,
                center_candidates=None):
        if return_loss:
            return self.forward_train(None, img_metas, feature_maps, match_graph=match_graph,
                                      center_candidates=center_candidates)
        else:
            return self.forward_test(None, img_metas, feature_maps)

    def forward_train(self,
                      img,
                      img_metas,
                      feature_maps,
                      targets_3d=None,
                      return_preds=None,
                      match_graph=None, center_candidates=None,
                      **kwargs):
        losses = dict()
        match_losses = self.match_module(feature_maps, img_metas,
                                         True, match_graph)
        human_centers, center_losses = self.refine_module(feature_maps, img_metas,
                                                          center_candidates, True)
        for k, v in match_losses.items():
            losses[k] = v * self.match_loss_weight
        for k, v in center_losses.items():
            losses[k] = v * self.center_loss_weight

        if return_preds:
            return human_centers, losses
        else:
            return losses

    def forward_test(self, img, img_metas, feature_maps=None, **kwargs):
        # tik = time()
        center_candidates = self.match_module(feature_maps, img_metas, False, None)
        # print(f'multiview match: {time() - tik}', flush=True)
        # tik = time()
        center_candidates = self.refine_module(feature_maps, img_metas, center_candidates, False)
        # print(f'center refinement: {time() - tik}', flush=True)
        return center_candidates
