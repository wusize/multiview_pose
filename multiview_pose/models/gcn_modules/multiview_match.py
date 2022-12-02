import torch
import torch.nn as nn
import torch.nn.functional as F
from multiview_pose.core.camera import CustomSimpleCameraTorch as SimpleCameraTorch
from multiview_pose.core.post_processing.post_transforms import transform_preds_torch
from multiview_pose.core.camera.utils import StereoGeometry, MonocularGeometry
from itertools import combinations, product
from .utils import compute_grid
from .builder import GCNS


@GCNS.register_module()
class MultiViewMatchModule(nn.Module):
    def __init__(self, feature_map_size, match_gcn, num_cameras=5,
                 match_threshold=0.5,
                 cfg_2d=dict(center_index=2,
                             nms_kernel=5,
                             val_threshold=0.3,
                             dist_threshold=5,
                             max_persons=10,
                             center_channel=512 + 2,
                             dist_coef=10),
                 cfg_3d=dict(space_size=[8000, 8000, 2000],
                             space_center=[0, -500, 800],
                             cube_size=[80, 80, 20],
                             dist_threshold=300)):
        self.cfg_2d = cfg_2d.copy()
        self.cfg_3d = cfg_3d.copy()
        self.match_threshold = match_threshold
        super(MultiViewMatchModule, self).__init__()
        self.register_buffer('feature_map_size', torch.tensor(feature_map_size))
        self.register_buffer('grid_samples', compute_grid(cfg_3d['space_size'],
                                                          cfg_3d['space_center'],
                                                          cfg_3d['cube_size']))
        self.loss = nn.BCELoss(reduction='none')
        self.match_gcn = GCNS.build(match_gcn)
        self.num_cameras = num_cameras

        # self._initial_edge_indices()
        self.pool = torch.nn.MaxPool2d(cfg_2d['nms_kernel'], 1,
                                       (cfg_2d['nms_kernel'] - 1) // 2)
        self._initialize_connections()

    def _get_sparse_candidates(self, samples_to_query):
        L, W, H = self.cfg_3d['cube_size']
        samples_to_query = samples_to_query.view(L, W, H)
        samples_to_query[::2, ::2, ::2] = 1
        return samples_to_query.view(-1)

    def _initialize_connections(self):
        camera_pairs = torch.tensor(
            list(combinations(range(self.num_cameras), 2)))
        person_pairs = torch.tensor(
            list(product(range(self.cfg_2d['max_persons']), repeat=2)))

        self.register_buffer('camera_pairs', torch.tensor(camera_pairs))
        self.register_buffer('person_pairs', torch.tensor(person_pairs))

    def forward(self, feature_maps, img_metas, return_loss=True, graph=None):
        """

        Args:
            feature_maps: [NCHW]xV -> NxVxCxHxW
            img_metas:
        Returns:
            center_candidates list: [num_candidates_i x 5] i=0:N-1
        """
        feature_maps = torch.stack(feature_maps, dim=1)  # NxVxCxHxW
        if return_loss:
            # debug start
            feature_maps = feature_maps.detach()
            # debug end
            graph = self.build_graph_from_input(feature_maps, graph)
            return self.forward_train(graph)
        else:
            return self.forward_test(feature_maps, img_metas)

    def nms(self, heatmaps):
        """Non-Maximum Suppression for heatmaps.

        Args:
            heatmap(torch.Tensor): Heatmaps before nms.

        Returns:
            torch.Tensor: Heatmaps after nms.
        """

        maxm = self.pool(heatmaps)
        maxm = torch.eq(maxm, heatmaps).float()
        heatmaps = heatmaps * maxm

        return heatmaps

    def top_k(self, heatmaps):
        """Find top_k values in an image.
        Args:
            heatmaps (torch.Tensor[NxVxHxW])

        Return:
        """
        heatmaps = self.nms(heatmaps)
        N, V, H, W = heatmaps.shape
        heatmaps = heatmaps.view(N, V, -1)
        val_k, ind = heatmaps.topk(self.cfg_2d['max_persons'],
                                   dim=2)  # NxVxP
        x = ind % W
        y = ind // W
        ind_k = torch.stack((x, y), dim=3)  # NxVxPx2

        return ind_k, val_k

    def forward_train(self, graph):
        edge_indices = graph['edge_indices'][graph['edge_valid'][:, 0] > 0].long()
        edge_scores = graph['edge_scores'][graph['edge_valid'][:, 0] > 0]
        edge_labels = graph['edge_labels'][graph['edge_valid'][:, 0] > 0]
        if (graph['edge_valid'][:, 0] > 0).sum() >= 1:
            _, preds = self.match_gcn(graph['node_features'], edge_indices, edge_scores)
            preds = preds.sigmoid()
            return dict(loss_match=self.get_loss(preds, edge_labels))

        else:
            return dict(loss_match=graph['edge_scores'].new_zeros(1)[0])

    def get_loss(self, preds, edge_labels):
        num_positives = edge_labels.sum()
        num_samples = edge_labels.shape[0]
        num_negatives = num_samples - num_positives
        loss = self.loss(preds, edge_labels)
        if (edge_labels > 0).sum() < 1:
            print(edge_labels.shape, flush=True)
        ones = torch.ones_like(edge_labels)
        mask = torch.where(edge_labels > 0, ones / (num_positives + 1e-12),
                           ones / (num_negatives + 1e-12))

        return (loss * mask).sum()

    def build_graph_from_input(self, feature_maps, graph):
        """

        Args:
            feature_maps: NxVxCxHxW
            graph:
                multiview_centers: Nx(VxP)x3
                edge_indices: NxEx2
                edge_scores: NxE
                edge_valid: NxE
                edge_labels: NxE

        Returns:
        """
        batch_size, num_nodes, _ = graph['multiview_centers'].shape
        for i in range(batch_size):
            graph['edge_indices'][i] = graph['edge_indices'][i] + num_nodes * i

        graph['edge_indices'] = graph['edge_indices'].view(-1, 2)
        graph['edge_scores'] = graph['edge_scores'].view(-1, 1)
        graph['edge_valid'] = graph['edge_valid'].view(-1, 1)
        graph['edge_labels'] = graph['edge_labels'].view(-1, 1)

        batch_size, num_cameras, num_channels, height, width = feature_maps.shape
        feature_maps = feature_maps.view(-1, num_channels, height, width)
        multiview_centers = graph['multiview_centers'].view(
            batch_size * num_cameras, 1, -1, 3)
        multiview_centers_norm = multiview_centers[..., :2] / (
                self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0

        features = F.grid_sample(feature_maps, multiview_centers_norm,
                                 align_corners=True)[:, :, 0]

        node_features = features.transpose(-1, -2).contiguous().view(-1, num_channels)
        graph['node_features'] = node_features

        return graph

    def forward_test(self, feature_maps, img_metas):
        device = feature_maps.device
        centers_pixel, center_values = self.top_k(
            feature_maps[:, :, self.cfg_2d['center_channel']])
        center_values = torch.where(
            center_values < self.cfg_2d['val_threshold'],
            torch.zeros_like(center_values), center_values)

        batch_size, num_cameras, num_persons = center_values.shape
        num_channels, height, width = feature_maps.shape[2:]
        feature_maps = feature_maps.view(-1, num_channels, height, width)
        multiview_centers = centers_pixel.view(batch_size * num_cameras,
                                               1, -1, 2)
        multiview_centers_norm = multiview_centers / (
                self.feature_map_size.view(1, 1, 1, 2).float() - 1) * 2.0 - 1.0
        features = F.grid_sample(feature_maps, multiview_centers_norm,
                                 align_corners=True)[:, :, 0]

        node_features = features.transpose(-1, -2).contiguous().view(-1, num_channels)

        camera_list = [SimpleCameraTorch(param=camera_param.copy(), device=device)
                       for camera_param in img_metas[0]['camera']]

        monocular_geometries = [MonocularGeometry(
            transform_preds_torch(centers_pixel[:, i].contiguous().view(-1, 2),
                                  img_metas[0]['center'][0],
                                  img_metas[0]['scale'][0] / 200.0,
                                  self.feature_map_size),
            camera_list[i]) for i in range(num_cameras)]
        edge_indices = []
        stereo_geometries = []
        edge_valid = []
        for camera_pair in self.camera_pairs:
            nodes_valid = center_values[:, camera_pair] > 0  # batch_size x 2 x num_persons
            nodes_valid_src, nodes_valid_tar = nodes_valid[:, 0], nodes_valid[:, 1]
            masks = torch.stack([self.person_pairs + i * num_persons
                                 for i in range(batch_size)], dim=0)  # batch_size x E x 2
            masks_src, masks_tar = masks[..., 0], masks[..., 1]
            # tik = time()
            stereo_geometry = StereoGeometry(
                monocular_geometries[camera_pair[0].item()],
                monocular_geometries[camera_pair[1].item()],
                masks_src.view(-1),
                masks_tar.view(-1)
            )
            edge_valid_ = nodes_valid_src[:, masks_src[0]] * nodes_valid_tar[:, masks_tar[0]]

            edge_index = torch.cat([self.person_pairs +
                                    camera_pair[None] * num_persons +
                                    i * num_persons * num_cameras
                                    for i in range(batch_size)], dim=0)  # batch_size * E x 2
            edge_indices.append(edge_index)
            edge_valid.append(edge_valid_.contiguous().view(-1))
            stereo_geometries.append(stereo_geometry)
        edge_scores_1to2 = torch.cat([torch.exp(-self.cfg_2d['dist_coef']
                                                * stereo_geometry.distance_1to2)
                                      for stereo_geometry in stereo_geometries], dim=0)  # num_pairs * batch_size * E
        edge_scores_2to1 = torch.cat([torch.exp(-self.cfg_2d['dist_coef']
                                                * stereo_geometry.distance_2to1)
                                      for stereo_geometry in stereo_geometries], dim=0)
        edge_indices = torch.cat(edge_indices, dim=0).long()  # num_pairs * batch_size * E x 2
        edge_valid = torch.cat(edge_valid, dim=0)

        input_edge_indices = torch.cat([edge_indices[edge_valid],
                                        edge_indices[edge_valid].flip([-1])], dim=0)
        input_edge_scores = torch.cat([edge_scores_1to2[edge_valid],
                                       edge_scores_2to1[edge_valid]], dim=0)
        edge_preds = torch.zeros_like(edge_scores_1to2)
        _, preds = self.match_gcn(node_features, input_edge_indices, input_edge_scores[:, None])
        preds = preds.sigmoid()
        edge_preds[edge_valid] = preds.view(2, -1).mean(0)

        stereo_reconstructions = torch.stack([stereo_geometry.reconstructions
                                              for stereo_geometry in stereo_geometries], dim=0)

        match_results = dict(edge_preds=edge_preds,  # num_pairs * batch_size * E
                             edge_valid=edge_valid,
                             node_valid=center_values > 0,  # batch_size x num_cameras x num_persons
                             edge_indices=edge_indices,
                             batch_size=batch_size,
                             num_cameras=num_cameras,
                             num_persons=num_persons,
                             stereo_reconstructions=stereo_reconstructions,  # num_pairs x batch_size * E x 3
                             monocular_geometries=monocular_geometries)  # num_cameras x batch_size*num_persons
        coarse_candidates, pixel_ray_candidates = self.get_coarse_candidates(match_results)
        center_candidates = self.generate_center_candidates(coarse_candidates, pixel_ray_candidates)
        return center_candidates

    def get_coarse_candidates(self, match_results):
        batch_size = match_results['batch_size']
        num_pairs = len(self.camera_pairs)
        edge_preds = match_results['edge_preds'].view(num_pairs, batch_size, -1)  # num_pairs * batch_size * E
        edge_valid = match_results['edge_valid'].view(num_pairs, batch_size, -1)
        node_valid = match_results['node_valid']
        edge_indices = match_results['edge_indices'].view(num_pairs, batch_size, -1, 2)
        stereo_reconstructions = match_results['stereo_reconstructions'].view(num_pairs, batch_size, -1, 3)
        monocular_geometries = match_results['monocular_geometries']
        camera_centers = torch.stack([mono_geo.camera_center.T
                                      for mono_geo in monocular_geometries], dim=0)  # num_cameras x 1 x 3
        ray_directions = torch.stack([mono_geo.ray_direction.T.view(batch_size, -1, 3)
                                      for mono_geo in monocular_geometries], dim=0)
        # num_cameras x batch_size x num_persons x 3

        coarse_candidates = []
        pixel_ray_candidates = []
        unmatched_nodes = node_valid.view(-1)
        for i in range(batch_size):
            edge_preds_i = edge_preds[:, i].contiguous().view(-1)
            edge_valid_i = edge_valid[:, i].contiguous().view(-1)
            edge_indices_i = edge_indices[:, i].contiguous().view(-1, 2)
            src_indices, tar_indices = edge_indices_i[:, 0], edge_indices_i[:, 1]
            matched = edge_valid_i * (edge_preds_i > self.match_threshold)

            unmatched_nodes[src_indices[matched]] = 0
            unmatched_nodes[tar_indices[matched]] = 0

            unmatched_nodes_i = unmatched_nodes.view(batch_size, -1)[i]

            reconstructions_i = stereo_reconstructions[:, i].contiguous().view(-1, 3)
            ray_directions_i = ray_directions[:, i]
            camera_centers_i = torch.zeros_like(ray_directions_i) + camera_centers
            pixel_rays_i = torch.cat([camera_centers_i,
                                      ray_directions_i], dim=-1).view(-1, 6)

            coarse_candidates.append(reconstructions_i[matched])
            pixel_ray_candidates.append(pixel_rays_i[unmatched_nodes_i])

        return coarse_candidates, pixel_ray_candidates

    def generate_center_candidates(self, coarse_candidates, pixel_ray_candidates):
        center_candidates = []
        batch_size = len(coarse_candidates)
        for i in range(batch_size):
            samples_to_query = self.grid_samples[:, 0].logical_not()
            coarse_candidates_i = coarse_candidates[i]
            pixel_ray_candidates_i = pixel_ray_candidates[i]

            if len(coarse_candidates_i) > 0:
                dist_to_candidates_i, _ = torch.norm(self.grid_samples[:, None]
                                                     - coarse_candidates_i[None], dim=-1).min(-1)
                samples_to_query[dist_to_candidates_i < self.cfg_3d['dist_threshold']] = 1

            if len(pixel_ray_candidates_i) > 0:
                s = torch.cross(pixel_ray_candidates_i[None, :, :3]
                                - self.grid_samples[:, None],
                                pixel_ray_candidates_i[None, :, :3]
                                + pixel_ray_candidates_i[None, :, 3:]
                                - self.grid_samples[:, None], dim=-1).norm(dim=-1)  # num_bins x num_rays
                dist_to_rays_i = s / (pixel_ray_candidates_i[None, :, 3:].norm(dim=-1) + 1e-12)
                dist_to_rays_i = dist_to_rays_i.min(-1)[0]
                samples_to_query[dist_to_rays_i < self.cfg_3d['dist_threshold']] = 1
            if samples_to_query.sum().int() == 0:
                samples_to_query = self._get_sparse_candidates(samples_to_query)
            center_candidates.append(samples_to_query)

        return center_candidates
