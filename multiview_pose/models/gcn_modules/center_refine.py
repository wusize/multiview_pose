import torch
import torch.nn as nn
from mmpose.models import builder
from itertools import permutations
from .builder import GCNS
from .utils import compute_grid, NonGridProjectLayer

@GCNS.register_module()
class CenterRefinementModule(nn.Module):
    def __init__(self,
                 project_layer, center_gcn, score_threshold=0.5,
                 max_persons=10, max_pool_kernel=5,
                 cfg_3d=dict(space_size=[8000, 8000, 2000],
                             space_center=[0, -500, 800],
                             cube_size=[80, 80, 20],
                             search_radiance=[200],
                             search_step=[50]),
                 train_cfg=dict(search_radiance=200,
                                search_step=50,
                                noise=30.0,
                                dist_thr=500.0,
                                neg_temp=0.002
                                )):
        super(CenterRefinementModule, self).__init__()
        self.cfg_3d = cfg_3d.copy()
        self.project_layer = NonGridProjectLayer(**project_layer)
        self.center_gcn = GCNS.build(center_gcn)
        mid_channels = self.center_gcn.mid_channels
        self.final_layer = nn.Sequential(nn.Linear(mid_channels, mid_channels),
                                         nn.LayerNorm(mid_channels),
                                         nn.ReLU(),
                                         nn.Linear(mid_channels, 1))
        self.loss = nn.MSELoss(reduction='none')
        self.score_threshold = score_threshold

        cubic_nms = dict(
            type='CuboidCenterHead',
            space_size=self.cfg_3d['space_size'],
            space_center=self.cfg_3d['space_center'],
            cube_size=self.cfg_3d['cube_size'],
            max_num=max_persons,
            max_pool_kernel=max_pool_kernel)
        self.cubic_nms = builder.build_head(cubic_nms)
        self.num_candidates = max_persons
        self.train_cfg = train_cfg

        self._get_3d_space_attributes()

    def _get_3d_space_attributes(self):
        self.register_buffer('grid_samples', compute_grid(self.cfg_3d['space_size'],
                                                          self.cfg_3d['space_center'],
                                                          self.cfg_3d['cube_size']))
        lower_bound = torch.tensor(self.cfg_3d['space_center']
                                   ) - 0.5 * torch.tensor(self.cfg_3d['space_size'])
        upper_bound = torch.tensor(self.cfg_3d['space_center']
                                   ) + 0.5 * torch.tensor(self.cfg_3d['space_size'])
        self.register_buffer('lower_bound', lower_bound)
        self.register_buffer('upper_bound', upper_bound)

        # generate sub_search_space
        for i, (r, d) in enumerate(zip(self.cfg_3d['search_radiance'],
                                       self.cfg_3d['search_step'])):
            self.register_buffer(f'queries_{i}',
                                 compute_grid(r * 2, 0, r * 2 // d))
        r = self.train_cfg['search_radiance']
        d = self.train_cfg['search_step']
        self.register_buffer(f'queries_train',
                             compute_grid(r * 2, 0, r * 2 // d))

    def forward(self, feature_maps, img_metas, center_candidates, return_loss=True):
        """

        Args:
            feature_maps:  [NxCxHxW]xV   # NxVxCxHxW
            img_metas:
            center_candidates: [num_candidates_i x 5] i=0:N-1
                or Nxnum_candidates_ix5
            return_loss

        Returns:
            human_centers: NXPX5
        """
        feature_maps = torch.stack(feature_maps, dim=1)
        if return_loss:
            return self.forward_train(feature_maps, img_metas, center_candidates)
        else:
            return self.forward_test(feature_maps, img_metas, center_candidates)

    def forward_train(self, feature_maps, img_metas, center_candidates):
        """

        Args:
            feature_maps: NxVxCxHxW
            img_metas:
            center_candidates: [num_candidates_i x 5] i=0:N-1
                or Nxnum_candidates_ix5
        """
        if not isinstance(center_candidates, list):
            center_candidates_list = []
            for center_candidate in center_candidates:
                center_candidates_list.append(center_candidate[center_candidate[:, -1] > 0])
            center_candidates = center_candidates_list
        feature_maps = feature_maps.detach()
        predicted_center_scores, center_candidates = \
            self.inference(feature_maps, img_metas, center_candidates, True)
        center_candidates = torch.cat(center_candidates)

        losses = dict()
        weights = center_candidates[:, -1:]
        weights = weights / (weights.sum() + 1e-12)
        loss = self.loss(predicted_center_scores, center_candidates[:, -2:-1])
        loss = (loss * weights).sum()

        losses['loss_center'] = loss

        center_candidates_for_next_stage = self.generate_samples_for_next_stage(feature_maps,
                                                                                img_metas)

        return center_candidates_for_next_stage, losses

    def inference(self, feature_maps, img_metas, center_candidates, discard_nan):
        multiview_features, bounding, center_candidates = \
            self.project_layer(feature_maps, img_metas, center_candidates, discard_nan)
        # [P_ixVxC] i in 0:N-1
        node_features, edge_indices = self.build_graph_for_samples(multiview_features)  # PxVxC, PxEx2
        bounding = torch.cat(bounding).view(-1)
        num_samples, num_views, _ = node_features.shape
        node_features = node_features.view(num_samples * num_views, -1)
        edge_indices = edge_indices.view(-1, 2).T
        node_features, _ = self.center_gcn(node_features, edge_indices)
        node_features = node_features.view(num_samples, num_views, -1)
        candidate_features, _ = node_features.max(1)
        predicted_scores = self.final_layer(candidate_features)
        if not discard_nan:
            predicted_scores[bounding.logical_not()] = 0.0

        return predicted_scores, center_candidates  # Px1

    @staticmethod
    def build_graph_for_samples(multiview_features):
        """

        Args:
            multiview_features: [P_ixVxC] i in range(N)

        Returns:
            edge_indices
        """
        # build graph or each sample
        multiview_features = torch.cat(multiview_features)
        num_samples, num_cameras, _ = multiview_features.shape
        device = multiview_features.device
        edge_indices = torch.tensor(list(permutations(range(num_cameras), 2)),
                                    device=device).view(1, -1, 2) + torch.arange(
            num_samples, device=device).view(-1, 1, 1) * num_cameras  # PxEx2

        return multiview_features, edge_indices.long()

    def forward_test(self, feature_maps, img_metas, center_candidates):
        batch_size = len(center_candidates)
        samples_to_query_s = [samples_to_query if len(samples_to_query) > 0
                              else self.grid_samples[:, 0] == self.grid_samples[:, 0]
                              for samples_to_query in center_candidates]
        center_candidates = [self.grid_samples[samples_to_query]
                             for samples_to_query in samples_to_query_s]

        predicted_center_scores, _ \
            = self.inference(feature_maps, img_metas, center_candidates, discard_nan=False)

        heatmap_cubes = feature_maps.new_zeros(batch_size,
                                               self.grid_samples.shape[0])
        cnt = 0
        for i, samples_to_query in enumerate(samples_to_query_s):
            num_samples = (samples_to_query).sum()
            heatmap_cubes[i, samples_to_query] = predicted_center_scores[cnt:cnt + num_samples, 0]
            cnt = cnt + num_samples

        heatmap_cubes = heatmap_cubes.view(batch_size, *self.cfg_3d['cube_size'])
        center_candidates = self.cubic_nms(heatmap_cubes)  # NxPx5
        center_candidates_valid = center_candidates[..., 4] > self.score_threshold  # NxP
        center_candidates[..., 3] = center_candidates_valid.float() - 1.0

        for b in range(batch_size):
            if center_candidates_valid[b].sum() <= 0:
                center_candidates_valid[b][0] = 1

        # refine
        for i in range(len(self.cfg_3d['search_radiance'])):
            queries = getattr(self, f'queries_{i}')
            num_queries = queries.shape[0]
            center_candidates_i = [self.generate_sub_samples(
                center_candidates[b][center_candidates_valid[b], :3],
                queries)
                for b in range(batch_size)]
            center_scores_i, _ = self.inference(feature_maps, img_metas, center_candidates_i, False)
            cnt = 0
            for b in range(batch_size):
                num_candidates = center_candidates_valid[b].sum()
                center_scores_i_b = center_scores_i[
                                    cnt:cnt + num_candidates * num_queries].view(-1, num_queries)
                candidates_scores_i_b, candidates_refined_i_b = center_scores_i_b.max(-1)
                cnt = cnt + num_candidates * num_queries
                center_candidates_i_b = center_candidates_i[b].view(-1, num_queries, 3)
                center_candidates[b][center_candidates_valid[b], :3] = \
                    center_candidates_i_b[range(num_candidates), candidates_refined_i_b]
                center_candidates[b][center_candidates_valid[b], 4] = candidates_scores_i_b

        return center_candidates

    def generate_sub_samples(self, center_candidates, queries):
        samples = (center_candidates[:, None] + queries[None]).view(-1, 3)
        samples = torch.where(samples > self.upper_bound,
                              self.upper_bound, samples)
        samples = torch.where(samples < self.lower_bound,
                              self.upper_bound, samples)

        return samples

    def generate_negative_samples(self, num, gts):
        dist_thr = self.train_cfg['dist_thr']
        negative_samples = self.grid_samples
        dist2gts = (negative_samples[:, None] - gts[None]).norm(dim=-1)
        dist2gts, _ = dist2gts.min(-1)
        negative_samples = negative_samples[dist2gts > dist_thr]
        dist2gts = dist2gts[dist2gts > dist_thr]
        possibilities = torch.exp(-torch.abs(dist2gts - (2 * dist_thr)) * self.train_cfg['neg_temp'])

        sampled = torch.multinomial(possibilities, min(num,
                                                       possibilities.shape[0]))

        return negative_samples[sampled]

    @torch.no_grad()
    def generate_samples_for_next_stage(self, feature_maps, img_metas):
        device = feature_maps.device
        # Add noises to gt centers
        batch_size = len(img_metas)
        gt_centers_origin = torch.stack([
            torch.tensor(img_meta['roots_3d'], device=device)
            for img_meta in img_metas
        ])  # BxPx3
        gt_num_persons = [
            img_meta['num_persons'] for img_meta in img_metas
        ]
        gt_centers = gt_centers_origin + (
                2 * torch.rand(*gt_centers_origin.shape).to(device) - 1.0) * self.train_cfg['noise']
        queries = self.queries_train
        num_queries = queries.shape[0]
        center_candidates = [self.generate_sub_samples(
            gt_centers[b][:gt_num_persons[b]], queries) for b in range(batch_size)]
        center_candidates_scores, _ = self.inference(feature_maps, img_metas, center_candidates, False)
        num_candidates_per_frame = [num * num_queries for num in gt_num_persons]
        center_candidates_scores_per_frame = center_candidates_scores.split(num_candidates_per_frame,
                                                                            dim=0)
        max_num_persons = max(self.num_candidates, max(gt_num_persons))
        human_centers = -torch.ones(
            batch_size, max_num_persons, 5, device=device)

        for b in range(batch_size):
            candidates_scores_b, candidates_max_b = \
                center_candidates_scores_per_frame[b].view(-1, num_queries).max(-1)
            center_candidates_b = center_candidates[b].view(-1, num_queries, 3)
            human_centers[b, :gt_num_persons[b], :3] = center_candidates_b[
                range(gt_num_persons[b]), candidates_max_b]
            human_centers[b, :gt_num_persons[b], 3] = torch.arange(gt_num_persons[b], device=device)  # gt_id
            human_centers[b, :gt_num_persons[b], 4] = candidates_scores_b
            if gt_num_persons[b] < max_num_persons:
                negative_samples = self.generate_negative_samples(max_num_persons - gt_num_persons[b],
                                                                  gt_centers_origin[b, :gt_num_persons[b]])
                human_centers[b, gt_num_persons[b]:, :3] = negative_samples

        return human_centers

