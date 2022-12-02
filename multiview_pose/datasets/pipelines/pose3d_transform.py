# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.utils import build_from_cfg
from itertools import combinations, product

from mmpose.core.camera import CAMERAS
from mmpose.core.post_processing import transform_preds
from mmpose.datasets.builder import PIPELINES
from multiview_pose.core.camera.utils import calculate_stereo_geometry


@PIPELINES.register_module()
class GenerateCenterCandidates:
    """Generate candidates for 3d human center detection.

    Required keys: 'joints'.
    Modified keys: 'target', and 'target_weight'.

    Args:
        sample_sigma:
        dist_sigma:
        samples_per_person:
        pos_to_neg_ratio:
    """

    def __init__(self,
                 sample_sigma=[400],
                 sample_type=['gaussian'],
                 dist_sigma=200,
                 samples_per_person=[1000],
                 pos_to_neg_ratio=4,
                 max_total_samples=5000):
        if not type(sample_sigma) in [type([]), type(())]:
            sample_sigma = [sample_sigma]
        if not type(sample_type) in [type([]), type(())]:
            sample_type = [sample_type]
        if not type(samples_per_person) in [type([]), type(())]:
            samples_per_person = [samples_per_person]
        assert len(sample_sigma) == len(sample_type)
        assert len(sample_sigma) == len(samples_per_person)
        self.sample_sigma = sample_sigma
        self.sample_type = sample_type
        self.dist_sigma = dist_sigma
        self.samples_per_person = samples_per_person
        self.pos_to_neg_ratio = pos_to_neg_ratio
        self.max_total_samples = max_total_samples

    def __call__(self, results):
        """Generate the target heatmap."""
        center_samples = np.zeros((self.max_total_samples, 5), dtype=np.float32)
        # sample_target_scores = np.zeros(self.max_total_samples, dtype=np.float32)
        # sample_weights = np.zeros(self.max_total_samples, dtype=np.float32)

        centers_3d = results['roots_3d']
        num_persons = results['num_persons']
        space_size = np.array(results['ann_info']['space_size'],
                              dtype=np.float32)
        space_center = np.array(results['ann_info']['space_center'],
                                dtype=np.float32)
        lower_bound = -0.5 * space_size + space_center
        upper_bound = 0.5 * space_size + space_center

        centers_3d = centers_3d[:num_persons]
        positive_samples = list(map(self.sample, self.sample_type,
                                    [num_persons] * len(self.sample_type),
                                    self.samples_per_person,
                                    self.sample_sigma))
        positive_samples = np.concatenate(positive_samples, axis=1) + centers_3d[:, None]

        num_positives = num_persons * sum(self.samples_per_person)
        num_negatives = int(num_positives // self.pos_to_neg_ratio)
        negative_samples = np.random.rand(num_negatives,
                                          3).astype(np.float32) * space_size[None] + lower_bound[None]
        samples = np.concatenate([positive_samples.reshape((-1, 3)),
                                  negative_samples], axis=0)
        np.random.shuffle(samples)

        # Get scores
        dist_squares = ((samples[:, None] - centers_3d[None]) ** 2).sum(-1).min(-1)
        scores = np.exp((- dist_squares / (2 * self.dist_sigma ** 2)))
        weights = ((samples >= lower_bound[None]) * (samples <= upper_bound)).prod(axis=-1)

        num_samples = min(self.max_total_samples, num_positives + num_negatives)

        center_samples[:num_samples, :3] = samples[:num_samples]
        center_samples[:num_samples, 3] = scores[:num_samples]
        center_samples[:num_samples, 4] = weights[:num_samples]

        results['center_candidates'] = center_samples

        return results

    @staticmethod
    def gaussian(num_persons, num_samples, sigma):
        return np.random.randn(num_persons, num_samples, 
                               3).astype(np.float32) * sigma
    
    @staticmethod
    def uniform(num_persons, num_samples, sigma):
        return 2 * np.random.rand(num_persons, num_samples,
                                  3).astype(np.float32) * sigma - sigma

    def sample(self, sample_type, num_persons, num_samples, sigma):
        func = getattr(self, sample_type)
        return func(num_persons, num_samples, sigma)


@PIPELINES.register_module()
class GenerateCenterPairs:
    """

    Args:
        center_index:
        disturbance:
        camera_type:
        dist_coef:
    """
    def __init__(self,
                 center_index,
                 disturbance,
                 camera_type,
                 dist_coef,
                 max_persons=5):
        self.disturbance = disturbance
        self.center_index = center_index
        self.camera_type = camera_type
        assert camera_type in ['CustomSimpleCamera']
        self.dist_coef = dist_coef
        self.max_persons = max_persons

    def _build_camera(self, param):
        cfgs = dict(type=self.camera_type, param=param)
        return build_from_cfg(cfgs, CAMERAS)

    def __call__(self, results):
        """Generate the match graph."""
        heatmap_size = results['ann_info']['heatmap_size'][0]
        if not isinstance(heatmap_size, np.ndarray):
            heatmap_size = np.array(heatmap_size)
        if heatmap_size.size > 1:
            assert len(heatmap_size) == 2
        else:
            heatmap_size = np.array([heatmap_size, heatmap_size])
        total_valid_persons = (results['person_ids'][0] >= 0).sum()
        selected_person_indices = np.arange(self.max_persons) \
            if self.max_persons > total_valid_persons \
            else np.random.choice(total_valid_persons, self.max_persons)

        multiview_centers = np.array([joints[0][selected_person_indices, self.center_index]
                                      for joints in results['joints']], dtype=np.float32)
        image_scales = [scale for scale in results['scale']]
        image_centers = [center for center in results['center']]
        
        multiview_person_ids = np.array([person_ids[selected_person_indices]
                                         for person_ids in results['person_ids']], dtype=np.int64)
        num_cameras, num_persons, _ = multiview_centers.shape

        multiview_centers[..., :2] = multiview_centers[..., :2] + (2 * np.random.rand(
            num_cameras, num_persons, 2) - 1).astype(np.float32) * self.disturbance
        multiview_centers[..., :2][multiview_centers[..., :2] < 0] = 0.0
        multiview_centers[..., :2] = np.where(multiview_centers[..., :2] <= heatmap_size[None, None] - 1,
                                              multiview_centers[..., :2], heatmap_size[None, None] - 1)
        multiview_centers = multiview_centers.reshape((-1, 3))
        multiview_person_ids = multiview_person_ids.flatten()

        # construct matching graph
        camera_pairs = np.array(list(combinations(range(num_cameras), 2)))
        person_pairs = np.array(list(product(range(num_persons), repeat=2)))

        edge_indices = []
        pair_distances = []
        edge_valid = []
        edge_labels = []

        camera_list = [self._build_camera(camera_param)
                       for camera_param in results['camera']]

        for camera_pair in camera_pairs:
            edge_index = person_pairs + camera_pair[None] * num_persons
            human_centers = multiview_centers[edge_index]
            person_ids = multiview_person_ids[edge_index]
            edge_label = person_ids[:, 0] == person_ids[:, 1]
            valid = human_centers[:, 0, 2] * human_centers[:, 1, 2]
            human_centers_1 = transform_preds(human_centers[:, 0, :2],
                                              image_centers[camera_pair[0]],
                                              image_scales[camera_pair[0]] / 200.0,
                                              heatmap_size)
            human_centers_2 = transform_preds(human_centers[:, 1, :2],
                                              image_centers[camera_pair[1]],
                                              image_scales[camera_pair[1]] / 200.0,
                                              heatmap_size)
            distance_1to2, distance_2to1 = calculate_stereo_geometry(
                human_centers_1, camera_list[camera_pair[0]],
                human_centers_2, camera_list[camera_pair[1]])
            # print(distance_1to2[valid > 0], distance_2to1[valid > 0], flush=True)
            # print(edge_label[valid > 0], flush=True)
            edge_indices.append(np.vstack([edge_index, np.flip(edge_index, -1)]))
            pair_distances.append(np.hstack([distance_1to2, distance_2to1]))
            edge_valid.append(np.hstack([valid, valid]))
            edge_labels.append(np.hstack([edge_label, edge_label]))


        results['match_graph'] = {}
        results['match_graph']['multiview_centers'] = multiview_centers
        results['match_graph']['edge_indices'] = np.array(edge_indices).reshape((-1, 2))
        pair_distances = np.array(pair_distances).flatten()
        results['match_graph']['edge_scores'] = np.exp(-self.dist_coef * pair_distances)
        results['match_graph']['edge_valid'] = np.array(edge_valid).flatten()
        results['match_graph']['edge_labels'] = np.array(edge_labels).flatten().astype(np.float32)

        return results
