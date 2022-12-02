# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp

from collections import OrderedDict
import tempfile
import mmcv
import numpy as np

from mmpose.datasets.builder import DATASETS
from mmpose.datasets.datasets.body3d import Body3DMviewDirectPanopticDataset


@DATASETS.register_module()
class CustomPanopticDataset(Body3DMviewDirectPanopticDataset):

    def evaluate(self, outputs, res_folder=None, metric='mpjpe', **kwargs):
        """

        Args:
            outputs list(dict(pose_3d, sample_id)):
                pose_3d (np.ndarray): predicted 3D human pose
                sample_id (np.ndarray): sample id of a frame.
            res_folder (str): Path of directory to save the results.
            metric (str | list[str]): Metric to be performed.
                Defaults: 'mpjpe'.
            **kwargs:

        Returns:

        """
        pose_3ds = np.concatenate([output['pose_3d'] for output in outputs],
                                  axis=0)
        center_3ds = np.concatenate([output['human_detection_3d'][..., None, :]
                                     for output in outputs],
                                    axis=0)

        sample_ids = []
        for output in outputs:
            sample_ids.extend(output['sample_id'])
        _outputs = [
            dict(sample_id=sample_id, pose_3d=pose_3d, center_3d=center_3d)
            for (sample_id, pose_3d, center_3d) in zip(sample_ids, pose_3ds, center_3ds)
        ]
        _outputs = self._sort_and_unique_outputs(_outputs, key='sample_id')

        metrics = metric if isinstance(metric, list) else [metric]
        for _metric in metrics:
            if _metric not in self.ALLOWED_METRICS:
                raise ValueError(
                    f'Unsupported metric "{_metric}"'
                    f'Supported metrics are {self.ALLOWED_METRICS}')

        if res_folder is not None:
            tmp_folder = None
            res_file = osp.join(res_folder, 'result_keypoints.json')
        else:
            tmp_folder = tempfile.TemporaryDirectory()
            res_file = osp.join(tmp_folder.name, 'result_keypoints.json')

        mmcv.dump(_outputs, res_file)

        results = dict()
        results.update(self._evaluate(_outputs, metrics))

        results.update(self._evaluate(_outputs, metrics,
                                      eval_name='center_3d',
                                      suffix='_c',
                                      joint_ids=[self.root_id]))
        if tmp_folder is not None:
            tmp_folder.cleanup()

        return results

    def _evaluate(self, _outputs, metrics, eval_name='pose_3d', suffix='', joint_ids=None):
        eval_list = []
        gt_num = self.db_size // self.num_cameras
        assert len(
            _outputs) == gt_num, f'number mismatch: {len(_outputs)}, {gt_num}'

        total_gt = 0
        for i in range(gt_num):
            index = self.num_cameras * i
            db_rec = copy.deepcopy(self.db[index])
            joints_3d = db_rec['joints_3d']
            joints_3d_vis = db_rec['joints_3d_visible']

            if joints_3d_vis.sum() < 1:
                continue

            pred = _outputs[i][eval_name].copy()
            pred = pred[pred[:, 0, 3] >= 0]
            for pose in pred:
                mpjpes = []
                for (gt, gt_vis) in zip(joints_3d, joints_3d_vis):
                    if joint_ids is not None:
                        gt = gt[joint_ids]
                        gt_vis = gt_vis[joint_ids]
                    vis = gt_vis[:, 0] > 0
                    if vis.sum() < 1:
                        break
                    mpjpe = np.mean(
                        np.sqrt(
                            np.sum((pose[vis, 0:3] - gt[vis])**2, axis=-1)))
                    mpjpes.append(mpjpe)
                min_gt = np.argmin(mpjpes)
                min_mpjpe = np.min(mpjpes)
                score = pose[0, 4]
                eval_list.append({
                    'mpjpe': float(min_mpjpe),
                    'score': float(score),
                    'gt_id': int(total_gt + min_gt)
                })

            total_gt += (joints_3d_vis[:, :, 0].sum(-1) >= 1).sum()

        mpjpe_threshold = np.arange(25, 155, 25)
        aps = []
        ars = []
        for t in mpjpe_threshold:
            ap, ar = self._eval_list_to_ap(eval_list, total_gt, t)
            aps.append(ap)
            ars.append(ar)

        name_value_tuples = []
        for _metric in metrics:
            if _metric == 'mpjpe':
                stats_names = ['RECALL 500mm', 'MPJPE 500mm']
                for i, stats_name in enumerate(stats_names):
                    stats_names[i] = stats_name + suffix
                info_str = list(
                    zip(stats_names, [
                        self._eval_list_to_recall(eval_list, total_gt),
                        self._eval_list_to_mpjpe(eval_list)
                    ]))
            elif _metric == 'mAP':
                stats_names = [
                    'AP 25', 'AP 50', 'AP 75', 'AP 100', 'AP 125', 'AP 150',
                    'mAP', 'AR 25', 'AR 50', 'AR 75', 'AR 100', 'AR 125',
                    'AR 150', 'mAR'
                ]
                for i, stats_name in enumerate(stats_names):
                    stats_names[i] = stats_name + suffix
                mAP = np.array(aps).mean()
                mAR = np.array(ars).mean()
                info_str = list(zip(stats_names, aps + [mAP] + ars + [mAR]))
            else:
                raise NotImplementedError
            name_value_tuples.extend(info_str)

        return OrderedDict(name_value_tuples)

    @staticmethod
    def _sort_and_unique_outputs(outputs, key='sample_id'):
        """sort outputs and remove the repeated ones."""
        outputs = sorted(outputs, key=lambda x: x[key])
        num_outputs = len(outputs)
        for i in range(num_outputs - 1, 0, -1):
            if outputs[i][key] == outputs[i - 1][key]:
                del outputs[i]

        return outputs
