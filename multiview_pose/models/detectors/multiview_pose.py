import torch
from mmpose.models.builder import POSENETS
from mmpose.models.detectors import DetectAndRegress
from multiview_pose.models.gcn_modules import GCNS


@POSENETS.register_module()
class GraphBasedModel(DetectAndRegress):
    def __init__(self, num_joints, pose_refiner, test_with_refine=True, freeze_keypoint_head=True,
                 *args, **kwargs):
        super(GraphBasedModel, self).__init__(*args, **kwargs)
        self.num_joints = num_joints
        if pose_refiner is not None:
            self.pose_refiner = GCNS.build(pose_refiner)
        else:
            self.pose_refiner = None
        self.test_with_refine = test_with_refine
        self.freeze_keypoint_head = freeze_keypoint_head

    def train(self, mode=True):
        """Sets the module in training mode.
        Args:
            mode (bool): whether to set training mode (``True``)
                or evaluation mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        super().train(mode)
        if mode and self.freeze_2d:
            if self.backbone is not None:
                self._freeze(self.backbone)
            if self.keypoint_head is not None and self.freeze_keypoint_head:
                self._freeze(self.keypoint_head)

        return self

    @property
    def has_keypoint_2d_loss(self):
        return (not self.freeze_2d) or (self.freeze_2d and not self.freeze_keypoint_head)

    def forward(self,
                img=None,
                img_metas=None,
                return_loss=True,
                target=None,
                mask=None,
                targets_3d=None,
                input_heatmaps=None,
                **kwargs):

        if return_loss:
            return self.forward_train(img, img_metas, target, mask,
                                      targets_3d, input_heatmaps, **kwargs)
        else:
            return self.forward_test(img, img_metas, input_heatmaps, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      target=None,
                      mask=None,
                      targets_3d=None,
                      input_heatmaps=None,
                      **kwargs):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        losses = dict()
        human_candidates, human_loss = self.human_detector.forward_train(
            None, img_metas, feature_maps, targets_3d, return_preds=True, **kwargs)
        losses.update(human_loss)

        pose_pred, pose_loss = self.pose_regressor.forward_train(
            None,
            img_metas,
            feature_maps=[f[:, -self.num_joints:].detach() for f in feature_maps],
            human_candidates=human_candidates,
            return_preds=True)
        losses.update(pose_loss)
        if self.pose_refiner is not None:
            losses.update(self.pose_refiner.forward_train(pose_pred, feature_maps, img_metas))

        if self.has_keypoint_2d_loss:
            losses_2d = {}
            heatmaps_tensor = torch.cat([f[:, -self.num_joints:] for f in feature_maps], dim=0)
            targets_tensor = torch.cat([t[0] for t in target], dim=0)
            masks_tensor = torch.cat([m[0] for m in mask], dim=0)
            losses_2d_ = self.keypoint_head.get_loss([heatmaps_tensor],
                                                     [targets_tensor], [masks_tensor])
            for k, v in losses_2d_.items():
                losses_2d[k + '_2d'] = v
            losses.update(losses_2d)

        return losses

    def forward_test(
        self,
        img,
        img_metas,
        input_heatmaps=None,
        **kwargs
    ):
        if self.backbone is None:
            assert input_heatmaps is not None
            feature_maps = []
            for input_heatmap in input_heatmaps:
                feature_maps.append(input_heatmap[0])
        else:
            feature_maps = []
            assert isinstance(img, list)
            for img_ in img:
                feature_maps.append(self.predict_heatmap(img_)[0])

        human_candidates = self.human_detector.forward_test(
            None, img_metas, feature_maps, **kwargs)

        human_poses = self.pose_regressor(
            None,
            img_metas,
            return_loss=False,
            feature_maps=[f[:, -self.num_joints:] for f in feature_maps],
            human_candidates=human_candidates)
        if self.pose_refiner is not None and self.test_with_refine:
            human_poses = self.pose_refiner.forward_test(human_poses, feature_maps, img_metas)

        result = {}
        result['pose_3d'] = human_poses.cpu().numpy()
        result['human_detection_3d'] = human_candidates.cpu().numpy()
        result['sample_id'] = [img_meta['sample_id'] for img_meta in img_metas]

        return result
