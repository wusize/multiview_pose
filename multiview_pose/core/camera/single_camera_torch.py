# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmpose.core.camera import CAMERAS
from mmpose.core.camera import SimpleCameraTorch
from .utils import undistortPoints_torch


@CAMERAS.register_module()
class CustomSimpleCameraTorch(SimpleCameraTorch):
    def __init__(self, param, device):
        super(CustomSimpleCameraTorch, self).__init__(param, device)
        self.device = device

    def pixel_to_camera(self, X, left_product=False):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2 and X.shape[-1] == 2
        X = X.float()
        camera_matrix = torch.eye(3, device=self.device)
        camera_matrix[:2] = self.param['K'].T
        _X = X.new_ones(X.shape[0], 3)
        _X[:, :2] = self.undistort(X)

        _X = torch.inverse(camera_matrix) @ _X.T

        if left_product:
            return _X
        else:
            return _X.T

    def undistort(self, X):
        if self.undistortion:
            camera_matrix = torch.eye(3, device=self.device)
            camera_matrix[:2] = self.param['K'].T
            uv = X.view(1, -1, 2)
            k = self.param['k']
            p = self.param['p']
            dist = X.new_zeros(1, 5)
            # dist = np.array([[k[0], k[1], p[0], p[1], k[2]]], dtype=np.float32)
            dist[0, :2] = k[:2]
            dist[0, 2:4] = p[:2]
            dist[0, 4] = k[2]
            uv = undistortPoints_torch(uv,
                                       camera_matrix[None],
                                       dist, camera_matrix[None])[0]
            return uv
        else:
            return X

    def camera_to_ray(self, X, left_product=False):
        assert isinstance(X, torch.Tensor)
        assert X.ndim >= 2
        assert X.shape[0] == 3 or X.shape[-1] == 3
        if X.shape[0] == 3:
            X = X.T

        _X = X / (X[:, 2:] + 1e-12)
        camera_center = self.param['T_c2w']    # 1x3
        ray_direction = _X @ self.param['R_c2w']   # Nx3

        if left_product:
            return camera_center.T, ray_direction.T
        else:
            return camera_center, ray_direction
