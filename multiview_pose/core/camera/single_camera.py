# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import cv2
from mmpose.core.camera import SimpleCamera, CAMERAS


@CAMERAS.register_module()
class CustomSimpleCamera(SimpleCamera):
    def undistort(self, X):
        if self.undistortion:
            camera_matrix = np.eye(3, dtype=np.float32)
            camera_matrix[:2] = self.param['K'].T
            uv = X.reshape(-1, 1, 2)
            k = self.param['k']
            p = self.param['p']
            dist = np.array([[k[0], k[1], p[0], p[1], k[2]]], dtype=np.float32)
            uv = cv2.undistortPoints(uv, camera_matrix, dist,
                                     None, camera_matrix).reshape(-1, 2)  # uv: Nx1x2 -> Nx2
            return uv
        else:
            return X

    def pixel_to_camera(self, X, left_product=False):
        assert isinstance(X, np.ndarray)
        assert X.ndim >= 2 and X.shape[-1] == 2
        camera_matrix = np.eye(3, dtype=np.float32)
        camera_matrix[:2] = self.param['K'].T
        _X = np.ones((X.shape[0], 3), dtype=np.float32)
        _X[:, :2] = self.undistort(X)

        _X = np.linalg.inv(camera_matrix) @ _X.T

        if left_product:
            return _X
        else:
            return _X.T

    def camera_to_ray(self, X, left_product=False):
        assert isinstance(X, np.ndarray)
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
