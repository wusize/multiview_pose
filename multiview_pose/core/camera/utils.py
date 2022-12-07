import numpy as np
import torch


class MonocularGeometry:
    def __init__(self, x, camera):
        self.x = x
        self.camera = camera
        self.x_camera_norm = camera.pixel_to_camera(x, left_product=True)
        self.camera_center, self.ray_direction \
            = camera.camera_to_ray(self.x_camera_norm, left_product=True)

    def get_x_camera_norm(self, mask=None):
        if mask is None:
            return self.x_camera_norm
        else:
            return self.x_camera_norm[:, mask]

    def get_ray(self, mask=None):
        if mask is None:
            return self.camera_center, self.ray_direction
        else:
            return self.camera_center, self.ray_direction[:, mask]

# from time import time
class StereoGeometry:
    def __init__(self, mono_geo_1, mono_geo_2, mask_1=None, mask_2=None):
        self.camera1 = mono_geo_1.camera
        self.camera2 = mono_geo_2.camera

        self.camera_center_1, self.ray_direction_1 = mono_geo_1.get_ray(mask_1)
        self.camera_center_2, self.ray_direction_2 = mono_geo_2.get_ray(mask_2)
        self.x_camera_norm_1 = mono_geo_1.get_x_camera_norm(mask_1)
        self.x_camera_norm_2 = mono_geo_2.get_x_camera_norm(mask_2)
        # tik = time()
        self._epipolar_distances()
        # print(f'_epipolar_distances: {time() - tik}', flush=True)
        # tik = time()
        self._reconstruct_candidates()
        # print(f'_reconstruct_candidates: {time() - tik}', flush=True)

    def _epipolar_distances(self):
        # project x1 -> image2
        epipole_1to2 = self.camera2.param['R_c2w'] @ (
                self.camera_center_1 - self.camera_center_2)
        epipole_1to2 = epipole_1to2[:2, :] / epipole_1to2[2, :]  # 2x1
        second_point_1to2 = self.camera2.param['R_c2w'] @ (
                self.camera_center_1 + self.ray_direction_1 - self.camera_center_2)  # 3xN
        second_point_1to2 = second_point_1to2[:2, :] / second_point_1to2[2:, :]  # 2xN
        self.distance_1to2 = calculate_point2line_distance(
            self.x_camera_norm_2[:2], epipole_1to2, second_point_1to2)  # N

        # project x2 -> image1
        epipole_2to1 = self.camera1.param['R_c2w'] @ (
                self.camera_center_2 - self.camera_center_1)
        epipole_2to1 = epipole_2to1[:2, :] / epipole_2to1[2, :]
        second_point_2to1 = self.camera1.param['R_c2w'] @ (
                self.camera_center_2 + self.ray_direction_2 - self.camera_center_1)
        second_point_2to1 = second_point_2to1[:2, :] / second_point_2to1[2:, :]
        self.distance_2to1 = calculate_point2line_distance(
            self.x_camera_norm_1[:2], epipole_2to1, second_point_2to1)  # N

    def _reconstruct_candidates(self):
        self.reconstructions = reconstruct_from_two_points(
            self.camera_center_1, self.ray_direction_1,
            self.camera_center_2, self.ray_direction_2)


def calculate_stereo_geometry(x1, camera1, x2, camera2, reconstruct=False):
    """
    Args
        x1: [N, 2] points in image_1
        x2: [N, 2] points in image_2
    """
    x_camera_norm_1 = camera1.pixel_to_camera(x1, left_product=True)
    camera_center_1, ray_direction_1 = camera1.camera_to_ray(x_camera_norm_1, left_product=True)
    
    x_camera_norm_2 = camera2.pixel_to_camera(x2, left_product=True)
    camera_center_2, ray_direction_2 = camera2.camera_to_ray(x_camera_norm_2, left_product=True)

    # project x1 -> image2
    epipole_1to2 = camera2.param['R_c2w'] @ (camera_center_1 - camera_center_2)
    epipole_1to2 = epipole_1to2[:2, :] / epipole_1to2[2, :]   # 2x1
    second_point_1to2 = camera2.param['R_c2w'] @ (camera_center_1 + ray_direction_1 - camera_center_2)     # 3xN
    second_point_1to2 = second_point_1to2[:2, :] / second_point_1to2[2:, :]      # 2xN
    distance_1to2 = calculate_point2line_distance(x_camera_norm_2[:2], epipole_1to2, second_point_1to2)   # N

    # project x2 -> image1
    epipole_2to1 = camera1.param['R_c2w'] @ (camera_center_2 - camera_center_1)
    epipole_2to1 = epipole_2to1[:2, :] / epipole_2to1[2, :]
    second_point_2to1 = camera1.param['R_c2w'] @ (camera_center_2 + ray_direction_2 - camera_center_1)
    second_point_2to1 = second_point_2to1[:2, :] / second_point_2to1[2:, :]
    distance_2to1 = calculate_point2line_distance(x_camera_norm_1[:2], epipole_2to1, second_point_2to1)   # N

    if reconstruct:
        reconstructed_candidates = reconstruct_from_two_points(
            camera_center_1, ray_direction_1, camera_center_2, ray_direction_2)
        return distance_1to2, distance_2to1, reconstructed_candidates
    else:
        return distance_1to2, distance_2to1


#   TODO numpy and pytorch
def reconstruct_from_two_points_(camera_center_1, ray_direction_1,
                                camera_center_2, ray_direction_2):
    assert ray_direction_1.shape[0] == 3
    # tik = time()
    ray_direction_s = torch.stack([ray_direction_1, ray_direction_2], dim=-1)  # 3xNx2
    ray_direction_s_x = crossprod2matrix(ray_direction_s.view(3, -1)).view(-1, 2, 3, 3)  # Nx2x3x3
    camera_center_s = torch.stack([camera_center_1, camera_center_2], dim=0)[None]  # 1x2x3x1

    b = (ray_direction_s_x @ camera_center_s).view(-1, 6, 1)
    A = ray_direction_s_x.view(-1, 6, 3)
    # print(f'construct equations: {time() - tik}', flush=True)
    #
    # tik = time()
    reconstructions = solve_eqs(A, b)   # Nx3
    # print(f'solve equations: {time() - tik}', flush=True)

    return reconstructions


def reconstruct_from_two_points(camera_center_1, ray_direction_1,
                                camera_center_2, ray_direction_2):
    assert ray_direction_1.shape[0] == 3

    O_m = (camera_center_1 - camera_center_2).view(-1, 3, 1)
    O_p = (camera_center_1 + camera_center_2).view(-1, 3, 1)

    D_m = torch.stack([ray_direction_1.T, - ray_direction_2.T], dim=-1)
    D_m_T = D_m.transpose(1, 2)
    D_p = torch.stack([ray_direction_1.T, ray_direction_2.T], dim=-1)

    reconstructions = 0.5 * O_p - 0.5 * D_p @ (D_m_T @ D_m).inverse() @ D_m_T @ O_m

    return reconstructions.contiguous().view(-1, 3)


#   TODO numpy and pytorch
def solve_eqs(A, b=None):
    # A: Nxmxn
    N, m, n = A.shape
    U, s, V = torch.svd(A, some=False)
    Vt = V.transpose(-2, -1)
    if b is None:
        # Ax=0
        assert m > n
        # Nxn
        return Vt[:, -1]
    else:
        # Ax=b
        # b: Nxmx1
        assert m >= n
        s_inv = torch.where(s == 0, torch.zeros_like(s), 1 / s)
        # Nxnxm
        sigma_inv = A.new_zeros(N, n, m)
        sigma_inv[:, [i for i in range(n)], [i for i in range(n)]] = s_inv

        # Nxnxn
        # V = Vt.transpose(2, 1)
        # Nxmxm
        Ut = U.transpose(2, 1)
        # Nxnx1
        x = V @ sigma_inv @ (Ut @ b)

        # Nxn
        return x[:, :, 0]


#   TODO numpy and pytorch
def crossprod2matrix(k):
    # 3XN
    K = k.new_zeros((k.shape[1], 3, 3))
    K[:, 0, 1] = -k[2]
    K[:, 0, 2] = k[1]
    K[:, 1, 0] = k[2]
    K[:, 1, 2] = -k[0]
    K[:, 2, 0] = -k[1]
    K[:, 2, 1] = k[0]

    return K


def calculate_point2line_distance(point, line_point_1, line_point_2):
    ray_direction_1, ray_direction_2 \
        = line_point_1 - point, line_point_2 - point     # 2xN, 2XN
    if isinstance(point, np.ndarray):
        s = abs(np.cross(ray_direction_1.T, ray_direction_2.T))
        h = s / np.linalg.norm(line_point_2 - line_point_1, axis=0)   # N
    else:
        ray_direction_1_ = ray_direction_1.new_zeros(3, ray_direction_1.shape[1])
        ray_direction_2_ = ray_direction_2.new_zeros(3, ray_direction_2.shape[1])

        ray_direction_1_[:2] = ray_direction_1
        ray_direction_2_[:2] = ray_direction_2

        s = torch.cross(ray_direction_1_.T, ray_direction_2_.T, dim=-1).norm(dim=-1)
        h = s / (line_point_2 - line_point_1).norm(dim=0)  # N

    return h


def distort_points(points, K, dist, new_K=None):
    r"""Distortion of a set of 2D points based on the lens distortion model.
    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.
    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.
    Example:
        >>> points = torch.rand(1, 1, 2)
        >>> K = torch.eye(3)[None]
        >>> dist_coeff = torch.rand(1, 4)
        >>> points_dist = distort_points(points, K, dist_coeff)
    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f'points shape is invalid. Got {points.shape}.')

    if K.shape[-2:] != (3, 3):
        raise ValueError(f'K matrix shape is invalid. Got {K.shape}.')

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f'new_K matrix shape is invalid. Got {new_K.shape}.')

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f'Invalid number of distortion coefficients. Got {dist.shape[-1]}')

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    new_cx = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy = new_K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x = (points[..., 0] - new_cx) / new_fx  # (BxN - Bx1)/Bx1 -> BxN or (N,)
    y = (points[..., 1] - new_cy) / new_fy  # (BxN - Bx1)/Bx1 -> BxN or (N,)

    # Distort points
    r2 = x * x + y * y

    rad_poly = (1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3) / (
        1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3
    )
    xd = (
        x * rad_poly
        + 2 * dist[..., 2:3] * x * y
        + dist[..., 3:4] * (r2 + 2 * x * x)
        + dist[..., 8:9] * r2
        + dist[..., 9:10] * r2 * r2
    )
    yd = (
        y * rad_poly
        + dist[..., 2:3] * (r2 + 2 * y * y)
        + 2 * dist[..., 3:4] * x * y
        + dist[..., 10:11] * r2
        + dist[..., 11:12] * r2 * r2
    )

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        tilt = tilt_projection(dist[..., 12], dist[..., 13])

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        points_untilt = torch.stack([xd, yd, torch.ones_like(xd)], -1) @ tilt.transpose(-2, -1)
        xd = points_untilt[..., 0] / points_untilt[..., 2]
        yd = points_untilt[..., 1] / points_untilt[..., 2]

    # Convert points from normalized camera coordinates to pixel coordinates
    cx = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx = K[..., 0:1, 0]  # focal in x (Bx1)
    fy = K[..., 1:2, 1]  # focal in y (Bx1)

    x = fx * xd + cx
    y = fy * yd + cy

    return torch.stack([x, y], -1)


def tilt_projection(taux, tauy, return_inverse=False):
    r"""Estimate the tilt projection matrix or the inverse tilt projection matrix.
    Args:
        taux: Rotation angle in radians around the :math:`x`-axis with shape :math:`(*, 1)`.
        tauy: Rotation angle in radians around the :math:`y`-axis with shape :math:`(*, 1)`.
        return_inverse: False to obtain the the tilt projection matrix. True for the inverse matrix.
    Returns:
        torch.Tensor: Inverse tilt projection matrix with shape :math:`(*, 3, 3)`.
    """
    if taux.shape != tauy.shape:
        raise ValueError(f'Shape of taux {taux.shape} and tauy {tauy.shape} do not match.')

    ndim: int = taux.dim()
    taux = taux.reshape(-1)
    tauy = tauy.reshape(-1)

    cTx = torch.cos(taux)
    sTx = torch.sin(taux)
    cTy = torch.cos(tauy)
    sTy = torch.sin(tauy)
    zero = torch.zeros_like(cTx)
    one = torch.ones_like(cTx)

    Rx = torch.stack([one, zero, zero, zero, cTx, sTx, zero, -sTx, cTx], -1).reshape(-1, 3, 3)
    Ry = torch.stack([cTy, zero, -sTy, zero, one, zero, sTy, zero, cTy], -1).reshape(-1, 3, 3)
    R = Ry @ Rx

    if return_inverse:
        invR22 = 1 / R[..., 2, 2]
        invPz = torch.stack(
            [invR22, zero, R[..., 0, 2] * invR22, zero, invR22, R[..., 1, 2] * invR22, zero, zero, one], -1
        ).reshape(-1, 3, 3)

        inv_tilt = R.transpose(-1, -2) @ invPz
        if ndim == 0:
            inv_tilt = torch.squeeze(inv_tilt)

        return inv_tilt

    Pz = torch.stack(
        [R[..., 2, 2], zero, -R[..., 0, 2], zero, R[..., 2, 2], -R[..., 1, 2], zero, zero, one], -1
    ).reshape(-1, 3, 3)

    tilt = Pz @ R.transpose(-1, -2)
    if ndim == 0:
        tilt = torch.squeeze(tilt)

    return tilt


def convert_points_to_homogeneous(points):
    r"""Function that converts points from Euclidean to homogeneous space.
    Args:
        points: the points to be transformed with shape :math:`(B, N, D)`.
    Returns:
        the points in homogeneous coordinates :math:`(B, N, D+1)`.
    Examples:
        >>> input = torch.tensor([[0., 0.]])
        >>> convert_points_to_homogeneous(input)
        tensor([[0., 0., 1.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")
    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    return torch.nn.functional.pad(points, [0, 1], "constant", 1.0)


def convert_points_from_homogeneous(points, eps: float = 1e-8):
    r"""Function that converts points from homogeneous to Euclidean space.
    Args:
        points: the points to be transformed of shape :math:`(B, N, D)`.
        eps: to avoid division by zero.
    Returns:
        the points in Euclidean space :math:`(B, N, D-1)`.
    Examples:
        >>> input = torch.tensor([[0., 0., 1.]])
        >>> convert_points_from_homogeneous(input)
        tensor([[0., 0.]])
    """
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(points)}")

    if len(points.shape) < 2:
        raise ValueError(f"Input must be at least a 2D tensor. Got {points.shape}")

    # we check for points at max_val
    z_vec = points[..., -1:]

    # set the results of division by zeror/near-zero to 1.0
    # follow the convention of opencv:
    # https://github.com/opencv/opencv/pull/14411/files
    mask = torch.abs(z_vec) > eps
    scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

    return scale * points[..., :-1]


def transform_points(trans_01, points_1):
    r"""Function that applies transformations to a set of points.
    Args:
        trans_01 (torch.Tensor): tensor for transformations of shape
          :math:`(B, D+1, D+1)`.
        points_1 (torch.Tensor): tensor of points of shape :math:`(B, N, D)`.
    Returns:
        torch.Tensor: tensor of N-dimensional points.
    Shape:
        - Output: :math:`(B, N, D)`
    Examples:
        >>> points_1 = torch.rand(2, 4, 3)  # BxNx3
        >>> trans_01 = torch.eye(4).view(1, 4, 4)  # Bx4x4
        >>> points_0 = transform_points(trans_01, points_1)  # BxNx3
    """

    if not trans_01.shape[0] == points_1.shape[0] and trans_01.shape[0] != 1:
        raise ValueError(
            "Input batch size must be the same for both tensors or 1."
            f"Got {trans_01.shape} and {points_1.shape}"
        )
    if not trans_01.shape[-1] == (points_1.shape[-1] + 1):
        raise ValueError(
            "Last input dimensions must differ by one unit"
            f"Got{trans_01} and {points_1}"
        )

    # We reshape to BxNxD in case we get more dimensions, e.g., MxBxNxD
    shape_inp = list(points_1.shape)
    points_1 = points_1.reshape(-1, points_1.shape[-2], points_1.shape[-1])
    trans_01 = trans_01.reshape(-1, trans_01.shape[-2], trans_01.shape[-1])
    # We expand trans_01 to match the dimensions needed for bmm
    trans_01 = torch.repeat_interleave(trans_01, repeats=points_1.shape[0] // trans_01.shape[0], dim=0)
    # to homogeneous
    points_1_h = convert_points_to_homogeneous(points_1)  # BxNxD+1
    # transform coordinates
    points_0_h = torch.bmm(points_1_h, trans_01.permute(0, 2, 1))
    points_0_h = torch.squeeze(points_0_h, dim=-1)
    # to euclidean
    points_0 = convert_points_from_homogeneous(points_0_h)  # BxNxD
    # reshape to the input shape
    shape_inp[-2] = points_0.shape[-2]
    shape_inp[-1] = points_0.shape[-1]
    points_0 = points_0.reshape(shape_inp)
    return points_0


def undistortPoints_torch(points, K, dist, new_K=None, num_iters=5):
    """
    Adapted from: "https://github.com/kornia/
        kornia/blob/master/kornia/geometry/
        calibration/undistort.py"
    Returns:

    """
    r"""Compensate for lens distortion a set of 2D image points.
    Radial :math:`(k_1, k_2, k_3, k_4, k_4, k_6)`,
    tangential :math:`(p_1, p_2)`, thin prism :math:`(s_1, s_2, s_3, s_4)`, and tilt :math:`(\tau_x, \tau_y)`
    distortion models are considered in this function.
    Args:
        points: Input image points with shape :math:`(*, N, 2)`.
        K: Intrinsic camera matrix with shape :math:`(*, 3, 3)`.
        dist: Distortion coefficients
            :math:`(k_1,k_2,p_1,p_2[,k_3[,k_4,k_5,k_6[,s_1,s_2,s_3,s_4[,\tau_x,\tau_y]]]])`. This is
            a vector with 4, 5, 8, 12 or 14 elements with shape :math:`(*, n)`.
        new_K: Intrinsic camera matrix of the distorted image. By default, it is the same as K but you may additionally
            scale and shift the result by using a different matrix. Shape: :math:`(*, 3, 3)`. Default: None.
        num_iters: Number of undistortion iterations. Default: 5.
    Returns:
        Undistorted 2D points with shape :math:`(*, N, 2)`.
    Example:
        >>> _ = torch.manual_seed(0)
        >>> x = torch.rand(1, 4, 2)
        >>> K = torch.eye(3)[None]
        >>> dist = torch.rand(1, 4)
        >>> undistort_points(x, K, dist)
        tensor([[[-0.1513, -0.1165],
                 [ 0.0711,  0.1100],
                 [-0.0697,  0.0228],
                 [-0.1843, -0.1606]]])
    """
    if points.dim() < 2 and points.shape[-1] != 2:
        raise ValueError(f'points shape is invalid. Got {points.shape}.')

    if K.shape[-2:] != (3, 3):
        raise ValueError(f'K matrix shape is invalid. Got {K.shape}.')

    if new_K is None:
        new_K = K
    elif new_K.shape[-2:] != (3, 3):
        raise ValueError(f'new_K matrix shape is invalid. Got {new_K.shape}.')

    if dist.shape[-1] not in [4, 5, 8, 12, 14]:
        raise ValueError(f"Invalid number of distortion coefficients. Got {dist.shape[-1]}")

    # Adding zeros to obtain vector with 14 coeffs.
    if dist.shape[-1] < 14:
        dist = torch.nn.functional.pad(dist, [0, 14 - dist.shape[-1]])

    # Convert 2D points from pixels to normalized camera coordinates
    cx = K[..., 0:1, 2]  # princial point in x (Bx1)
    cy = K[..., 1:2, 2]  # princial point in y (Bx1)
    fx = K[..., 0:1, 0]  # focal in x (Bx1)
    fy = K[..., 1:2, 1]  # focal in y (Bx1)

    # This is equivalent to K^-1 [u,v,1]^T
    x = (points[..., 0] - cx) / fx  # (BxN - Bx1)/Bx1 -> BxN
    y = (points[..., 1] - cy) / fy  # (BxN - Bx1)/Bx1 -> BxN

    # Compensate for tilt distortion
    if torch.any(dist[..., 12] != 0) or torch.any(dist[..., 13] != 0):
        inv_tilt = tilt_projection(dist[..., 12], dist[..., 13], True)

        # Transposed untilt points (instead of [x,y,1]^T, we obtain [x,y,1])
        x, y = transform_points(inv_tilt, torch.stack([x, y], dim=-1)).unbind(-1)

    # Iteratively undistort points
    x0, y0 = x, y
    for _ in range(num_iters):
        r2 = x * x + y * y

        inv_rad_poly = (1 + dist[..., 5:6] * r2 + dist[..., 6:7] * r2 * r2 + dist[..., 7:8] * r2 ** 3) / (
                1 + dist[..., 0:1] * r2 + dist[..., 1:2] * r2 * r2 + dist[..., 4:5] * r2 ** 3
        )
        deltaX = (
                2 * dist[..., 2:3] * x * y
                + dist[..., 3:4] * (r2 + 2 * x * x)
                + dist[..., 8:9] * r2
                + dist[..., 9:10] * r2 * r2
        )
        deltaY = (
                dist[..., 2:3] * (r2 + 2 * y * y)
                + 2 * dist[..., 3:4] * x * y
                + dist[..., 10:11] * r2
                + dist[..., 11:12] * r2 * r2
        )

        x = (x0 - deltaX) * inv_rad_poly
        y = (y0 - deltaY) * inv_rad_poly

    # Convert points from normalized camera coordinates to pixel coordinates
    new_cx = new_K[..., 0:1, 2]  # princial point in x (Bx1)
    new_cy = new_K[..., 1:2, 2]  # princial point in y (Bx1)
    new_fx = new_K[..., 0:1, 0]  # focal in x (Bx1)
    new_fy = new_K[..., 1:2, 1]  # focal in y (Bx1)
    x = new_fx * x + new_cx
    y = new_fy * y + new_cy
    return torch.stack([x, y], -1)
