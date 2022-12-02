# Copyright (c) OpenMMLab. All rights reserved.
from mmpose.core.camera import *  # noqa
from .single_camera import CustomSimpleCamera
from .single_camera_torch import CustomSimpleCameraTorch

__all__ = ['CustomSimpleCamera', 'CustomSimpleCameraTorch']
