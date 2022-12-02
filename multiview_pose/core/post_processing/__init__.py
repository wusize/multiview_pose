# Copyright (c) OpenMMLab. All rights reserved.
from .post_transforms import transform_preds_torch
from mmpose.core.post_processing import *   # noqa

__all__ = [
     'transform_preds_torch',
]
