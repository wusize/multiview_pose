# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn.functional as F
from mmpose.models.heads import DeconvHead
from mmpose.models.builder import HEADS


@HEADS.register_module()
class CustomDeconvHead(DeconvHead):
    def __init__(self, feature_extract_layers, *args, **kwargs):
        super(CustomDeconvHead, self).__init__(*args, **kwargs)
        self.feature_extract_layers = feature_extract_layers

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)

        if len(self.feature_extract_layers) == 0:
            x = self.deconv_layers(x)
            y = self.final_layer(x)

            return [y]
        else:
            features = []
            for i, layer in enumerate(self.deconv_layers):
                x = layer(x)
                if i in self.feature_extract_layers:
                    features.append(x)
            y = self.final_layer(x)
            target_shape = y.shape[2:4]
            reshaped_features = [F.interpolate(f, size=target_shape,
                                               mode="bilinear")
                                 for f in features]
            aggregated_features = torch.cat(reshaped_features, dim=1)

            return [torch.cat([aggregated_features, y], dim=1)]
