from mmpose.models.losses import HeatmapLoss
from mmpose.models import LOSSES


@LOSSES.register_module()
class CustomHeatmapLoss(HeatmapLoss):
    def __init__(self, loss_weight=1.0, *args, **kwargs):
        super(CustomHeatmapLoss, self).__init__(*args, **kwargs)
        self.loss_weight = loss_weight

    def forward(self, *args, **kwargs):
        return self.loss_weight * super(CustomHeatmapLoss, self).forward(*args, **kwargs)
