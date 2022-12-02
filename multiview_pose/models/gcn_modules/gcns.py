import torch
import torch.nn as nn
from .builder import GCNS
from torch_geometric.nn import MessagePassing
from mmcv.cnn.bricks import build_norm_layer


def custom_build_norm_layer(norm_type, num_features):
    if norm_type is None:
        return nn.Identity()
    else:
        return build_norm_layer(cfg=dict(type=norm_type),
                                num_features=num_features)[1]


@GCNS.register_module()
class EdgeConv(MessagePassing):
    def __init__(self, node_mlp, edge_mlp, aggregate='max',
                 node_edge_merge='cat', residual=True):
        super(EdgeConv, self).__init__(aggr=aggregate)  # "Max" aggregation.
        self.node_mlp = node_mlp
        self.edge_mlp = edge_mlp
        self.residual = residual
        self.node_edge_merge = node_edge_merge

    def forward(self, x, edge_index, edge_feature=None):
        """

        Args:
            x:
            edge_index: [N, in_channels]
            edge_feature: [2, E]

        Returns:

        """
        # x has shape
        # edge_index has shape [2, E]
        x_updated = self.propagate(edge_index, x=x, edge_feature=edge_feature)

        if self.edge_mlp is None:
            if edge_feature is None:
                edge_feature_updated = None
            else:
                edge_feature_updated = torch.zeros_like(edge_feature) \
                    if self.residual else edge_feature
        else:
            # assert edge_feature is not None
            edge_feature_updated = self.update_edge_feature(edge_index, x=x, edge_feature=edge_feature)

        return x_updated, edge_feature_updated

    def message(self, x_i, x_j, edge_feature):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]
        # edge_feature has shape [E, e_channels]
        if edge_feature is None:
            assert self.node_edge_merge == 'add'
            tmp = torch.cat([x_i, x_j - x_i], dim=1)
        else:
            if self.node_edge_merge == 'cat':
                tmp = torch.cat([x_i, x_j - x_i, edge_feature], dim=1)
            elif self.node_edge_merge == 'add':
                tmp = torch.cat([x_i, x_j - x_i + edge_feature], dim=1)
            else:
                raise ValueError(f'{self.node_edge_merge} not supported')

        return self.node_mlp(tmp)

    def update_edge_feature(self, edge_index, x, edge_feature):
        # assert edge_feature is not None
        src, tar = edge_index
        x_j, x_i = x[src], x[tar]

        if edge_feature is None:
            assert self.node_edge_merge == 'add'
            tmp = x_j - x_i
        else:
            if self.node_edge_merge == 'cat':
                tmp = torch.cat([x_j - x_i, edge_feature], dim=1)
            elif self.node_edge_merge == 'add':
                tmp = x_j - x_i + edge_feature
            else:
                raise ValueError(f'{self.node_edge_merge} not supported')

        return self.edge_mlp(tmp)


@GCNS.register_module()
class EdgeConvLayers(nn.Module):
    def __init__(self,
                 node_channels,
                 edge_channels,
                 mid_channels,
                 node_out_channels,
                 edge_out_channels=None,
                 node_edge_merge='cat',
                 node_layers=[2, 2],
                 edge_layers=[0, 0], residual=True,
                 norm_type='LN'):
        super(EdgeConvLayers, self).__init__()

        self.residual = residual
        self.node_feature_mapping = nn.Identity() \
            if node_channels == mid_channels \
            else nn.Linear(node_channels, mid_channels)
        if edge_channels is None:
            self.edge_feature_mapping = None
        else:
            self.edge_feature_mapping = nn.Identity() \
                if edge_channels == mid_channels \
                else nn.Linear(edge_channels, mid_channels)
        self.num_convs = len(node_layers)
        assert self.num_convs == len(edge_layers)

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.mid_channels = mid_channels
        self.node_out_channels = node_out_channels
        self.edge_out_channels = edge_out_channels
        self.node_edge_merge = node_edge_merge
        self.norm_type = norm_type
        if self.node_edge_merge == 'cat':
            in_edge_channels = 2 * mid_channels
        elif self.node_edge_merge == 'add':
            in_edge_channels = mid_channels
        else:
            raise ValueError
        for i in range(self.num_convs):
            conv_name = f'conv_{i}'
            assert node_layers[i] > 0
            node_mlp = nn.Sequential(*[nn.Sequential(nn.Linear(in_edge_channels + mid_channels, mid_channels),
                                                     custom_build_norm_layer(norm_type=norm_type,
                                                                             num_features=mid_channels),
                                                     nn.ReLU())
                                       if j == 0 else nn.Sequential(nn.Linear(mid_channels, mid_channels),
                                                                    custom_build_norm_layer(norm_type=norm_type,
                                                                                            num_features=mid_channels),
                                                                    nn.ReLU())
                                       for j in range(node_layers[i])])
            edge_mlp = nn.Sequential(*[nn.Sequential(nn.Linear(in_edge_channels, mid_channels),
                                                     custom_build_norm_layer(norm_type=norm_type,
                                                                             num_features=mid_channels),
                                                     nn.ReLU())
                                       if j == 0 else nn.Sequential(nn.Linear(mid_channels, mid_channels),
                                                                    custom_build_norm_layer(norm_type=norm_type,
                                                                                            num_features=mid_channels),
                                                                    nn.ReLU())
                                       for j in range(edge_layers[i])]) \
                if edge_layers[i] > 0 else None
            self.add_module(conv_name, EdgeConv(node_mlp, edge_mlp,
                                                aggregate='max', node_edge_merge=node_edge_merge,
                                                residual=residual))

        self.edge_out_layer = None if edge_out_channels is None \
            else nn.Sequential(nn.Linear(in_edge_channels, mid_channels),
                               custom_build_norm_layer(norm_type=norm_type,
                                                       num_features=mid_channels),
                               nn.ReLU(),
                               nn.Linear(mid_channels, edge_out_channels))
        self.node_out_layer = nn.Identity() if node_out_channels is None \
            else nn.Sequential(nn.Linear(mid_channels, mid_channels),
                               custom_build_norm_layer(norm_type=norm_type,
                                                       num_features=mid_channels),
                               nn.ReLU(),
                               nn.Linear(mid_channels, node_out_channels))

    def forward(self, node_features, edge_indices, edge_features=None):
        """

        Args:
            node_features: PxC
            edge_features: PxC'
            edge_indices: Ex2

        Returns:

        """
        if edge_indices.shape[0] != 2:
            edge_indices = edge_indices.T
        node_features = self.node_feature_mapping(node_features)
        if self.edge_feature_mapping is not None:
            edge_features = self.edge_feature_mapping(edge_features)

        for i in range(self.num_convs):
            conv = getattr(self, f'conv_{i}')
            x, y = conv(node_features, edge_indices,
                        edge_feature=edge_features)
            node_features = node_features + x if self.residual else x
            if edge_features is not None:
                edge_features = edge_features + y if self.residual else y
            elif y is not None:
                edge_features = y

        node_output = self.node_out_layer(node_features)
        if self.edge_out_layer is None:
            edge_out = None
        else:
            src, tar = edge_indices
            x_j, x_i = node_features[src], node_features[tar]

            if self.node_edge_merge == 'cat':
                if edge_features is None:
                    edge_features = x_j - x_i
                tmp = torch.cat([x_j - x_i, edge_features], dim=1)
            else:
                if edge_features is None:
                    edge_features = torch.zeros_like(x_i)
                tmp = x_j - x_i + edge_features
            edge_out = self.edge_out_layer(tmp)

        return node_output, edge_out
