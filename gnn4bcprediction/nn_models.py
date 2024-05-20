import copy

import torch
from torch.nn import Linear
from torch.nn.functional import relu, sigmoid
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv


class SequentialLayersWithActivation(torch.nn.Module):
    def __init__(self, layer_type, activation_layer, input_dim, hidden_dim,
                 output_dim, num_hidden_layers):
        super(SequentialLayersWithActivation, self).__init__()

        num_hidden_layers = max(0, num_hidden_layers)

        self.layers = torch.nn.ModuleList(
            [layer_type(input_dim, hidden_dim)] + [
                layer_type(hidden_dim, hidden_dim) for _ in
                range(num_hidden_layers)] + [
                layer_type(hidden_dim, output_dim)])

        self.activation = activation_layer

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index=None):
        h = copy.deepcopy(x)

        for i, layer in enumerate(self.layers):
            if edge_index is None:
                h = layer(h)
            else:
                h = layer(h, edge_index)

            if i != len(self.layers) - 1:
                h = self.activation(h)

        h = sigmoid(h) * 0.5

        return h


class GCN(SequentialLayersWithActivation):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(GCN, self).__init__(GCNConv, relu, input_dim, hidden_dim,
                                  output_dim, num_hidden_layers)


class GATv2(SequentialLayersWithActivation):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(GATv2, self).__init__(GATv2Conv, relu, input_dim, hidden_dim,
                                    output_dim, num_hidden_layers)


class GraphSAGE(SequentialLayersWithActivation):
    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(GraphSAGE, self).__init__(SAGEConv, relu, input_dim, hidden_dim,
                                        output_dim, num_hidden_layers)


class MLP(SequentialLayersWithActivation):

    def __init__(self, input_dim, hidden_dim, output_dim, num_hidden_layers):
        super(MLP, self).__init__(Linear, relu, input_dim, hidden_dim,
                                  output_dim, num_hidden_layers)
