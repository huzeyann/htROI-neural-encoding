import torch
import torch.nn as nn


def build_fc(cfg, input_dim, output_dim, part='full'):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
    }

    layer_hidden = cfg.MODEL.NECK.FC_HIDDEN_DIM

    module_list = []
    for i in range(cfg.MODEL.NECK.FC_NUM_LAYERS):
        if i == 0:
            size1, size2 = input_dim, layer_hidden
        else:
            size1, size2 = layer_hidden, layer_hidden
        module_list.append(nn.Linear(size1, size2))
        if cfg.MODEL.NECK.FC_BATCH_NORM:
            module_list.append(nn.BatchNorm1d(size2))
        module_list.append(activations[cfg.MODEL.NECK.FC_ACTIVATION])
        if cfg.MODEL.NECK.FC_DROPOUT > 0:
            module_list.append(nn.Dropout(cfg.MODEL.NECK.FC_DROPOUT))
        if i == 0:
            layer_one_size = len(module_list)

    # last layer
    module_list.append(nn.Linear(size2, output_dim))

    if part == 'full':
        module_list = module_list
    elif part == 'first':
        module_list = module_list[:layer_one_size]
    elif part == 'last':
        module_list = module_list[layer_one_size:]
    else:
        NotImplementedError()

    return nn.Sequential(*module_list)


class FcFusion(nn.Module):
    def __init__(self, fusion_type='concat'):
        super(FcFusion, self).__init__()
        assert fusion_type in ['add', 'avg', 'concat', ]
        self.fusion_type = fusion_type

    def init_weights(self):
        pass

    def forward(self, input):
        assert (isinstance(input, tuple)) or (isinstance(input, dict)) or (isinstance(input, list))
        if isinstance(input, dict):
            input = tuple(input.values())

        if self.fusion_type == 'add':
            out = torch.sum(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'avg':
            out = torch.mean(torch.stack(input, -1), -1, keepdim=False)

        elif self.fusion_type == 'concat':
            out = torch.cat(input, -1)

        else:
            raise ValueError

        return out
