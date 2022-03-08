import numpy as np
import torchvision


def modify_densnet_2d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        x = self.features[:4](x)

        ret_dict = {}

        x1 = self.features[4:6](x)
        ret_dict['x1'] = x1
        if max_depth >= 2:
            x2 = self.features[6:8](x1)
            ret_dict['x2'] = x2
        if max_depth >= 3:
            x3 = self.features[8:10](x2)
            ret_dict['x3'] = x3
        if max_depth >= 4:
            x4 = self.features[10:12](x3)
            ret_dict['x4'] = x4

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def densnet169(pretrained=True):
    model = torchvision.models.densenet169(pretrained=pretrained)
    return model
