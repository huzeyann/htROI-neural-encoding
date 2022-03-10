# source: https://github.com/richzhang/colorization

import numpy as np
import torch

from .colorizers import eccv16, siggraph17

#
# def modify_colorizer_partial(model, layers):
#     depths = [int(layer[1]) for layer in layers]
#     max_depth = np.max(depths)
#
#     def forward(self, input_A, input_B=None, mask_B=None):
#         if (input_B is None):
#             input_B = torch.cat((input_A * 0, input_A * 0), dim=1)
#         if (mask_B is None):
#             mask_B = input_A * 0
#
#         conv1_2 = self.model1(torch.cat((self.normalize_l(input_A), self.normalize_ab(input_B), mask_B), dim=1))
#         conv2_2 = self.model2(conv1_2[:, :, ::2, ::2])
#         conv3_3 = self.model3(conv2_2[:, :, ::2, ::2])
#         conv4_3 = self.model4(conv3_3[:, :, ::2, ::2])
#         conv5_3 = self.model5(conv4_3)
#         conv6_3 = self.model6(conv5_3)
#         conv7_3 = self.model7(conv6_3)
#
#         # conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
#         # conv8_3 = self.model8(conv8_up)
#         # conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
#         # conv9_3 = self.model9(conv9_up)
#         # conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
#         # conv10_2 = self.model10(conv10_up)
#         # out_reg = self.model_out(conv10_2)
#
#         ret_dict = {
#             'x1': conv2_2,
#             'x2': conv4_3,
#             'x3': conv6_3,
#             'x4': conv7_3,
#         }
#
#         return ret_dict
#
#     setattr(model.__class__, 'forward', forward)
#     return model
#
#
# def get_colorizer(pretrained=True):
#     model = siggraph17(pretrained=pretrained)
#     return model


def modify_colorizer_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, input_l):

        ret_dict = {}

        conv1_2 = self.model1(self.normalize_l(input_l))
        conv2_2 = self.model2(conv1_2)
        ret_dict['x1'] = conv2_2

        if max_depth >= 2:
            conv3_3 = self.model3(conv2_2)
            ret_dict['x2'] = conv3_3
        if max_depth >= 3:
            conv4_3 = self.model4(conv3_3)
            conv5_3 = self.model5(conv4_3)
            ret_dict['x3'] = conv5_3
        if max_depth >= 4:
            conv6_3 = self.model6(conv5_3)
            conv7_3 = self.model7(conv6_3)
            ret_dict['x4'] = conv7_3
        # conv8_3 = self.model8(conv7_3)

        # out = self.softmax(conv8_3)
        # out_reg = self.model_out(out)

        # return self.unnormalize_ab(self.upsample4(out_reg))
        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def get_colorizer(pretrained=True):
    model = eccv16(pretrained=pretrained)
    return model
