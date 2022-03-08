# source: https://github.com/pkuCactus/BDCN

# backbone download link:
download_url = 'https://drive.google.com/u/0/uc?id=1CmDMypSlLM6EAvOt5yjwUQ7O5w-xCm1n&export=download'
# unzip after download
import os

import numpy as np
import torch

from .bdcn import BDCN, crop

import torch.nn.functional as F


def modify_bdcnvgg_2d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        features = self.features(x)
        sum1 = self.conv1_1_down(self.msblock1_1(features[0])) + \
               self.conv1_2_down(self.msblock1_2(features[1]))
        s1 = self.score_dsn1(sum1)
        s11 = self.score_dsn1_1(sum1)
        sum2 = self.conv2_1_down(self.msblock2_1(features[2])) + \
               self.conv2_2_down(self.msblock2_2(features[3]))
        s2 = self.score_dsn2(sum2)
        s21 = self.score_dsn2_1(sum2)
        s2 = self.upsample_2(s2)
        s21 = self.upsample_2(s21)
        s2 = crop(s2, x, 1, 1)
        s21 = crop(s21, x, 1, 1)
        sum3 = self.conv3_1_down(self.msblock3_1(features[4])) + \
               self.conv3_2_down(self.msblock3_2(features[5])) + \
               self.conv3_3_down(self.msblock3_3(features[6]))
        s3 = self.score_dsn3(sum3)
        s3 = self.upsample_4(s3)
        s3 = crop(s3, x, 2, 2)
        s31 = self.score_dsn3_1(sum3)
        s31 = self.upsample_4(s31)
        s31 = crop(s31, x, 2, 2)
        sum4 = self.conv4_1_down(self.msblock4_1(features[7])) + \
               self.conv4_2_down(self.msblock4_2(features[8])) + \
               self.conv4_3_down(self.msblock4_3(features[9]))
        s4 = self.score_dsn4(sum4)
        s4 = self.upsample_8(s4)
        s4 = crop(s4, x, 4, 4)
        s41 = self.score_dsn4_1(sum4)
        s41 = self.upsample_8(s41)
        s41 = crop(s41, x, 4, 4)
        sum5 = self.conv5_1_down(self.msblock5_1(features[10])) + \
               self.conv5_2_down(self.msblock5_2(features[11])) + \
               self.conv5_3_down(self.msblock5_3(features[12]))
        s5 = self.score_dsn5(sum5)
        s5 = self.upsample_8_5(s5)
        s5 = crop(s5, x, 0, 0)
        s51 = self.score_dsn5_1(sum5)
        s51 = self.upsample_8_5(s51)
        s51 = crop(s51, x, 0, 0)
        o1, o2, o3, o4, o5 = s1.detach(), s2.detach(), s3.detach(), s4.detach(), s5.detach()
        o11, o21, o31, o41, o51 = s11.detach(), s21.detach(), s31.detach(), s41.detach(), s51.detach()
        p1_1 = s1
        p2_1 = s2 + o1
        p3_1 = s3 + o2 + o1
        p4_1 = s4 + o3 + o2 + o1
        p5_1 = s5 + o4 + o3 + o2 + o1
        p1_2 = s11 + o21 + o31 + o41 + o51
        p2_2 = s21 + o31 + o41 + o51
        p3_2 = s31 + o41 + o51
        p4_2 = s41 + o51
        p5_2 = s51

        fuse = self.fuse(torch.cat([p1_1, p2_1, p3_1, p4_1, p5_1, p1_2, p2_2, p3_2, p4_2, p5_2], 1))

        # ret_dict = {
        #     'x1': sum2,
        #     'x2': sum3,
        #     'x3': sum4,
        #     'x4': sum5,
        # }

        ret_dict = {
            'x1': F.sigmoid(p3_1),
            'x2': F.sigmoid(p1_2),
            'x3': F.sigmoid(p4_2),
            'x4': F.sigmoid(fuse),
        }

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def get_bdcn(pretrained=True, cache_dir='~/.cache/'):
    model = BDCN()

    if pretrained:
        cache_dir = os.path.expanduser(cache_dir)
        path = os.path.join(cache_dir, 'bdcn_pretrained_on_bsds500.pth')
        if not os.path.exists(path):
            raise Exception(f'{path} not exists! please download from: {download_url}')
        model.load_state_dict(torch.load(path))
    return model
