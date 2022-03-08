# source: https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

download_url = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth'

import numpy as np
import torch

from .swin_transformer import SwinTransformer


def modify_seg_swin_2d_partial(model, layers):
    depths = [int(layer[1]) for layer in layers]
    max_depth = np.max(depths)

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        ret_dict = {}
        for i in range(self.num_layers):
            # drop layers to run faster
            if i + 1 > max_depth:
                break

            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww)

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x_out)

                out = x_out.view(-1, H, W, self.num_features[i]).permute(0, 3, 1, 2).contiguous()

                ret_dict[f'x{i + 1}'] = out

        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def load_backbone(model, state_dict):
    backbone_dict = {k.replace('backbone.', ''): v for k, v in state_dict['state_dict'].items() if
                     k.split('.')[0] == 'backbone'}

    model_dict = model.state_dict()
    # 2. overwrite entries in the existing state dict
    model_dict.update(backbone_dict)
    # 3. load the new state dict
    model.load_state_dict(backbone_dict)

    return model


def get_mmseg(pretrained=True):
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False
    )

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(download_url)
        load_backbone(model, state_dict)
    return model
