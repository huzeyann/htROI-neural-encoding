# source:

# backbone download link:
download_url = 'https://drive.google.com/uc?export=download&id=1s4-n58ZClFJwVbnrO74qgn8leir8Dj4l'
import os

import numpy as np
import torch
from einops import rearrange
from .network.vggish import VGGish


def modify_vggish_audio_partial(model):
    def forward(self, x):
        ret_dict = {}
        batch_size = x.shape[0]
        x = rearrange(x, 'b t c h w -> (b t) c h w')

        x = self.features[0:3](x)
        ret_dict['x1'] = rearrange(x, '(b t) c h w -> b (t c) h w', b=batch_size)

        x = self.features[3:6](x)
        ret_dict['x2'] = rearrange(x, '(b t) c h w -> b (t c) h w', b=batch_size)

        x = self.features[6:11](x)
        ret_dict['x3'] = rearrange(x, '(b t) c h w -> b (t c) h w', b=batch_size)

        x = self.features[11:16](x)
        ret_dict['x4'] = rearrange(x, '(b t) c h w -> b (t c) h w', b=batch_size)

        # Perform feature extraction with CNN module, and change layout to make compatible with embedding layers.
        x = x.permute(0, 2, 3, 1)
        # Perform embedding generation with FC module
        x = self.embedding(x)

        ret_dict['x_label'] = rearrange(x, '(b t) c -> b (t c)', b=batch_size)
        return ret_dict

    setattr(model.__class__, 'forward', forward)
    return model


def get_vggish_torch(pretrained=True, cache_dir='~/.cache/'):
    model = VGGish()

    if pretrained:
        cache_dir = os.path.expanduser(cache_dir)
        path = os.path.join(cache_dir, 'vggish_model.pt')
        if not os.path.exists(path):
            raise Exception(f'{path} not exists! please download from: {download_url}')
        state_dict = torch.load(path)
        model.load_state_dict(state_dict)

    return model
