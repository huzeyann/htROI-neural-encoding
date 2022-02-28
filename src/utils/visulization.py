import itertools
from pathlib import Path

import numpy as np
from PIL.Image import Image

from PIL import Image, ImageChops

import nibabel as nib
from nilearn import plotting
from tqdm.auto import tqdm


def save_as_nii(example_nii_path, nii_save_path, nii_data):
    img = nib.load(example_nii_path)
    nii_img = nib.Nifti1Image(nii_data, img.affine, img.header)
    nib.save(nii_img, nii_save_path)


def mean_over_subs(voxel_masks, voxel_data):
    # voxel_data: shape [num_subjects, x, y, x]
    d = voxel_masks.sum(0)
    d[d == 0] = 1
    d = np.stack([d for _ in range(voxel_masks.shape[0])])
    v = voxel_data / d
    v = v.sum(0)
    return v


def get_nii(voxel_masks, flattened_voxel_data,
            example_nii_path, nii_save_path):
    voxel_data = np.zeros(shape=voxel_masks.shape)
    voxel_data[voxel_masks == 1] = flattened_voxel_data
    voxel_data = mean_over_subs(voxel_masks, voxel_data)

    save_as_nii(example_nii_path, nii_save_path, voxel_data)


def trim_bg(im, border=(255, 255, 255)):
    bg = Image.new(im.mode, im.size, border)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def trim_wrap(path):
    im = Image.open(path)
    im = trim_bg(im)
    im.save(path)


def nice_plot(fig, axes, dpi: int = 360, tmp_dir: str = './tmp'):
    assert len(axes) == 12 + 1 # 2x6 + 1 color bar

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(exist_ok=True)

    for i in tqdm(range(12), desc='axes'):
        ax = axes[i]
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        path = tmp_dir.joinpath(Path(f'ax{i}_figure.jpeg'))
        fig.savefig(path, bbox_inches=extent.expanded(0.95, 0.95), dpi=dpi,
                    pil_kwargs={'quality': 100})
        trim_wrap(path)

    ax = axes[-1]  # cbar
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    path = tmp_dir.joinpath(Path(f'ax_cbar_figure.jpeg'))
    fig.savefig(path, bbox_inches=extent.expanded(1.5, 4.0), dpi=dpi,
                pil_kwargs={'quality': 100})
    trim_wrap(path)

    im1 = Image.open(tmp_dir.joinpath(Path(f'ax0_figure.jpeg')))

    ### from canvas
    grids_x = 100
    grids_y = 160
    new_size = (int(np.ceil(im1.size[1] * (grids_y / 34.285))),
                int(np.ceil(im1.size[0] * (grids_x / 47.790))),
                3)

    new_img = np.ones(shape=(new_size), dtype=np.uint8) * 255

    starting_points = [
        (0, 0),
        (50, 0),
        (0, 38),
        (50, 38),
        (0, 76),
        (0, 94),
        (50, 94),
        (50, 76),
        (24, 114),
        (5, 114),
        (55, 114),
        (74, 114),
        (33, 150)
    ]

    for i in range(12):
        path = tmp_dir.joinpath(Path(f'ax{i}_figure.jpeg'))
        im = np.asarray(Image.open(path))
        x = int(starting_points[i][1] * (new_img.shape[0] / grids_y))
        y = int(starting_points[i][0] * (new_img.shape[1] / grids_x))
        new_img[x:x + im.shape[0], y:y + im.shape[1]] = im

    path = tmp_dir.joinpath(Path(f'ax_cbar_figure.jpeg'))
    im = np.asarray(Image.open(path))
    x = int(starting_points[-1][1] * (new_img.shape[0] / grids_y))
    y = int(starting_points[-1][0] * (new_img.shape[1] / grids_x))
    new_img[x:x + im.shape[0], y:y + im.shape[1]] = im

    new_im = Image.fromarray(new_img)

    # new_im.save(new_save_path, quality=95)

    return new_im
