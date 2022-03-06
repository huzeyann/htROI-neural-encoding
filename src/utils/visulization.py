import itertools
from pathlib import Path

import nibabel as nib
import nilearn
import numpy as np
from PIL import Image, ImageChops
from matplotlib import pyplot as plt
from nilearn.plotting import plot_surf_stat_map
from nilearn.surface import vol_to_surf
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


def save_voxel_data_as_subject_mean_nii(voxel_masks: np.array, flattened_voxel_data: np.array,
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
    im.save(path, quality=100)


def make_horizontal_color_bar(cmap, vmax, vmin, cbar_label_text='',
                              figsize=(10, 10), fontsize=20, offset=-1.1):
    plt.rcParams["font.weight"] = "bold"
    plt.rcParams["axes.labelweight"] = "bold"

    fig, ax = plt.subplots(figsize=figsize)

    data = np.array([[vmin, vmax]])

    cax = ax.imshow(data, cmap=cmap)
    plt.gca().set_visible(False)

    cbar = fig.colorbar(cax, ticks=[vmin, vmax], orientation='horizontal', )
    cbar.ax.set_xticklabels([f'{vmin:.1f}', f'{vmax:.1f}'], fontsize=fontsize)  # horizontal colorbar

    cbar.ax.text(0., offset, cbar_label_text, rotation=0, fontsize=fontsize, ha="center", va="center", zorder=10)
    return cbar


def my_nice_nilearn_plot(
        nii_img_path,
        mask_img_path=None,
        tmp_dir='./tmp/figs',
        dpi=350,
        fsaverage='fsaverage',
        inflate=True,
        fill_nan_value=0,
        **kwargs, ):

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    modes = ['lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior']
    hemis = ['left', 'right']

    surf_mesh = nilearn.datasets.fetch_surf_fsaverage(fsaverage)

    mesh_prefix = "infl" if inflate else "pial"
    surf = {
        'left': surf_mesh[mesh_prefix + '_left'],
        'right': surf_mesh[mesh_prefix + '_right'],
    }
    texture = {
        'left': vol_to_surf(nii_img_path, surf_mesh['pial_left'],
                            mask_img=mask_img_path),
        'right': vol_to_surf(nii_img_path, surf_mesh['pial_right'],
                             mask_img=mask_img_path),
    }

    def fill_nan_f(arr):
        arr[np.isnan(arr)] = fill_nan_value
        return arr

    texture = dict(map(lambda kv: (kv[0], fill_nan_f(kv[1])), texture.items()))

    # plot and save to temp file
    for i, (mode, hemi) in enumerate(itertools.product(modes, hemis)):
        bg_map = surf_mesh['sulc_%s' % hemi]
        # ax = fig.add_subplot(grid[i + len(hemis)], projection="3d")
        # axes.append(ax)
        plot_surf_stat_map(surf[hemi],
                           texture[hemi],
                           view=mode,
                           hemi=hemi,
                           bg_map=bg_map,
                           colorbar=False,  # Colorbar created externally.
                           **kwargs
                           )
        # ax.set_facecolor("#e0e0e0")
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        plt.dist = 7
        path = tmp_dir.joinpath(Path(f'ax{i}_figure.jpeg'))
        plt.savefig(path,
                    dpi=dpi,
                    pil_kwargs={'quality': 100})
        plt.close()
        # plt.show()

        # trim white space
        trim_wrap(path)

    ### rearrange jpeg to a nice grid

    im1 = Image.open(tmp_dir.joinpath(Path(f'ax0_figure.jpeg')))

    ### defined from inkspace canvas
    grids_x = 100
    grids_y = 150
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
    ]

    for i in range(12):
        path = tmp_dir.joinpath(Path(f'ax{i}_figure.jpeg'))
        im = np.asarray(Image.open(path))
        x = int(starting_points[i][1] * (new_img.shape[0] / grids_y))
        y = int(starting_points[i][0] * (new_img.shape[1] / grids_x))
        new_img[x:x + im.shape[0], y:y + im.shape[1]] = im

    new_im = Image.fromarray(new_img)

    return new_im


def nilearn_6x2plot_to_nice_plot(fig, axes, dpi: int = 360, tmp_dir: str = './tmp'):
    assert len(axes) == 12 + 1  # 2x6 + 1 color bar

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
