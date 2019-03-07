"""Script for generating patches"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import click
import skimage.io
import skimage.util
import numpy as np

# mean = np.array([11.4296465, 13.140564, 12.277675], dtype=np.float32).reshape(1, 1, 3)
# std = np.array([27.561337, 29.903225, 29.525864], dtype=np.float32).reshape(1, 1, 3)

def get_patches(input_fold, output_folder='', output_np_folder='', masks_folder='',
                patch_shape=(512, 512), crop_shape=None, stride=(100, 100), keep_dark_region_ratio=0.01):
    """
    Generates cropped patches from the full-size input images
    Args:
        input_fold:     string.         path to the raw train data folder
        output_folder:  string.         path to the output folder generated patches
        output_np_folder: string.       path to the output folder with numpy file
        masks_folder:   string.         if not '' generates a patch with a mask corresponding to an image patch
        patch_shape:    tuple of ints.  (height, width), specifies patch shape
        crop_shape:     tuple of ints.  (height, width), specifies the area cropped from the patch.
        stride:         tuple of ints.  stride in height and width direction of a sliding window
        keep_dark_region_ratio: float.  ratio of black pixels which is allowed for a patch to be considered
    Returns:
        list of numpy arrays. image patches and mask patches
    """

    assert os.path.isdir(input_fold)
    if input_fold[-1] != '/':
        input_fold += '/'
    if output_folder != '':
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)
        if output_folder[-1] != '/':
            output_folder += '/'
    if output_np_folder != '':
        assert os.path.isdir(output_np_folder)
        if output_np_folder[-1] != '/':
            output_np_folder += '/'
    if masks_folder != '':
        assert os.path.isdir(masks_folder)
        if masks_folder[-1] != '/':
            masks_folder += '/'
    assert type(patch_shape) is tuple
    assert type(patch_shape[0]) is int and type(patch_shape[1]) is int
    assert patch_shape[0] > 0 and patch_shape[1] > 0

    if crop_shape != None:
        crop_bef_h = (patch_shape[0] - crop_shape[0]) // 2
        crop_bef_w = (patch_shape[1] - crop_shape[1]) // 2

    assert type(stride) is tuple
    assert type(stride[0]) is int and type(stride[1]) is int
    assert stride[0] > 0 and stride[1] > 0

    img_count = 0
    for subdir, dir, files in os.walk(input_fold):
        for file in files:
            if file.split('.')[-1] not in ['tiff', 'tif', 'png', 'jpg', 'JPEG', 'jpeg']:
                continue

            img_patches = []
            mask_patches = []

            img_file = os.path.join(subdir, file)
            full_img = skimage.io.imread(img_file)

            if masks_folder != '':
                mask_file = os.path.join(masks_folder, file.split('.')[0] + '_mask.' + file.split('.')[-1])
                # mask_file = os.path.join(masks_folder, file.split('.')[0] + '.' + file.split('.')[-1])
                assert os.path.isfile(mask_file)
                full_mask = skimage.io.imread(mask_file)
                # assert full_img.shape[0] == full_mask.shape[0] and full_img.shape[1] == full_mask.shape[1]

            for row in range(0, full_img.shape[0] - patch_shape[0], stride[0]):
                for col in range(0, full_img.shape[1] - patch_shape[1], stride[1]):
                    crop_img = full_img[row:row + patch_shape[0], col:col + patch_shape[1]]
                    # do not save patches where the number of black pixels is higher than skip_dark_region_ratio
                    if (crop_img == 0).sum() / crop_img.size > keep_dark_region_ratio:
                        continue

                    img_patches.append(crop_img)

                    if masks_folder != '':
                        crop_mask = full_mask[row:row + patch_shape[0], col:col + patch_shape[1]]
                        mask_patches.append(crop_mask)

            if output_folder != '':
                full_path = os.path.join(output_folder, file.split('.')[0])
                if not os.path.isdir(full_path):
                    os.mkdir(full_path)
                for ind, img_patch in enumerate(img_patches):
                    skimage.io.imsave(os.path.join(full_path, str(img_count + ind).zfill(6) + '.png'), img_patch)
                    if crop_shape != None:
                        crop_patch_out = os.path.join(output_folder, 'crops', file.split('.')[0])
                        if not os.path.isdir(crop_patch_out):
                            os.makedirs(crop_patch_out)
                        crop_patch = img_patch[crop_bef_h:crop_bef_h + crop_shape[0],
                                     crop_bef_w:crop_bef_w + crop_shape[1]]
                        skimage.io.imsave(os.path.join(crop_patch_out, str(img_count + ind).zfill(6) + '.png'),
                                          crop_patch)

                if masks_folder != '':
                    mask_output_folder = os.path.join(output_folder, 'masks', file.split('.')[0])
                    if not os.path.isdir(mask_output_folder):
                        os.makedirs(mask_output_folder)
                    for ind, mask_patch in enumerate(mask_patches):
                        if crop_shape != None:
                            mask_patch = mask_patch[crop_bef_h:crop_bef_h + crop_shape[0],
                                         crop_bef_w:crop_bef_w + crop_shape[1]]
                        skimage.io.imsave(os.path.join(mask_output_folder, str(img_count + ind).zfill(6) + '.png'),
                                          mask_patch.astype('uint8'))

            img_count += len(img_patches)

            if output_np_folder != '':
                full_path = os.path.join(output_folder, file.split('.')[0])
                if not os.path.isdir(full_path):
                    os.mkdir(full_path)
                img_patches_np = np.asarray(img_patches, dtype='float32')
                np.save(os.path.join(full_path, 'img_patches'), img_patches_np)
                mask_patches_np = np.asarray(mask_patches)
                np.save(os.path.join(full_path, 'mask_patches'), mask_patches_np)

    return img_patches, mask_patches


@click.command()
@click.argument('input_folder', type=click.STRING)
@click.option('--output_folder', type=click.STRING, default='')
@click.option('--output_np_folder', type=click.STRING, default='')
@click.option('--masks_folder', type=click.STRING, default='')
@click.option('--patch_shape', nargs=2, type=click.INT, default=(512, 512), help='Height and width of a crop')
@click.option('--crop_shape', nargs=2, type=click.INT, default=(309, 309))
@click.option('--stride', nargs=2, type=click.INT, default=(100, 100), help='Stride in height and widht direction')
@click.option('--keep_dark_region_ratio', type=click.FLOAT, default=0.01)
def main(input_folder, output_folder, output_np_folder, masks_folder, patch_shape, crop_shape,
         stride, keep_dark_region_ratio):
    get_patches(input_folder, output_folder=output_folder, output_np_folder=output_np_folder,
                masks_folder=masks_folder, patch_shape=patch_shape, crop_shape=crop_shape, stride=stride,
                keep_dark_region_ratio=keep_dark_region_ratio)


if __name__ == '__main__':
    main()
