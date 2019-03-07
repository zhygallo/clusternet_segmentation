from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import os
import skimage.io

from osgeo import gdal


def array_to_tiff_file(image, output_file, geoprojection=None, geotransform=None):
    """Gets images in a form of numpy arrays and saves them as images with spacial information
    Args:
        image:  numpy array.    input image of shape (num_channels, w, h) or (w, h) in case of 1 channel
        output_file:    string. path to the output file
        geoprojection:  driver_in.GetProjection()
        geotransform:   driver_in.GetGeoTransform()
    Returns:
        -
    """
    driver = gdal.GetDriverByName("GTiff")

    if image.ndim == 3:
        n_channels, h, w = image.shape
    elif image.ndim == 2:
        h, w = image.shape
        n_channels = 1
    ds_out = driver.Create(output_file, w, h, n_channels, gdal.GDT_Float32)
    if geoprojection and geotransform:
        ds_out.SetProjection(geoprojection)
        ds_out.SetGeoTransform(geotransform)

    # write channels separately
    for i in range(n_channels):
        band = ds_out.GetRasterBand(i + 1)  # 1-based index
        if n_channels > 1:
            channel = image[i]
        else:
            channel = image
        band.WriteArray(channel)
        band.FlushCache()

    del ds_out


def get_prediction(in_full_img_fold, in_pred_mask_fold, crop_shape=(512, 512), out_fold=''):
    """Generates masks of a crop_shape for images in the input_folder
    Args:
        input_folder:   string. path to a folder with images on which model will be applied
        output_folder:  string. path to the output folder with predicted masks
    Returns:
        -
    """
    assert os.path.isdir(in_full_img_fold)
    if in_full_img_fold[-1] != '/':
        in_full_img_fold += '/'
    if out_fold != '':
        if not os.path.isdir(out_fold):
            os.mkdir(out_fold)
        if out_fold != '/':
            out_fold += '/'

    img_files = os.listdir(in_full_img_fold)
    for img_file in img_files:
        dataset = gdal.Open(os.path.join(in_full_img_fold, img_file))

        geotransform = dataset.GetGeoTransform()
        geoprojection = dataset.GetProjection()

        x_origin = geotransform[0]
        y_origin = geotransform[3]
        pixel_width = geotransform[1]
        pixel_height = geotransform[5]

        image = dataset.ReadAsArray()
        image = image.transpose((1, 2, 0))

        mask_files = os.listdir(in_pred_mask_fold)
        mask_files.sort()

        count = 0
        for row in range(0, image.shape[0] - crop_shape[0], crop_shape[0]):
            for col in range(0, image.shape[1] - crop_shape[1], crop_shape[1]):
                crop_img = image[row:row + crop_shape[0], col:col + crop_shape[1], :].astype('float32')
                if (crop_img == 0).sum() / crop_img.size > 0.1:
                    continue
                crop_geotransform = (x_origin + col * pixel_width, pixel_width, geotransform[2],
                                     y_origin + row * pixel_height, geotransform[4], pixel_height)
                full_path_mask = os.path.join(in_pred_mask_fold, mask_files[count])
                pred_mask = skimage.io.imread(full_path_mask)

                if out_fold != '':
                    if not os.path.isdir(os.path.join(out_fold, img_file.split('.')[0])):
                        os.mkdir(os.path.join(out_fold, img_file.split('.')[0]))
                    array_to_tiff_file(pred_mask,
                                       os.path.join(out_fold, img_file.split('.')[0],
                                                    'pred_mask_' + str(count).zfill(6) + '.tif'),
                                       geoprojection=geoprojection,
                                       geotransform=crop_geotransform)

                count += 1


@click.command()
@click.argument('in_full_img_fold', type=click.STRING)
@click.argument('in_pred_mask_fold', type=click.STRING)
@click.option('--crop_shape', nargs=2, type=click.INT, default=(512, 512))
@click.option('--out_fold', type=click.STRING, default='')
def main(in_full_img_fold, in_pred_mask_fold, crop_shape, out_fold):
    get_prediction(in_full_img_fold, in_pred_mask_fold, crop_shape=crop_shape, out_fold=out_fold)


if __name__ == '__main__':
    main()
