from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import os
import skimage.io
import skimage.util
import numpy
import math

@click.command()
@click.argument('input_images', type=click.STRING)
@click.option('--crop_shape', nargs=2, type=click.INT, default=(309, 309))
@click.option('--output_folder', type=click.STRING)
def main(input_images, crop_shape, output_folder):
    assert os.path.isdir(input_images)

    height_crop = int((512 - crop_shape[0])) / 2
    width_crop = int((512 - crop_shape[1])) / 2

    for file in os.listdir(input_images):
        image_file = os.path.join(input_images, file)
        image = skimage.io.imread(image_file)

        crop_img = skimage.util.crop(image, ((math.ceil(height_crop), int(height_crop)), (math.ceil(width_crop), int(width_crop))))

        out_image_file = os.path.join(output_folder, file)
        skimage.io.imsave(out_image_file, crop_img)

    return 0


if __name__ == "__main__":
    main()