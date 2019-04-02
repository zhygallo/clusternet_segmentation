from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import skimage.io
import os
import click


@click.command()
@click.argument('base_image_file', type=click.STRING)
@click.argument('pos_neg_mask_file', type=click.STRING)
@click.argument('masks_folder', type=click.STRING)
@click.option('--output_folder', type=click.STRING, default='')
def main(base_image_file, pos_neg_mask_file, masks_folder, output_folder):
    base_image = skimage.io.imread(base_image_file)
    pos_neg_mask = np.load(pos_neg_mask_file)

    pos_clusts = np.unique(base_image[pos_neg_mask == 1])
    neg_clusts = np.unique(base_image[pos_neg_mask == -1])

    intersect = np.intersect1d(pos_clusts, neg_clusts)

    for clust in intersect:
        pos = np.sum((base_image == clust) * (pos_neg_mask == 1))
        neg = np.sum((base_image == clust) * (pos_neg_mask == -1))
        pos_ratio = pos / (pos + neg)

        if pos_ratio < 0.0:
            index = np.argwhere(pos_clusts == clust)
            pos_clusts = np.delete(pos_clusts, index)

    for file in os.listdir(masks_folder):
        file_path = os.path.join(masks_folder, file)
        mask = skimage.io.imread(file_path).astype('float32')

        if output_folder != '':
            out_file_path = os.path.join(output_folder, file)
            skimage.io.imsave(out_file_path, np.isin(mask, pos_clusts))


if __name__ == "__main__":
    main()
