from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import skimage.io
import numpy as np
import random
import os


def gen_classes_from_clusters(in_clust_test, in_clust_train, in_gt_train, gt_fraction, keep_clust_threshold,
                              out_fold, per_cluster_out_fold=''):
    assert os.path.isdir(in_clust_test)
    if in_clust_test[-1] != '/':
        in_clust_test += '/'
    assert os.path.isdir(in_clust_train)
    if in_clust_train[-1] != '/':
        in_clust_train += '/'
    assert os.path.isdir(in_gt_train)
    if in_gt_train[-1] != '/':
        in_gt_train += '/'
    assert type(gt_fraction) is float
    assert gt_fraction > 0 and gt_fraction <= 1
    assert type(keep_clust_threshold) is float
    assert keep_clust_threshold >= 0 and keep_clust_threshold <= 1
    if out_fold != '':
        if not os.path.isdir(out_fold):
            os.mkdir(out_fold)
    if per_cluster_out_fold != '':
        if not os.path.isdir(per_cluster_out_fold):
            os.mkdir(per_cluster_out_fold)

    all_clust_test_files = []
    for dirpath, dirnames, filenames in os.walk(in_clust_test):
        all_clust_test_files.extend([os.path.join(dirpath, f)
                                     for f in filenames if f.endswith('.png')])

    all_clust_train_files = []
    for dirpath, dirnames, filenames in os.walk(in_clust_train):
        all_clust_train_files.extend([os.path.join(os.path.relpath(dirpath, in_clust_train), f)
                                      for f in filenames if f.endswith('.png')])

    all_gt_train_files = []
    for dirpath, dirnames, filenames in os.walk(in_gt_train):
        all_gt_train_files.extend([os.path.join(os.path.relpath(dirpath, in_gt_train), f)
                                   for f in filenames if f.endswith('.png')])


    assert len(all_gt_train_files) * gt_fraction > 0
    assert len(all_clust_train_files) * gt_fraction > 0

    SEED = 1
    random.seed(SEED)
    random.shuffle(all_gt_train_files)
    num_gt = int(gt_fraction*len(all_gt_train_files))
    sub_gt_files = all_gt_train_files[:num_gt]
    assert np.sum([gt_file in all_clust_train_files for gt_file in sub_gt_files]) == len(sub_gt_files)

    sub_gt_train_full_path = [os.path.join(in_gt_train, file) for file in sub_gt_files]
    sub_clust_train_full_path = [os.path.join(in_clust_train, file) for file in sub_gt_files]

    gt_train_masks = np.asarray(skimage.io.imread_collection(sub_gt_train_full_path))
    clust_train_masks = np.asarray(skimage.io.imread_collection(sub_clust_train_full_path))

    selected_clusters = []
    for clust in np.unique(clust_train_masks):
        clust_pixel_num = (clust_train_masks == clust).sum()
        gt_pixel_num = (gt_train_masks[clust_train_masks == clust] == 1).sum()
        if gt_pixel_num / clust_pixel_num >= keep_clust_threshold:
            selected_clusters.append(clust)

    result_masks = []
    for clust_test_file in all_clust_test_files:
        clust_image = skimage.io.imread(clust_test_file)
        result_mask = np.zeros(clust_image.shape)
        for c in selected_clusters:
            result_mask[clust_image == c] = 1
        result_masks.append(result_mask)

        if out_fold != '':
            rel_path = os.path.relpath(clust_test_file, in_clust_test)
            full_out_path = os.path.join(out_fold, rel_path)
            if not os.path.isdir(os.path.dirname(full_out_path)):
                os.makedirs(os.path.dirname(full_out_path))
            skimage.io.imsave(full_out_path, result_mask.astype('uint8'))

    return result_masks


@click.command()
@click.argument('in_clust_test', type=click.STRING)
@click.argument('in_clust_train', type=click.STRING)
@click.argument('in_gt_train', type=click.STRING)
@click.option('--gt_fraction', type=click.FLOAT, default='')
@click.option('--keep_clust_threshold', type=click.FLOAT, default=0)
@click.option('--out_fold', type=click.STRING, default='')
@click.option('--per_cluster_out_fold', type=click.STRING, default='')
def main(in_clust_test, in_clust_train, in_gt_train, gt_fraction,
         keep_clust_threshold, out_fold, per_cluster_out_fold):
    gen_classes_from_clusters(in_clust_test, in_clust_train, in_gt_train, gt_fraction=gt_fraction,
                              keep_clust_threshold=keep_clust_threshold, out_fold=out_fold,
                              per_cluster_out_fold=per_cluster_out_fold)


if __name__ == '__main__':
    main()
