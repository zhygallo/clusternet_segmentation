from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import click
import os
import scipy.spatial
import sklearn.metrics
import skimage.io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.clusters2classes import gen_classes_from_clusters
from models.UNetValid import get_model


def final_prediction(out_fold, image_generator, model):
    num_iter = int(np.ceil(image_generator.n / float(image_generator.batch_size)))
    batch_size = image_generator.batch_size

    test_files = image_generator.filenames

    print('Final Predicton:')
    for i in range(num_iter):
        print('%i:%i/%i' % (i * batch_size, (i + 1) * batch_size, image_generator.n))

        if i < num_iter - 1:
            batch_files = test_files[i * batch_size: (i + 1) * batch_size]
        else:
            batch_files = test_files[i * batch_size:]

        batch_imgs = image_generator[i]

        pred_mask_probs = model.predict(batch_imgs, batch_size=1, verbose=1)
        for file, pred_mask_prob in zip(batch_files, pred_mask_probs):
            file_dir = os.path.dirname(file)
            file_name = file.split('/')[-1]
            full_out_dir = os.path.join(out_fold, file_dir)
            if not os.path.isdir(full_out_dir):
                os.makedirs(full_out_dir)
            mask = pred_mask_prob.argmax(axis=-1).astype('uint16')
            skimage.io.imsave(os.path.join(full_out_dir, file_name), mask)


def evaluate(input_gt, input_pred_mask, output_file=''):
    assert os.path.isdir(input_gt)
    if input_gt[-1] != '/':
        input_gt += '/'
    assert os.path.isdir(input_pred_mask)
    if input_pred_mask[-1] != '/':
        input_pred_mask += '/'

    result_dict = {}
    all_gt_files = []
    for dirpath, dirnames, filenames in os.walk(input_gt):
        all_gt_files.extend([os.path.join(os.path.relpath(dirpath, input_gt), f)
                             for f in filenames if f.endswith('.png')])

    all_pred_files_full_path = [os.path.join(input_pred_mask, file) for file in all_gt_files]
    all_gt_files_full_path = [os.path.join(input_gt, file) for file in all_gt_files]

    all_gt = np.asarray(skimage.io.imread_collection(all_gt_files_full_path))
    all_pred_mask = np.asarray(skimage.io.imread_collection(all_pred_files_full_path))

    result_dict['all_dice'] = 1 - scipy.spatial.distance.dice(all_gt.flatten(), all_pred_mask.flatten())
    result_dict['all_accuracy'] = sklearn.metrics.accuracy_score(all_gt.flatten(), all_pred_mask.flatten())

    if output_file != '':
        with open(output_file, 'w') as fp:
            json.dump(result_dict, fp, indent=2, sort_keys=True)

    return result_dict


def evaluate_per_epoch():
    test = 'test'
    in_result_dict = ''
    out_figure = 'data/patches_512_lux/figure_%s.png' % test

    if in_result_dict == '':
        # in_gt_train_fold = 'data/patches_512_lux/gt_masks/train_masks'
        in_gt_train_fold = 'data/patches_512_lux/gt_masks/test_masks'
        in_gt_test_fold = 'data/patches_512_lux/gt_masks/%s_masks' % test
        in_pred_clust_fold = 'data/patches_512_lux/pred_masks'
        out_fold_final_preds = 'data/patches_512_lux/final_preds'
        out_result_file = 'data/patches_512_lux/result_dict.json'

        gt_fraction = 0.6
        keep_clust_threshold = 0.4

        epochs = list(range(20))

        result = {}
        for epoch_dir in sorted(os.listdir(in_pred_clust_fold), key=int):
            if not int(epoch_dir) in epochs:
                continue
            epoch_dirpath = os.path.join(in_pred_clust_fold, epoch_dir)
            epoch_test_dirpath = os.path.join(epoch_dirpath)
            epoch_train_dirpath = os.path.join(epoch_dirpath)
            # epoch_test_dirpath = os.path.join(epoch_dirpath, test)
            # epoch_train_dirpath = os.path.join(epoch_dirpath, 'train')

            epoch_out_fold = os.path.join(out_fold_final_preds, epoch_dir)
            if not os.path.isdir(epoch_out_fold):
                os.makedirs(epoch_out_fold)

            if not set(os.listdir(epoch_test_dirpath)) < set(os.listdir(epoch_out_fold)):
                gen_classes_from_clusters(in_clust_test=epoch_test_dirpath,
                                          in_clust_train=epoch_train_dirpath,
                                          in_gt_train=in_gt_train_fold,
                                          gt_fraction=gt_fraction,
                                          keep_clust_threshold=keep_clust_threshold,
                                          out_fold=epoch_out_fold)

            result[epoch_dir] = evaluate(input_gt=in_gt_test_fold, input_pred_mask=epoch_out_fold)

        with open(out_result_file, 'w') as fp:
            json.dump(result, fp, indent=2, sort_keys=True)

    else:
        with open(in_result_dict, 'r') as json_data:
            result = json.load(json_data)

    df = pd.DataFrame({'epochs': list(result.keys()), 'dice': [item['all_dice'] for item in result.values()],
                       'accuracy': [item['all_accuracy'] for item in result.values()]})

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot('epochs', 'dice', data=df, marker='o', markerfacecolor='black', color='blue', linewidth=4)
    # ax.plot('epochs', 'accuracy', data=df, marker='o', markerfacecolor='black', color='green', linewidth=4)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Dice')
    fig.savefig(out_figure)


@click.command()
@click.option('--image_path', type=click.STRING)
@click.option('--pred_mask_path', type=click.STRING, default='')
@click.option('--model_weights', type=click.STRING, default='')
@click.option('--num_clust', type=click.INT, default=2)
@click.option('--gt_mask_path', type=click.STRING, default='')
@click.option('--out_pred_mask', type=click.STRING, default='')
def main(image_path, pred_mask_path, model_weights, num_clust, gt_mask_path, out_pred_mask):
    if pred_mask_path == '':
        image = skimage.io.imread(image_path).astype('float32')
        mean = np.array([11.4296465, 13.140564, 12.277675], dtype=np.float32).reshape(1, 1, 3)
        std = np.array([27.561337, 29.903225, 29.525864], dtype=np.float32).reshape(1, 1, 3)
        image -= mean
        image /= std
        model = get_model(image.shape, num_clust)
        model.load_weights(model_weights)

        pred_mask = model.predict(np.expand_dims(image, axis=0)).argmax(axis=-1).astype('uint16')[0]
    else:
        pred_mask = skimage.io.imread(pred_mask_path)

    gt_mask = skimage.io.imread(gt_mask_path)

    dice = 1 - scipy.spatial.distance.dice(gt_mask.flatten(), pred_mask.flatten())
    accuracy = sklearn.metrics.accuracy_score(gt_mask.flatten(), pred_mask.flatten())

    overlap = gt_mask.flatten() * pred_mask.flatten()
    union = gt_mask.flatten() + pred_mask.flatten()

    IOU = overlap.sum()/float((union>0).sum())

    print('Dice Score: %f' % dice)
    print('Accuracy: %f' % accuracy)
    print('IoU: %f' % IOU)

    if out_pred_mask != '' and pred_mask_path == '':
        skimage.io.imsave(out_pred_mask, pred_mask)


if __name__ == '__main__':
    main()
