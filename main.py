from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import click
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
import keras.backend as K

from models.UNetValid import get_model
from utils import clustering
from tools.evaluation import final_prediction

@click.command()
@click.argument('train_img_folder', type=click.STRING)
@click.option('--test_img_folder', type=click.STRING, default='')
@click.option('--img_shape', nargs=2, type=click.INT, default=(256, 256))
@click.option('--num_clusters', type=click.INT, default=100)
@click.option('--num_epochs', type=click.INT, default=100)
@click.option('--learn_rate', type=click.FLOAT, default=1e-4)
@click.option('--clust_batch_size', type=click.INT, default=32)
@click.option('--train_batch_size', type=click.INT, default=4)
@click.option('--current_epoch', type=click.INT, default=0)
@click.option('--pretrained_weights', type=click.STRING, default='')
@click.option('--out_weights_file', type=click.STRING, default='')
@click.option('--out_pred_masks_test', type=click.STRING, default='')
def main(train_img_folder, test_img_folder, img_shape, num_clusters, num_epochs, learn_rate,
         clust_batch_size, train_batch_size, current_epoch, pretrained_weights, out_weights_file,
         out_pred_masks_test):
    assert os.path.isdir(train_img_folder)
    if train_img_folder[-1] != '/':
        train_img_folder += '/'
    if test_img_folder[-1] != '/':
        assert os.path.isdir(test_img_folder)
        test_img_folder += '/'

    train_image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        data_format='channels_last',
    )

    # # (512, 512) lux
    train_image_datagen.mean = np.array([11.4296465, 13.140564, 12.277675], dtype=np.float32).reshape(1, 1, 3)
    train_image_datagen.std = np.array([27.561337, 29.903225, 29.525864], dtype=np.float32).reshape(1, 1, 3)

    SEED = 1
    train_image_generator = train_image_datagen.flow_from_directory(train_img_folder,
                                                                    target_size=img_shape,
                                                                    batch_size=clust_batch_size,
                                                                    class_mode=None,
                                                                    shuffle=True,
                                                                    seed=SEED)


    test_image_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        data_format='channels_last',
    )

    test_image_datagen.mean = train_image_datagen.mean
    test_image_datagen.std = train_image_datagen.std


    test_image_generator = test_image_datagen.flow_from_directory(test_img_folder,
                                                                  target_size=img_shape,
                                                                  batch_size=32,
                                                                  class_mode=None,
                                                                  shuffle=False)

    num_images = train_image_generator.n
    model = get_model(train_image_generator.image_shape, num_classes=num_clusters)

    alpha = K.variable(1.)
    losses = {'clust_output': 'categorical_crossentropy'}
    loss_weights = {'clust_output': alpha}

    before_last_layer_model = Model(inputs=model.input, outputs=model.get_layer('for_clust').output)

    if pretrained_weights != '':
    #     for layer in model.layers[-1:]:
    #         layer.name += '_lux'
        model.load_weights(pretrained_weights, by_name=True)

    deepcluster = clustering.Kmeans(num_clusters)

    while current_epoch <= num_epochs:
        print('\n############# Epoch %i / %i #############' % ((current_epoch), num_epochs))
        num_iter = int(np.ceil(num_images / float(clust_batch_size)))
        for i in range(num_iter):
            print('%i:%i/%i' % (i * clust_batch_size, (i + 1) * clust_batch_size, num_images))

            images_batch = train_image_generator[i]

            print('Prediction:')
            features_batch = before_last_layer_model.predict(images_batch, verbose=1, batch_size=train_batch_size)

            flat_feat_batch = features_batch.reshape(-1, features_batch.shape[-1])
            print('Clustering:')
            deepcluster.cluster(flat_feat_batch, verbose=True)

            masks_batch_flat = np.zeros((flat_feat_batch.shape[0], 1))
            for clust_ind, clust in enumerate(deepcluster.images_lists):
                masks_batch_flat[clust] = clust_ind
            masks_batch_flat = to_categorical(masks_batch_flat, num_clusters)
            masks_batch = masks_batch_flat.reshape((features_batch.shape[0], features_batch.shape[1],
                                                    features_batch.shape[2], num_clusters))

            masks = {'clust_output': masks_batch}

            print('Training:')

            model.compile(optimizer=Adam(lr=(learn_rate)), loss=losses, loss_weights=loss_weights)
            model.fit(images_batch, masks, batch_size=train_batch_size, epochs=1, verbose=1)

        if out_pred_masks_test != '':
            out_pred_epoch_fold = os.path.join(out_pred_masks_test, str(current_epoch))
            final_pred_model = Model(inputs=model.input, outputs=model.get_layer('clust_output').output)
            final_prediction(out_pred_epoch_fold, test_image_generator, final_pred_model)
            model.save_weights(os.path.join(out_pred_epoch_fold, 'model.h5'))

        current_epoch += 1


if __name__ == '__main__':
    main()