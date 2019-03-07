from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import click
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import Model

from models.ContextEncoder import get_model
from tools.evaluation import final_prediction

def crop_center(image):
    crop_shape = (309, 309)
    crop_bef_h = (image.shape[0] - crop_shape[0]) // 2
    crop_bef_w = (image.shape[1] - crop_shape[1]) // 2
    out_img = image.copy()
    out_img[crop_bef_h:crop_bef_h + crop_shape[0], crop_bef_w:crop_bef_w + crop_shape[1]] = 0.5
    return out_img

@click.command()
@click.argument('train_img_folder', type=click.STRING)
@click.argument('train_mask_folder', type=click.STRING)
@click.option('--test_img_folder', type=click.STRING, default='')
@click.option('--test_mask_folder', type=click.STRING, default='')
@click.option('--in_img_shape', nargs=2, type=click.INT, default=(512, 512))
@click.option('--out_img_shape', nargs=2, type=click.INT, default=(309, 309))
@click.option('--num_classes', type=click.INT, default=800)
@click.option('--num_epochs', type=click.INT, default=100)
@click.option('--learn_rate', type=click.FLOAT, default=1e-3)
@click.option('--batch_size', type=click.INT, default=4)
@click.option('--current_epoch', type=click.INT, default=0)
@click.option('--pretrained_weights', type=click.STRING, default='')
@click.option('--out_weights_file', type=click.STRING, default='')
@click.option('--out_pred_masks_test', type=click.STRING, default='')
def main(train_img_folder, train_mask_folder, test_img_folder, test_mask_folder,
         in_img_shape, out_img_shape, num_classes, num_epochs, learn_rate, batch_size,
         current_epoch, pretrained_weights, out_weights_file, out_pred_masks_test):
    assert os.path.isdir(train_img_folder)
    if train_img_folder[-1] != '/':
        train_img_folder += '/'
    assert os.path.isdir(test_img_folder)
    if test_img_folder[-1] != '/':
        test_img_folder += '/'

    train_image_datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,
        data_format='channels_last',
        preprocessing_function=crop_center
    )

    train_mask_datagen = ImageDataGenerator(
        rescale=1. / 255,
        # featurewise_center=True,
        # featurewise_std_normalization=True,
    )

    # (512, 512) lux
    # train_image_datagen.mean = np.array([11.4296465, 13.140564, 12.277675], dtype=np.float32).reshape(1, 1, 3)
    # train_image_datagen.std = np.array([27.561337, 29.903225, 29.525864], dtype=np.float32).reshape(1, 1, 3)

    SEED = 1
    train_image_gen = train_image_datagen.flow_from_directory(train_img_folder,
                                                              target_size=in_img_shape,
                                                              batch_size=batch_size,
                                                              class_mode=None,
                                                              shuffle=True,
                                                              seed=SEED)

    train_mask_gen = train_mask_datagen.flow_from_directory(train_mask_folder,
                                                            target_size=out_img_shape,
                                                            batch_size=batch_size,
                                                            # color_mode='grayscale',
                                                            class_mode=None,
                                                            shuffle=True,
                                                            seed=SEED)

    test_image_datagen = ImageDataGenerator(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1. / 255,
        data_format='channels_last',
        preprocessing_function=crop_center
    )
    test_image_gen = test_image_datagen.flow_from_directory(test_img_folder,
                                                            target_size=in_img_shape,
                                                            batch_size=batch_size,
                                                            class_mode=None,
                                                            shuffle=False)

    # test_image_datagen.mean = train_image_datagen.mean
    # test_image_datagen.std = train_image_datagen.mean

    model = get_model(train_image_gen.image_shape, num_classes=num_classes)
    model.compile(optimizer=Adam(lr=(learn_rate)), loss='mse')

    if pretrained_weights != '':
        # for layer in model.layers[-2:]:
        #     layer.name += '_lux'
        model.load_weights(pretrained_weights, by_name=True)

    callback_list = []
    if out_weights_file != '':
        model_checkpoint = ModelCheckpoint(out_weights_file, period=1, save_weights_only=True)
        callback_list.append(model_checkpoint)

    while current_epoch <= num_epochs:
        print('\n############# Epoch %i / %i #############' % ((current_epoch), num_epochs))
        steps_train_epoch = train_image_gen.n // batch_size
        for i in range(steps_train_epoch):
            train_img_batch = train_image_gen[i]
            train_mask_batch = train_mask_gen[i]
            # train_mask_batch = to_categorical(train_mask_gen[i], num_classes)

            model.fit(train_img_batch, train_mask_batch, batch_size=8, epochs=1,
                      callbacks=callback_list, verbose=1)


        if out_pred_masks_test != '':
            out_pred_epoch_fold = os.path.join(out_pred_masks_test, str(current_epoch))
            final_prediction(out_pred_epoch_fold, test_image_gen, model)
            model.save_weights(os.path.join(out_pred_epoch_fold, 'model.h5'))

        current_epoch += 1


if __name__ == '__main__':
    main()
