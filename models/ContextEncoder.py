"""Keras implementation of UNet model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, Concatenate, Cropping2D, BatchNormalization, Activation, MaxPooling2D
from keras import backend as K

K.set_image_data_format('channels_last')


def get_model(img_shape, num_classes=2):
    inputs = Input(shape=img_shape)

    conv1_1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(inputs)
    conv1_2 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv1_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1_2)

    conv2_1 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool1)
    conv2_2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv2_1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_2)

    conv3_1 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool2)
    conv3_2 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_1 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool3)
    conv4_2 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_1 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(pool4)
    conv5_2 = Conv2D(1024, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv5_1)

    up6 = Conv2D(512, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5_2))
    height_dif = int((conv4_2.shape[1] - up6.shape[1])) / 2
    width_dif = int((conv4_2.shape[2] - up6.shape[2])) / 2
    crop_conv4_2 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)), (math.ceil(width_dif), int(width_dif))))(conv4_2)
    concat6 = Concatenate()([crop_conv4_2, up6])
    conv6_1 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(concat6)
    conv6_2 = Conv2D(512, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv6_1)

    up7 = Conv2D(256, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6_2))
    height_dif = int((conv3_2.shape[1] - up7.shape[1])) / 2
    width_dif = int((conv3_2.shape[2] - up7.shape[2])) / 2
    crop_conv3_2 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)), (math.ceil(width_dif), int(width_dif))))(conv3_2)
    concat7 = Concatenate()([crop_conv3_2, up7])
    conv7_1 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(concat7)
    conv7_2 = Conv2D(256, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv7_1)

    up8 = Conv2D(128, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7_2))
    height_dif = int((conv2_2.shape[1] - up8.shape[1])) / 2
    width_dif = int((conv2_2.shape[2] - up8.shape[2])) / 2
    crop_conv2_2 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)), (math.ceil(width_dif), int(width_dif))))(conv2_2)
    concat8 = Concatenate()([crop_conv2_2, up8])
    conv8_1 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(concat8)
    conv8_2 = Conv2D(128, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(conv8_1)

    up9 = Conv2D(64, 2, activation='relu', padding='valid', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8_2))
    height_dif = int((conv1_2.shape[1] - up9.shape[1])) / 2
    width_dif = int((conv1_2.shape[2] - up9.shape[2])) / 2
    crop_conv1_2 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)), (math.ceil(width_dif), int(width_dif))))(conv1_2)
    concat9 = Concatenate()([crop_conv1_2, up9])
    conv9_1 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal')(concat9)
    conv9_2 = Conv2D(64, 3, activation='relu', padding='valid', kernel_initializer='he_normal', name='for_clust')(conv9_1)

    conv10 = Conv2D(num_classes, (1, 1), activation='softmax', name='clust_output')(conv9_2)

    conv11 = Conv2D(3, (1, 1), activation='sigmoid', name='rgb_output')(conv9_2)

    model = Model(inputs=inputs, outputs=[conv10, conv11])

    return model


def main():
    model = get_model(img_shape=(512, 512, 3))
    return 0


if __name__ == "__main__":
    main()