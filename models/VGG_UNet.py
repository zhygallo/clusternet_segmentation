"""Keras implementation of UNet model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D, Concatenate, Cropping2D, BatchNormalization, Activation, MaxPooling2D
from keras.applications.vgg16 import VGG16
from keras import backend as K

K.set_image_data_format('channels_last')


def get_model(img_shape, num_classes=2):
    class_model = VGG16(weights='imagenet', include_top=False, input_shape=img_shape)

    class_model.get_layer('block1_conv1').trainable = False
    class_model.get_layer('block1_conv2').trainable = False
    class_model.get_layer('block2_conv1').trainable = False
    class_model.get_layer('block2_conv2').trainable = False

    class_model = Model(inputs=class_model.inputs, outputs=class_model.get_layer('block3_pool').output)

    skip1 = class_model.get_layer('block3_conv3').output
    skip2 = class_model.get_layer('block2_conv2').output
    skip3 = class_model.get_layer('block1_conv2').output

    conv4_1 = Conv2D(512, 3, padding='valid')(class_model.output)
    batch4_1 = BatchNormalization()(conv4_1)
    act4_1 = Activation('relu')(batch4_1)
    conv4_2 = Conv2D(512, 3, padding='valid')(act4_1)
    batch4_2 = BatchNormalization()(conv4_2)
    act4_2 = Activation('relu')(batch4_2)

    # up5 = UpSampling2D(size=(2, 2))(act4_2)
    up5 = Conv2D(256, 2, activation='relu', padding='valid')(
        UpSampling2D(size=(2, 2))(act4_2))
    height_dif = int((skip1.shape[1] - up5.shape[1])) / 2
    width_dif = int((skip1.shape[2] - up5.shape[2])) / 2
    crop_skip1 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)),
                                      (math.ceil(width_dif), int(width_dif))))(skip1)
    concat5 = Concatenate()([crop_skip1, up5])
    conv5_1 = Conv2D(256, 3, padding='valid')(concat5)
    batch5_1 = BatchNormalization()(conv5_1)
    act5_1 = Activation('relu')(batch5_1)
    conv5_2 = Conv2D(256, 3, padding='valid')(act5_1)
    batch5_2 = BatchNormalization()(conv5_2)
    act5_2 = Activation('relu')(batch5_2)

    # up6 = UpSampling2D(size=(2, 2))(act5_2)
    up6 = Conv2D(128, 2, activation='relu', padding='valid')(
        UpSampling2D(size=(2, 2))(act5_2))
    height_dif = int((skip2.shape[1] - up6.shape[1])) / 2
    width_dif = int((skip2.shape[2] - up6.shape[2])) / 2
    crop_skip2 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)),
                                      (math.ceil(width_dif), int(width_dif))))(skip2)
    concat6 = Concatenate()([crop_skip2, up6])
    conv6_1 = Conv2D(128, 3, padding='valid')(concat6)
    batch6_1 = BatchNormalization()(conv6_1)
    act6_1 = Activation('relu')(batch6_1)
    conv6_2 = Conv2D(128, 3, padding='valid')(act6_1)
    batch6_2 = BatchNormalization()(conv6_2)
    act6_2 = Activation('relu')(batch6_2)

    up7 = Conv2D(64, 2, activation='relu', padding='valid')(
        UpSampling2D(size=(2, 2))(act6_2))
    height_dif = int((skip3.shape[1] - up7.shape[1])) / 2
    width_dif = int((skip3.shape[2] - up7.shape[2])) / 2
    crop_skip3 = Cropping2D(cropping=((math.ceil(height_dif), int(height_dif)),
                                      (math.ceil(width_dif), int(width_dif))))(skip3)
    concat7 = Concatenate()([crop_skip3, up7])
    conv7_1 = Conv2D(64, 3, padding='valid')(concat7)
    batch7_1 = BatchNormalization()(conv7_1)
    act7_1 = Activation('relu')(batch7_1)
    conv7_2 = Conv2D(64, 3, padding='valid')(act7_1)
    batch7_2 = BatchNormalization()(conv7_2)
    act7_2 = Activation('relu', name='for_clust')(batch7_2)

    conv8 = Conv2D(num_classes, (1, 1), activation='softmax', name='clust_output')(act7_2)

    # conv9 = Conv2D(3, (1, 1), activation='sigmoid', name='rgb_output')(act7_2)

    model = Model(inputs=class_model.inputs, outputs=[conv8])

    return model
