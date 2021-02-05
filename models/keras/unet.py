# -*- coding: utf-8 -*-

from keras import backend as K
from keras.layers import Input, Dropout, Conv2D
from keras.layers import MaxPooling2D, concatenate, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model  # , load_model
from keras.layers.core import Activation  # , Reshape, Permute


def dice_coef(y_true, y_pred):
    '''Cost function'''
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    print(y_true_f.shape)
    smooth = 1
    # intersection = K.sum(y_true_f * y_pred_f)
    intersection = K.abs(K.abs(y_true_f - y_pred_f)-1)
    sh = K.cast(K.shape(y_true_f), dtype='float32')
    sh = K.print_tensor(sh, message='sh = ')
    return (intersection + smooth) / (sh + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# def jaccard_distance_loss(weights):
def jaccard_distance_loss(y_true, y_pred, weights, smooth=100):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
    = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets.
    This has been shifted so it converges on 0 and is smoothed to
    avoid exploding or disapearing gradient.

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    w = K.flatten(weights)
    intersection = K.sum(K.abs(w * y_true * y_pred))
    sum_ = K.sum(K.abs(w * y_true) + K.abs(w * y_pred))
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    # jac = K.print_tensor(jac, message='jac = ')
    return (1 - jac) * smooth


#  https://stackoverflow.com/questions/50124158/keras-loss-function-with-additional-dynamic-parameter/50127646#50127646

# Model
def Umodel(freqs, width=64, height=64, nbClass=1, kernel=3):
    '''U-net implementation'''
    # Input layer
    img_input = Input(shape=(freqs, height, width), name='img_input')
    y_true = Input(shape=(1, height, width), name='y_true')
    weights = Input(shape=(1, height, width), name='weight')

    # Encoder layer 1
    #  Kaiming: Best practive for Relu
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(img_input)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization(axis=1)(conv1)
    conv1 = Dropout(0.25)(conv1)  # Fjern dropout?
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    # Encoder layer 2
    conv2 = Conv2D(64, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(pool1)  # Gå opp i features.
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Conv2D(64, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization(axis=1)(conv2)
    conv2 = Dropout(0.25)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Encoder layer 3 # Dropp eit lag? Prøve med 4 input data.
    conv3 = Conv2D(128, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Dropout(0.25)(conv3)
    conv3 = Conv2D(128, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(conv3)
    # Bruka batch norm 2d?Dersom det finnes
    conv3 = BatchNormalization(axis=1)(conv3)
    conv3 = Dropout(0.25)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    # Center layer
    convC = Conv2D(256, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(pool3)
    convC = BatchNormalization(axis=1)(convC)
    convC = Conv2D(256, (kernel, kernel), padding="same", activation='relu',
                   kernel_initializer='he_normal')(convC)
    convC = BatchNormalization(axis=1)(convC)
    convC = Dropout(0.25)(convC)

    # Decoder layer 3
    up3 = concatenate([Conv2DTranspose(128, (kernel, kernel), strides=(2, 2),
                                       padding='same', activation='relu',
                                       kernel_initializer='he_normal')(convC),
                       conv3],
                      axis=1)
    decod3 = BatchNormalization(axis=1)(up3)
    decod3 = Conv2D(128, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod3)
    decod3 = BatchNormalization(axis=1)(decod3)
    decod3 = Conv2D(128, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod3)
    decod3 = BatchNormalization(axis=1)(decod3)
    decod3 = Dropout(0.25)(decod3)

    # Decoder layer 2
    up2 = concatenate([Conv2DTranspose(64, (kernel, kernel), strides=(2, 2),
                                       padding='same', activation='relu',
                                       kernel_initializer='he_normal')(decod3),
                       conv2], axis=1)
    decod2 = BatchNormalization(axis=1)(up2)
    decod2 = Conv2D(64, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod2)
    decod2 = BatchNormalization(axis=1)(decod2)
    decod2 = Conv2D(64, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod2)
    decod2 = BatchNormalization(axis=1)(decod2)
    decod2 = Dropout(0.25)(decod2)
    # Decoder layer 1
    up1 = concatenate([Conv2DTranspose(32, (kernel, kernel), strides=(2, 2),
                                       padding='same', activation='relu',
                                       kernel_initializer='he_normal')(decod2),
                       conv1], axis=1)
    decod1 = BatchNormalization(axis=1)(up1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Conv2D(32, (kernel, kernel), padding="same", activation='relu',
                    kernel_initializer='he_normal')(decod1)
    decod1 = BatchNormalization(axis=1)(decod1)
    decod1 = Dropout(0.25)(decod1)

    # Segmentation Layer
    x = Conv2D(nbClass, (1, 1), padding="valid")(decod1)
    # x = Reshape((nbClass, height * width))(x)
    # x = Permute((2, 1))(x)
    y_pred = Activation("softmax")(x)

    # Define model
    model = Model(inputs=[img_input, y_true, weights], outputs=y_pred)

    # Compiling the model
    # model2.compile(optimizer='adam', loss=dice_coef_loss,
    #                metrics=['categorical_accuracy', mean_dice_coef])
    # model.compile(optimizer='adam', loss=K.categorical_crossentropy,
    #              metrics=[jaccard_distance_loss])
    model.add_loss(jaccard_distance_loss(y_true, y_pred, weights))
    model.compile(optimizer='adam', loss=None, metrics=['loss'])

    # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # Endra læringsrate.

    predict_model = Model(inputs=img_input, outputs=y_pred,
                          name='prediction_only')
    return model, predict_model
