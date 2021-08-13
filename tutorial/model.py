from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, add, Input, MaxPool2D, \
    GlobalAvgPool2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K


def conv2d_bn(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):

    x = Conv2D(nb_filter, kernel_size=kernel_size, strides=strides, padding=padding,
               kernel_regularizer=regularizers.l2(0.00012))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def conv2d_bn1(x, nb_filter, kernel_size, strides=(1, 1), padding='same'):
    x = Conv2D(nb_filter, kernel_size=kernel_size,
               strides=strides,
               padding=padding,
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = BatchNormalization()(x)
    return x


def shortcut(input, residual):
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    # 如果維度不同，則使用1x1卷積進行調整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_regularizer=regularizers.l2(0.0001))(input)

    return add([identity, residual])


def shortcut1(input, residual,nb_filter):

    input = conv2d_bn1(input, nb_filter, kernel_size=(3, 3), strides=(2,2))
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_height = int(round(input_shape[1] / residual_shape[1]))
    stride_width = int(round(input_shape[2] / residual_shape[2]))
    equal_channels = input_shape[3] == residual_shape[3]

    identity = input
    # 如果維度不同，則使用1x1卷積進行調整
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        identity = Conv2D(filters=residual_shape[3], kernel_size=(1, 1), strides=(stride_width, stride_height),
                          padding="valid", kernel_regularizer=regularizers.l2(0.0001))(input)

    return add([identity, residual])


def basic_block(nb_filter, strides=(1, 1)):

    def f(input):

        conv1 = conv2d_bn(input, nb_filter, kernel_size=(3, 3), strides=strides)
        residual = conv2d_bn1(conv1, nb_filter, kernel_size=(3, 3))

        return shortcut(input, residual)

    return f


def basic_block1(nb_filter, strides=(1, 1)):

    def f(input):

        conv1 = conv2d_bn1(input, nb_filter, kernel_size=(3, 3), strides=strides)

        return shortcut(input, conv1)

    return f


def resnet_10(input_shape=(64, 64, 3), nclass=1):
    input_ = Input(shape=input_shape)

    conv1 = conv2d_bn(input_, 64, kernel_size=(3, 3), strides=(2, 2))
    pool1 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(conv1)

    conv3 = basic_block(64, strides=(1, 1))(pool1)
    conv3 = Activation('relu')(conv3)

    conv5 = basic_block1(128, strides=(2, 2))(conv3)
    conv5 = Activation('relu')(conv5)

    conv7 = basic_block1(256, strides=(2, 2))(conv5)
    conv7 = Activation('relu')(conv7)

    conv9 = basic_block1(512, strides=(2, 2))(conv7)
    conv9 = Activation('relu')(conv9)

    pool2 = GlobalAvgPool2D()(conv9)
    x = Dropout(0.4)(pool2)
    output_ = Dense(nclass, activation='linear')(x)

    model = Model(inputs=[input_], outputs=output_)
    model.summary()

    return model