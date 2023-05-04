import tensorflow as tf
from keras import layers, models


def residual_block(x, filters, kernel_size=3, strides=1, use_conv_shortcut=True):
    # Residual block implementation
    shortcut = x
    if use_conv_shortcut:
        shortcut = layers.Conv2D(filters, kernel_size=1, strides=strides, padding='same')(x)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([shortcut, x])
    x = layers.ReLU()(x)

    return x


def resnet101(input_shape, num_classes):
    # ResNet101 model implementation
    inputs = layers.Input(shape=input_shape)

    # Initial convolution and pooling layers
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Residual blocks
    for _ in range(3):
        x = residual_block(x, 64)

    x = residual_block(x, 128, strides=2)
    for _ in range(3):
        x = residual_block(x, 128)

    x = residual_block(x, 256, strides=2)
    for _ in range(22):
        x = residual_block(x, 256)

    x = residual_block(x, 512, strides=2)
    for _ in range(2):
        x = residual_block(x, 512)

    # Final pooling and dense layers
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


input_shape = (224, 224, 3)
num_classes = 6

model = resnet101(input_shape, num_classes)
model.summary()
