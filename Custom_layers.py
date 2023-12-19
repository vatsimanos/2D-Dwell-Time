import tensorflow as tf



def inception_A_res(x, bn = False):

    #1
    x1 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), padding="same")(x)
    if bn : x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)



    #2
    x2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), padding="same")(x)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=8, kernel_size=(3,3), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)



    #3
    x3 = tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), padding="same")(x)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    x3 = tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same")(x3)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    x3 = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),  padding="same")(x3)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)



    #concat
    xc = tf.keras.layers.Concatenate(axis=3)([x1,x2,x3])

    xc = tf.keras.layers.Conv2D(filters=96, kernel_size=(1,1), padding="same")(xc)
    if bn : xc = tf.keras.layers.BatchNormalization()(xc)



    #add
    x = tf.keras.layers.Add()([x, xc])
    if bn : xc = tf.keras.layers.BatchNormalization()(xc)
    x = tf.keras.layers.ReLU()(x)

    return x



def inception_B_res(x,bn = False):

    #1
    x1 = tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), padding="same")(x)
    if bn : x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)



    #2
    x2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), padding="same")(x)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=40, kernel_size=(1, 7), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(7,1), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)



    #concat
    xc = tf.keras.layers.Concatenate(axis=3)([x1, x2])

    xc = tf.keras.layers.Conv2D(filters=288, kernel_size=(1, 1))(xc)
    if bn : xc = tf.keras.layers.BatchNormalization()(xc)

    #add
    x = tf.keras.layers.Add()([x, xc])
    if bn : x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x



def inception_C_res(x,filter_reduction = 1, bn = False):

    #1
    x1 = tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), padding="same")(x)
    if bn : x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)

    #2
    x2 = tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), padding="same")(x)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=56, kernel_size=(1, 3), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,1), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)



    #concat
    xc = tf.keras.layers.Concatenate(axis=3)([x1, x2])

    xc = tf.keras.layers.Conv2D(filters=536, kernel_size=(1, 1), padding="same")(xc)
    if bn : xc = tf.keras.layers.BatchNormalization()(xc)



    #add
    x = tf.keras.layers.Add()([x, xc])
    if bn : x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    return x



def reduction_A_res(x, bn = False):

    #1
    x1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)



    #2
    x2 = tf.keras.layers.Conv2D(filters=96, strides=(2,2), kernel_size=(3, 3), )(x)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)



    #3
    x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding="same")(x)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),  padding="same")(x3)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    x3 = tf.keras.layers.Conv2D(filters=96, strides =(2,2), kernel_size=(3, 3))(x3)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    #concat
    x = tf.keras.layers.Concatenate(axis=3)([x1, x2, x3])

    return x



def channel_increase(x, bn = False):

    #1
    x1 = tf.keras.layers.Conv2D(filters=288, kernel_size=(1,1),padding="same")(x)
    if bn : x1 = tf.keras.layers.BatchNormalization()(x1)
    x1 = tf.keras.layers.ReLU()(x1)



    #2
    x2 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding="same")(x)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)

    x2 = tf.keras.layers.Conv2D(filters=96, kernel_size=(3, 3), padding="same")(x2)
    if bn : x2 = tf.keras.layers.BatchNormalization()(x2)
    x2 = tf.keras.layers.ReLU()(x2)



    #3
    x3 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1),  padding="same")(x)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)

    x3 = tf.keras.layers.Conv2D(filters=72,kernel_size=(3, 3), padding="same")(x3)
    if bn : x3 = tf.keras.layers.BatchNormalization()(x3)
    x3 = tf.keras.layers.ReLU()(x3)



    #4
    x4 = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,1), padding="same")(x)
    if bn : x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)

    x4 = tf.keras.layers.Conv2D(filters=72, kernel_size=(3,3), padding="same")(x4)
    if bn : x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)

    x4 = tf.keras.layers.Conv2D(filters=80,kernel_size=(3, 3), padding="same")(x4)
    if bn : x4 = tf.keras.layers.BatchNormalization()(x4)
    x4 = tf.keras.layers.ReLU()(x4)



    #concat
    x = tf.keras.layers.Concatenate(axis=3)([x1,x2,x3,x4])

    return x