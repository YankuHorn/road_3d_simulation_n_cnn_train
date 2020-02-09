import tensorflow as tf


def my_L2_loss(y_true, y_pred):

    L2 = tf.nn.l2_loss(y_pred - y_true)
    return L2


def my_scene_class_loss(y_true, y_pred):
    if y_true[3] == 1:
        return 0
    else:
        return tf.keras.backend.categorical_crossentropy(y_true[:3], y_pred)