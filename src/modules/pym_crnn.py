import string
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, LeakyReLU, Dropout
from tensorflow.keras.layers import GRU, Bidirectional, Dense, Lambda
    
def ctc_loss(labels, logits):
    """ Loss function used when training the CRNN model
    """
    return tf.reduce_mean(tf.nn.ctc_loss(
        labels = labels,
        logits = logits,
        logit_length = [logits.shape[1]]*logits.shape[0],
        label_length = None,
        logits_time_major = False,
        blank_index=-1))
    
def build_crnn(input_shape, charList = list(string.printable)):
    """ Build and returns a CRNN model
    """
    model = tf.keras.Sequential()
    
    # CNN
    # Layer 1
    model.add(Conv2D(filters=32, kernel_size=(5,5), padding='SAME', input_shape = input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 2
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    # Layer 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    # Layer 4
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))

    # Layer 5
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding='SAME'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D(pool_size=(2,1), strides=(2,1)))
    model.add(Dropout(0.4))

    model.add(Lambda(lambda x :tf.squeeze(x, axis=1)))

    numHidden = 256
    # Bidirectionnal RNN
    model.add(Bidirectional(GRU(numHidden, return_sequences=True)))
    # Classification of characters
    model.add(Dense(len(charList)+1))

    return model       