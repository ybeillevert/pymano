import string
import numpy as np
import tensorflow as tf

def encode_labels(labels, char_list = list(string.printable)):
    table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(char_list, np.arange(len(char_list)), value_dtype=tf.int32),
        default_value = len(char_list),
        name='chard2id')
    return table.lookup(tf.compat.v1.string_split(labels, sep=''))

def decode_codes(codes, char_list = list(string.printable)):
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            np.arange(len(char_list)),
            char_list,
            key_dtype=tf.int32
        ),
        '',
        name='id2char'
    )
    return table.lookup(codes)

def greedy_decoder(logits, char_list = list(string.printable)):
    # ctc beam search decoder
    predicted_codes, _ = tf.nn.ctc_greedy_decoder(
        # shape of tensor [max_time x batch_size x num_classes]
        tf.transpose(logits, (1, 0, 2)),
        [logits.shape[1]]*logits.shape[0]
    )

    # convert to int32
    codes = tf.cast(predicted_codes[0], tf.int32)

    # Decode the index of caracter
    text = decode_codes(codes, char_list)

    # Convert a SparseTensor to string
    text = tf.sparse.to_dense(text).numpy().astype(str)

    return list(map(lambda x: ''.join(x), text))