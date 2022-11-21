import tensorflow as tf
import matplotlib.pyplot as plt 

@tf.function
def preprocessstream(img, imgSize=(32, 128), scale=0.8):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_png(img, channels=3)    
    img = img/255

    # increase dataset size by applying random stretches to the image (img will be different after each epoch)    
    # it will change randomly the ratio width/height
    stretch = scale*(tf.random.uniform([1], 0, 1)[0] - 0.3) # -0.5 .. +0.5
    wStretched = tf.maximum(tf.cast(tf.cast(tf.shape(img)[0],tf.float32) * (1 + stretch),tf.int32), 1) # random width, but at least 1
    img = tf.image.resize(img, (wStretched, tf.shape(img)[1])) # stretch horizontally by factor 0.5 .. 1.5

    # Resize the streched image so that it could fit in an image with the dimension of imgSize variable
    (wt, ht) = imgSize
    w, h = tf.cast(tf.shape(img)[0],tf.float32), tf.cast(tf.shape(img)[1],tf.float32)
    fx = w / wt
    fy = h / ht
    f = tf.maximum(fx, fy)
    newSize = (tf.maximum(tf.minimum(wt, tf.cast(w / f, tf.int32)), 1), tf.maximum(tf.minimum(ht, tf.cast(h / f, tf.int32)), 1)) 
    img = tf.image.resize(img, newSize)

    # put streched image in a random position in the img
    dx = wt - newSize[0]
    dy = ht - newSize[1]
    dx1=tf.cond(dx==0, lambda: 0, lambda: tf.random.uniform([1], 0, dx, tf.int32)[0])
    dy1=tf.cond(dy==0, lambda: 0, lambda: tf.random.uniform([1], 0, dy, tf.int32)[0])           
    img = tf.pad(img[..., 0], [[dx1, dx-dx1], [dy1, dy-dy1]], constant_values=1)

    # transform the gray background to white (all pixels >= 0.8 become 1) 
    img = 1-(1-img)*tf.cast(img < 0.8, tf.float32)
    
    # add one dimension at the end (32, 128) => (32, 128, 1)
    return tf.expand_dims(img, 0)