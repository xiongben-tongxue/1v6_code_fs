import tensorflow as tf
from utilz import *
import numpy as np

acoustic = load_features('C:/Test/MSA Datasets/data/acoustic_wav2vec.pkl')
label = load_features('C:/Test/MSA Datasets/data/labels.pkl')

shapeReturn = np.shape(acoustic['train'])
print(shapeReturn)

x = tf.keras.layers.Input((128, 512))
h = tf.keras.layers.LSTM(32, dropout=0.5)(x)
res = tf.keras.layers.Dense(3, 'softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy)
model.summary()