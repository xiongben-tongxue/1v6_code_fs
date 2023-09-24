import tensorflow as tf
from utilz import *
import numpy as np

acoustic = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')

shapeReturn = np.shape(acoustic['train'])
print(shapeReturn)

x = tf.keras.layers.Input((128, 512))
h = tf.keras.layers.LSTM(32, dropout=0.5)(x)
res = tf.keras.layers.Dense(3, 'softmax')(h)
model = tf.keras.Model(inputs=x, outputs=res)


model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))
model.summary()
print(model.summary())
output_layer = model.layers[-1]
print(output_layer.activation)



callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/model.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]
model.fit(x=np.asarray(acoustic['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
          validation_data=[np.asarray(acoustic['valid']), np.asarray(label['valid'])],
          callbacks=callback_list)