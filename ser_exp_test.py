# 导入库
import tensorflow as tf
from utilz import *
import numpy as np

# 使用 TensorFlow 和 Keras 构建、编译和训练一个简单的循环神经网络模型的例子

# 加载数据，使用utilz模块中的load_features函数加载音频特征数据和对应的标签
acoustic = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')

# 获取数据形状，获取训练数据并打印
shapeReturn = np.shape(acoustic['train'])
print(shapeReturn)

# 构建模型
# 定义了一个简单的循环神经网络模型，该模型由一个输入层、一个 LSTM 层和一个全连接层组成。
x = tf.keras.layers.Input((128, 512))
h = tf.keras.layers.LSTM(32, dropout=0.5)(x)
res = tf.keras.layers.Dense(3, 'softmax')(h)
model = tf.keras.Model(inputs=x, outputs=res)

# 编译模型 使用 Adam 优化器和稀疏分类交叉熵损失函数编译模型。
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False))

# 打印模型概述
model.summary()
print(model.summary())

# 打印输出层激活函数
output_layer = model.layers[-1]
print(output_layer.activation)

# 定义回调函数,定义了一个回调函数列表，用于在每个 epoch 结束时保存验证损失最低的模型。
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/model.tf',
                                                    monitor='val_loss',
                                                    save_best_only=True,
                                                    save_freq='epoch')]
# 使用加载的训练数据训练模型，设置了批大小为 16，总共训练 30 个 epoch，并使用验证数据进行验证。
model.fit(x=np.asarray(acoustic['train']),
          y=np.asarray(label['train']),
          batch_size=16,
          epochs=30,
          validation_data=[np.asarray(acoustic['valid']), np.asarray(label['valid'])],
          callbacks=callback_list)