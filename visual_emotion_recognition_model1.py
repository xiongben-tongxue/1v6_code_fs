# 导入所需的库和模块
# 包括TensorFlow（用于深度学习）、numpy（用于数值计算）、logging（用于日志记录）以及scikit-learn的评估工具。
import tensorflow as tf
from utilz import *
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix

# 设置日志记录
# 设置日志记录的基本配置
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# 加载数据
visual_ori = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/visual_mpori.pkl')
#  下面这两个是没有跑通的
# visual_of = load_features('./data/visual_OFfts.pkl')
# visual_clip = load_features('./data/visual_clip.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')
print(np.shape(visual_ori['train']))

logging.info(f"start summary data ")
# Model 1
# 构建模型
# 以下代码定义了一个3D卷积神经网络模型，包含3个卷积层，一个平坦化层和一个密集连接层。
x = tf.keras.layers.Input((10, 64, 64, 1))
h = tf.keras.layers.Conv3D(16, [3,3,3], [2,2,2], 'same')(x)
h = tf.keras.layers.Conv3D(32, [1,1,1], [2,2,2], 'same')(h)
h = tf.keras.layers.Conv3D(8, [3,3,3], [2,2,2], 'same')(h)

h = tf.keras.layers.Flatten()(h)
res = tf.keras.layers.Dense(3, 'softmax')(h)
model = tf.keras.Model(inputs=x, outputs=res)

# 编译模型：定义了模型后，我们需要编译它，设置优化器、损失函数和评估指标。
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()
logging.info(f"model.summary Loaded data for {model.summary()} ")

logging.info(f"callback_list start")
# 定义回调函数
# 此代码定义了一个回调函数列表，该列表中只包含一个回调：ModelCheckpoint，用于在每个时期结束时保存表现最好的模型。
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/V_model_CNN_mpori.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]

# 训练模型：使用model.fit()函数训练模型，并使用提供的训练和验证数据。
model.fit(x=np.expand_dims(visual_ori['train'], axis=-1), y=np.asarray(label['train']), batch_size=16, epochs=30,
            validation_data=[np.expand_dims(visual_ori['valid'], axis=-1), np.asarray(label['valid'])],
            callbacks=callback_list)
# 评估模型：加载最佳模型，对测试数据进行预测，然后使用classification_report和confusion_matrix计算和打印性能指标。
model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/V_model_CNN_mpori.tf/')
pred = model.predict(np.expand_dims(visual_ori['test'], axis=-1))
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])

eval_res = classification_report(numeric_test_labels, predicted_test_labels,
                                 target_names=['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)

print(eval_res)

cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)

# 对于关于OpenFace模式的数据集，暂时无法进行训练
# model = tf.keras.models.load_model('./res/V_model_CNN_of.tf/')
# 对于Clip模式的数据集，暂时没有数据进行训练
# callback_list = [ModelCheckpoint(filepath='./res/V_model_CNN_clip.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]

#Model 2
