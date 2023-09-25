import tensorflow as tf
from utilz import *
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

#from tensorflow.keras.callbacks import ModelCheckpoint
#from tf.keras import Model
#from tensorflow.keras.layers import Input, Bidirectional, Dense, Conv1D, LSTM, Flatten, Attention


# 输出结果：Num GPUs Available:  0  TensorFlow仍然会使用CPU来执行计算，只是在某些任务上可能会较慢。
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# 加载数据，这里加载的数据，如果用苹果电脑跑不出来的话，需要借助window10系统来跑，跑完了拷贝过来使用
acoustic = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')


# 模型1 (BLSTM)：
# 输入层
# 双向LSTM层
# Dense输出层
x = tf.keras.layers.Input((128,512))
h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.3))(x)
res = tf.keras.layers.Dense(3, 'softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics='acc')
model.summary()

# save_best_only=False 这里应该要是False不要没有更好的结果，不保存文件
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_BLSTM_wav2vec.tf', monitor='val_loss', save_best_only=False, save_freq='epoch')]
model.fit(x=np.asarray(acoustic['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
            validation_data=(np.asarray(acoustic['valid']), np.asarray(label['valid'])),
            callbacks=callback_list)

model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_BLSTM_wav2vec.tf')
pred = model.predict(np.asarray(acoustic['test']))
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])

unique_labels = np.unique(np.concatenate((numeric_test_labels, predicted_test_labels)))
print("Unique labels in test data:", unique_labels)

eval_res = classification_report(numeric_test_labels, predicted_test_labels,
                                 target_names = ['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)

print(eval_res)

cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)