import tensorflow as tf
from utilz import *
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据，这里加载的数据，如果用苹果电脑跑不出来的话，需要借助window10系统来跑，跑完了拷贝过来使用
acoustic = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')

x = tf.keras.layers.Input((128,512))

# 这个模型在双向LSTM层前增加了一个一维卷积层（Conv1D），形成了CNN-BLSTM的结构。
# 这一结构可以在序列数据中捕获更丰富的局部特征，并将这些特征输入到BLSTM层进行序列建模。
# 增强了模型在处理声学信号这类时序数据时的特征提取能力。
h = tf.keras.layers.Conv1D(64, 3, 2)(x)
h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, dropout=0.3))(h)
res = tf.keras.layers.Dense(3, 'softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics='acc')
model.summary()

callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_CNN-BLSTM_wav2vec.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]
model.fit(x=np.asarray(acoustic['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
            validation_data=[np.asarray(acoustic['valid']), np.asarray(label['valid'])],
            callbacks=callback_list)

model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_CNN-BLSTM_wav2vec.tf')
pred = model.predict(np.asarray(acoustic['test']))
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])

eval_res = classification_report(numeric_test_labels, predicted_test_labels,
                                 target_names=['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)

print(eval_res)
cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)
