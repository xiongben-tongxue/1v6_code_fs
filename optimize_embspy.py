import tensorflow as tf
import numpy as np
from utilz import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#加载数据
w2v_embs = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_wav2vec.pkl')
bert_embs = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_bert.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')

# 这是老师给的模型 accuracy是0.5470
# 应该将Input层的参数更改为适当的3维格式
# x = tf.keras.layers.Input(shape=(36, 100))
# query = tf.keras.layers.Conv1D(64,3,1)(x)
# value = tf.keras.layers.Conv1D(64,3,1)(x)
# qv_attention = tf.keras.layers.Attention()([query,value])
# qv_attention = tf.keras.layers.GlobalAveragePooling1D()(qv_attention)
# q = tf.keras.layers.GlobalAveragePooling1D()(query)
# h = tf.keras.layers.Concatenate()([qv_attention,q])
#
# res = tf.keras.layers.Dense(3, 'softmax')(h)
#
# model = tf.keras.Model(inputs=x, outputs=res)
# model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
# model.summary()

# ChatGPT给的模型 accuracy为0.5689，确实有了一定的提升效果
# ... 其他导入和数据加载代码 ...
x = tf.keras.layers.Input(shape=(36, 100))

query = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
query = tf.keras.layers.MaxPooling1D(2)(query)
query = tf.keras.layers.Dropout(0.2)(query)

value = tf.keras.layers.Conv1D(128, 3, activation='relu', padding='same')(x)
value = tf.keras.layers.MaxPooling1D(2)(value)
value = tf.keras.layers.Dropout(0.2)(value)

qv_attention = tf.keras.layers.Attention()([query, value])
qv_attention = tf.keras.layers.GlobalAveragePooling1D()(qv_attention)

q = tf.keras.layers.GlobalAveragePooling1D()(query)

h = tf.keras.layers.Concatenate()([qv_attention, q])
h = tf.keras.layers.Dense(64, activation='relu')(h)
h = tf.keras.layers.Dropout(0.5)(h)

res = tf.keras.layers.Dense(3, activation='softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()

# 进行训练
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/T_model_LSTM_BERT_qv.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]
model.fit(x=np.asarray(w2v_embs['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
          validation_data=[np.asarray(w2v_embs['valid']), np.asarray(label['valid'])],
          callbacks=callback_list)

# 查看模型的训练笑果
model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/T_model_LSTM_BERT_qv.tf')
pred = model.predict(np.asarray(w2v_embs['test']))
predicted_test_labels = pred.argmax(axis=1)
nuneric_test_labels = np.array(label['test'])

eval_res = classification_report(nuneric_test_labels, predicted_test_labels,
                                 target_names=['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)
print(eval_res)
cm = confusion_matrix(y_true=nuneric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)

# 跑出的结果accuracy是0.5514