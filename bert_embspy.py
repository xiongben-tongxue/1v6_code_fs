import tensorflow as tf
import numpy as np
from utilz import *
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


#加载数据
w2v_embs = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_wav2vec.pkl')
bert_embs = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/textual_bert.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')


# 应该将Input层的参数更改为适当的3维格式
x = tf.keras.layers.Input(shape=(36, 768))
h = tf.keras.layers.LSTM(64, return_sequences=False, return_state=False)(x)
res = tf.keras.layers.Dense(3, 'softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()

# 进行训练
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/T_model_LSTM_BERT.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]
model.fit(x=np.asarray(bert_embs['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
          validation_data=[np.asarray(bert_embs['valid']), np.asarray(label['valid'])],
          callbacks=callback_list)
# 查看模型的训练笑果
model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res_tmp/T_model_LSTM_word2vec.tf')
pred = model.predict(np.asarray(bert_embs['test']))
predicted_test_labels = pred.argmax(axis=1)
nuneric_test_labels = np.array(label['test'])

eval_res = classification_report(nuneric_test_labels, predicted_test_labels,
                                 target_names=['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)
print(eval_res)
cm = confusion_matrix(y_true=nuneric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)