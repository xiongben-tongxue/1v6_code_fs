import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import Model, layers
#from tensorflow.keras.layers import Input, Bidirectional, Dense, Conv1D, Conv3D, LSTM, Flatten, Attention, MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Add, Dropout, Softmax
#from tensorflow.keras.regularizers import l2
from utilz import *
#from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# 输出的结果是0，因为mac电脑没有GPU
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#多模态的融合
# 处理视觉
visual_clip = load_features('./data/visual_clip.pkl')
# 处理语音
acoustic = load_features('./data/acoustic_wav2vec.pkl')
# 处理文字
bert_embs = load_features('./data/textual_bert.pkl')
label = load_features('./data/labels.pkl')


class Attention_Self(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Attention_Self, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.S = tf.keras.layers.Dense(1)
        self.units = units

    def call(self, features):
        features_ = tf.expand_dims(features, 1)
        v = self.W1(features)
        q = self.W2(features_)
        score = tf.nn.tanh(q + v)
        attention_weights = tf.nn.softmax(self.S(score), axis=1)
        ATTN = attention_weights * (v)
        ATTN = tf.reduce_sum(ATTN, axis=1)

        return ATTN

    def get_config(self):
        config = super(Attention_Self, self).get_config()
        config.update({"units": self.units})
        return config


# './res/V_model_CNNLSTM_clip.tf'
vis_ipt = tf.keras.layers.Input((10, 512))
vis_h = tf.keras.layers.Conv1D(64, 3, 1, 'same')(vis_ipt)
vis_h = tf.keras.layers.Conv1D(64, 1, 1, 'same')(vis_h)
vis_h = tf.keras.layers.Conv1D(64, 3, 1, 'same')(vis_h)
vis_h = tf.keras.layers.LSTM(64, activation='relu')(vis_h)

# './res/A_model_Att-BLSTM_wav2vec_v2.tf'
aud_ipt = tf.keras.layers.Input((128,512))
aud_h = tf.keras.layers.Conv1D(64, 3, 2)(aud_ipt)
aud_h = Attention_Self(32)(aud_h)
aud_h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(aud_h)

# './res/T_model_AttCNN_bert.tf'
tex_ipt = tf.keras.layers.Input((36,768))
tex_q = tf.keras.layers.Conv1D(64, 3, 1)(tex_ipt)
tex_v = tf.keras.layers.Conv1D(64, 3, 1)(tex_ipt)
tex_qv_attention = tf.keras.layers.Attention()([tex_q, tex_v])
tex_q = tf.keras.layers.GlobalAveragePooling1D()(tex_q)
tex_qv_attention = tf.keras.layers.GlobalAveragePooling1D()(tex_qv_attention)
# h = Concatenate()([q, qv_attention])

h = tf.keras.layers.Concatenate()([vis_h, aud_h, tex_q, tex_qv_attention])
h = tf.keras.layers.Dense(64)(h)
res = tf.keras.layers.Dense(3, activation='softmax')(h)

model = tf.keras.Model(inputs=[vis_ipt, aud_ipt, tex_ipt], outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()

callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='./res/multi_model_concate.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]

model.fit(x=[np.asarray(visual_clip['train']), np.asarray(acoustic['train']), np.asarray(bert_embs['train'])], y=np.asarray(label['train']), batch_size=16, epochs=30,
            validation_data=[[np.asarray(visual_clip['valid']), np.asarray(acoustic['valid']), np.asarray(bert_embs['valid'])], np.asarray(label['valid'])],
            callbacks=callback_list)

model = tf.keras.models.load_model('./res/multi_model_concate.tf/')
pred = model.predict([np.asarray(visual_clip['test']), np.asarray(acoustic['test']), np.asarray(bert_embs['test'])])
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])

eval_res = classification_report(numeric_test_labels, predicted_test_labels,
                                 target_names=['Neg', 'Pos', 'Neu'],
                                 digits=4, output_dict=False)

print(eval_res)

cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)