import tensorflow as tf
import numpy as np
from utilz import *
from sklearn.metrics import classification_report, confusion_matrix


# 加载数据，这里加载的数据，如果用苹果电脑跑不出来的话，需要借助window10系统来跑，跑完了拷贝过来使用
acoustic = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/acoustic_wav2vec.pkl')
label = load_features('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/data/labels.pkl')

# 使用了自定义的Attention_Self层来实现自注意力机制
# 在输入序列上进行自注意力计算，每个时间步的输出都是基于整个输入序列的加权和。
class Attention_Self(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(Attention_Self, self).__init__(**kwargs)
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.S = tf.keras.layers.Dense(1)
        self.units = units

    def call(self, features):
        features_ = tf.expand_dims(features, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(features_))
        attention_weights = tf.nn.softmax(self.S(score), axis=1)
        ATTN = attention_weights * (features)
        ATTN = tf.reduce_sum(ATTN, axis=1)

        return ATTN

    def get_config(self):
        config = super(Attention_Self, self).get_config()
        config.update({"units": self.units})
        return config

# 模型3 (Att-BLSTM)：
# 输入层
# 一维卷积层 (Conv1D)
# 自定义注意力层 (Self Attention)  是与模型1和模型2最主要的区别。
# 自注意力机制可以帮助模型在序列数据中识别和利用不同时间步之间的依赖关系，对于处理自然语言和声学信号等序列数据来说，这是一种非常有用的机制。
# 定义了一个名为Attention_Self的自定义层，该层实现了自注意力机制。在模型的构建过程中，该自注意力层被插入到了一维卷积层和双向LSTM层之间，用于对卷积层输出的特征进行进一步的处理和加权。
# 三个模型都是针对时序数据设计的，但它们使用了不同的层和结构来提取特征和建模时序依赖性，其中模型3引入了自注意力机制，可能会有更好的性能
# 双向LSTM层
# Dense输出层
x = tf.keras.layers.Input((128,512))
h = tf.keras.layers.Conv1D(64, 3, 2)(x)
h = Attention_Self(32)(h)
h = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(h)
res = tf.keras.layers.Dense(3, 'softmax')(h)

model = tf.keras.Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()

callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_Att-BLSTM_wav2vec.tf',
                                                    monitor='val_loss', save_best_only=True, save_freq='epoch')]
model.fit(x=np.asarray(acoustic['train']), y=np.asarray(label['train']), batch_size=16, epochs=30,
            validation_data=[np.asarray(acoustic['valid']), np.asarray(label['valid'])],
            callbacks=callback_list)

model = tf.keras.models.load_model('/Users/wangdong/WorkSpace/MSA Datasets/SIMS/res/A_model_Att-BLSTM_wav2vec.tf')
pred = model.predict(np.asarray(acoustic['test']))
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])

eval_res = classification_report(numeric_test_labels, predicted_test_labels,
                                    target_names = ['Neg', 'Pos', 'Neu'],
                                    digits=4, output_dict=False)

print(eval_res)
cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)