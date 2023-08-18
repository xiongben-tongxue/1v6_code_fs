import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Bidirectional, Dense, Conv1D, LSTM, Flatten, Attention
from utilz import *
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

acoustic = load_features('./data/acoustic_opensml.pkl')
label = load_features('./data/labels.pkl')

x = Input((256,65))
h = Bidirectional(LSTM(32, dropout=0.3))(x)
res = Dense(3, 'softmax')(h)
    
model = Model(inputs=x, outputs=res)
model.compile(optimizer='Adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics='acc')
model.summary()


callback_list = [ModelCheckpoint(filepath='./res/A_model_BLSTM_sml.tf', monitor='val_loss', save_best_only=True, save_freq='epoch')]

model.fit(x=np.asarray(acoustic['train']), y=np.asarray(label['train']), batch_size=16, epochs=30, 
            validation_data=[np.asarray(acoustic['valid']), np.asarray(label['valid'])],
            callbacks=callback_list)

model = tf.keras.models.load_model('./res/A_model_BLSTM_sml.tf')
pred = model.predict(np.asarray(acoustic['test']))
predicted_test_labels = pred.argmax(axis=1)
numeric_test_labels = np.array(label['test'])
            
eval_res = classification_report(numeric_test_labels, predicted_test_labels, 
                                    target_names = ['Neg', 'Pos', 'Neu'], 
                                    digits=4, output_dict=False)

print(eval_res)

cm = confusion_matrix(y_true=numeric_test_labels.tolist(), y_pred=predicted_test_labels.tolist())
print(cm)