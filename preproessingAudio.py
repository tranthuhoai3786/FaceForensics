import librosa
import pandas as pd
from scipy.io import wavfile as wav
import numpy as np
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from tensorflow.keras import models
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
from sklearn import metrics


audio_file_path =  './ASVspoof2017_V2_dev/Real/D_1000001.wav'
# librosa_audio_data, librosa_sample_rate = librosa.load(audio_file_path)
# print(librosa_audio_data)
#
# wav_sample_rate, wave_audio = wav.read(audio_file_path)
# print(wave_audio)
#
# #trich xuat dac trung
# mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr =  librosa_sample_rate)
# print(mfccs.shape) #(20,185)
#
# print(mfccs)

def feature_extractor(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis =0)
    return mfccs_scaled_features

# print(feature_extractor(audio_file_path))


dirData = './ASVspoof2017_V2_dev/Real'
extracted_feature = []


for filename in os.listdir(dirData):
    file_path = os.path.join(dirData, filename)#./ASVspoof2017_V2_dev/D_1000001.wav
    data = feature_extractor(file_path)
    extracted_feature.append([data,0])


dirData = './ASVspoof2017_V2_dev/Fake'
for filename in os.listdir(dirData):
    file_path = os.path.join(dirData, filename)#./ASVspoof2017_V2_dev/D_1000001.wav
    data = feature_extractor(file_path)
    extracted_feature.append([data,1])

extracted_feature_df = pd.DataFrame(extracted_feature, columns=['feature', 'class'])
print(extracted_feature_df.head())


# for index_num, row
X= np.array(extracted_feature_df['feature'].tolist())
y = np.array(extracted_feature_df['class'].tolist())
y = np.array(pd.get_dummies(y))
print(X.shape) #(1710,40)
print(y.shape)#(1710,2)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=0)

print(X_train.shape)#(1368,40)
print(X_test.shape)#(342,40)
print(y_train.shape)#(1368,2)
print(y_test.shape)#(342,2)

print(X_train)
print(y_train)

num_labels = y.shape[1]

# Dense()

'''
model
'''
model  = Sequential()

model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(200,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'],
              optimizer ='adam')

num_epochs =20
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='save_model/audio_classification.h5',verbose=1, save_best_only = True)
start = datetime.now()
model.fit(X_train, y_train, batch_size = num_batch_size, epochs = num_epochs,validation_data = (X_test, y_test),callbacks = [checkpointer] )


duration = datetime.now() - start
print("hoafn thanfh: ", duration)


'''
predict
'''
model = load_model('save_model/audio_classification.h5')
filename = './ASVspoof2017_V2_dev/Real/D_1000001.wav'
predict_feature = feature_extractor(filename)
predict_feature = predict_feature.reshape(1,-1)
model.predict_classes(predict_feature)





