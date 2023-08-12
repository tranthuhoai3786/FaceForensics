import sys

sys.path.append('C://Users//ihado//AppData//Local//Programs//Python//Python310//Lib//site-packages')
import numpy as np
import os
from PIL import Image
import cv2
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
import time

TRAIN_DATA = 'image/train'
TEST_DATA = 'image/test'
# # TRAIN_DATA = 'dataset/Extracted Faces/Extracted Faces'
X_train = []
y_train = []

X_test = []
y_test = []
#
dic ={'Deepfakes': [1,0,0,0,0], 'Face2Face': [0,1,0,0,0], 'FaceSwap':[0,0,1,0,0], 'NeuralTextures': [0,0,0,1,0], 'youtube': [0,0,0,0,1]}
#
#
def getData(dirData, listData):
    for whatelse in os.listdir(dirData):
        whatelse_path = os.path.join(dirData, whatelse)#data/train/Deepfakes
        # print(whatelse_path)
        list_filename_path = []
        for filename in os.listdir(whatelse_path):
            filename_path = os.path.join(whatelse_path, filename)##data/train/Deepfakes/roi_1.jpg
            # print(filename_path)
            label = filename_path.split('\\')[1]
            img = np.array(Image.open(filename_path))
            list_filename_path.append((img,dic[label])) #5 list ma tran image
        listData.extend(list_filename_path)
    return listData

X_train = getData(TRAIN_DATA,X_train) #len =4568
X_test = getData(TEST_DATA,X_test)
# print(len(X_train))
# print((X_test[10]))
'''
model
'''

def Xception(num_classes=5):
    def sepconv_bn_relu(filters, kernel_size, strides=1):
        def _sepconv_bn_relu(inputs):
            x = layers.SeparableConv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            return x
        return _sepconv_bn_relu

    def entry_flow():
        inputs = tensorflow.keras.Input(shape=(299, 299, 3))
        x = layers.Conv2D(32, 3, strides=2, use_bias=False, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(64, 3, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2D(128, 3, strides=2, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        # Additional 1x1 convolution to adjust the number of channels
        x = layers.Conv2D(728, 1, use_bias=False, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        return tensorflow.keras.Model(inputs, x, name='entry_flow')

    def middle_flow(inputs, num_blocks=8):
        def _middle_block(inputs):
            x = inputs
            for _ in range(num_blocks):
                residual = x
                x = sepconv_bn_relu(728, 3)(x)
                x = sepconv_bn_relu(728, 3)(x)
                x = sepconv_bn_relu(728, 3)(x)
                x = layers.Add()([x, residual])
            return x

        x = _middle_block(inputs)
        return x

    def exit_flow(num_classes):
        def _exit_block(inputs):
            residual = layers.Conv2D(1024, 1, strides=2, use_bias=False, padding='same')(inputs)
            residual = layers.BatchNormalization()(residual)

            x = sepconv_bn_relu(728, 3)(inputs)
            x = sepconv_bn_relu(1024, 3)(x)
            x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
            x = layers.Add()([x, residual])

            x = sepconv_bn_relu(1536, 3)(x)
            x = sepconv_bn_relu(2048, 3)(x)

            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(num_classes, activation='softmax')(x)

            return x

        return _exit_block

    inputs = tensorflow.keras.Input(shape=(299, 299, 3))
    x = entry_flow()(inputs)
    x = middle_flow(x)
    outputs = exit_flow(num_classes)(x)

    model = tensorflow.keras.Model(inputs, outputs, name='Xception')
    return model

# Compile the model
model = Xception(num_classes=5)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
# # Fit the model (provide your training data and labels)
# model.fit(...)
#
# # Save the model
# model.save('xception_model.h5')
model.compile(optimizer = 'adam',#ky thuat toi uu hoa adam
                    loss= 'categorical_crossentropy',#entropy: ức độ hỗn loạn không đồng nhất của dữ liệu, crossentropy so sánh sự không đồng đều giữa 2 entropy
                    #cross entropy = entropy + hệ số phân tán KL devergence (p,q) = -sum([p[i]*np.log(p[i]/q[i]) for i in range len(p[i])])
                    metrics = ['accuracy'])
model.fit(np.array([x[0] for _,x in enumerate(X_train)]),np.array([y[1] for _,y in enumerate(X_train)]) , epochs =10, batch_size=16)#chay 20 lan
model.save('xception_model_10epochs.keras')

# Fit mô hình
# model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Lưu mô hình
# model.save('xception_model.h5')





# model_training_first = models.Sequential([
#
# ])
#model_training_first.summary()#ket cau model san sinh ra bao nhieu sieu tham so
# model_training_first.compile(optimizer = 'adam',#ky thuat toi uu hoa adam
#                              loss= 'categorical_crossentropy',#entropy: ức độ hỗn loạn không đồng nhất của dữ liệu, crossentropy so sánh sự không đồng đều giữa 2 entropy
#                                 #cross entropy = entropy + hệ số phân tán KL devergence (p,q) = -sum([p[i]*np.log(p[i]/q[i]) for i in range len(p[i])])
#                              metrics = ['accuracy'])
# model_training_first.fit(np.array([x[0] for _,x in enumerate(X_train)]),np.array([y[1] for _,y in enumerate(X_train)]) , epochs =20)#chay 20 lan
# model_training_first.save('model_CNN_20epochs.keras')
# import time

# listResult = ['Fake','Fake','Fake','Fake','Real']
# cam = cv2.VideoCapture(0)
# face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')  # tap tin detec face
# models = models.load_model('model_CNN_20epochs.keras')
# while (True):
#     OK, frame = cam.read()
#     faces = face_detector.detectMultiScale(frame, 1.3, 5)
#     # time.sleep(0.5)
#     for (X, y, w, h) in faces:
#         roi = cv2.resize(frame[y + 2:y + h - 2, X + 2:X + w - 2], (299, 299))
#         # cv2.imwrite('img_roi/roi_{}.jpg'.format(count), roi)
#         result = np.argmax(models.predict(roi.reshape((-1,299,299,3))))
#         cv2.rectangle(frame, (X, y), (X + w, y + h), (0, 255, 0), 1)
#         cv2.putText(frame,listResult[result], (X+15,y-15), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,255), 2)
#
#     cv2.imshow('FRAME', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindow()

