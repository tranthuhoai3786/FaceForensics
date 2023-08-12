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

TRAIN_DATA = 'data/train'
TEST_DATA = 'data/test'
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
#
# model_training_first = models.Sequential([
#     layers.Conv2D(64, kernel_size=7, strides=2, input_shape = (64,64,3), activation='relu'),
#     layers.MaxPooling2D(pool_size=3, strides=2),
#     layers.BatchNormalization(),
#
#     layers.Conv2D(48, kernel_size=5, strides=1, activation='relu'),
#     layers.MaxPooling2D(pool_size=3, strides=2),
#     layers.BatchNormalization(),
#
#     layers.Flatten(),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(4096, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(5, activation='softmax')
# ])
# #model_training_first.summary()#ket cau model san sinh ra bao nhieu sieu tham so
# model_training_first.compile(optimizer = 'adam',#ky thuat toi uu hoa adam
#                              loss= 'categorical_crossentropy',#entropy: ức độ hỗn loạn không đồng nhất của dữ liệu, crossentropy so sánh sự không đồng đều giữa 2 entropy
#                                 #cross entropy = entropy + hệ số phân tán KL devergence (p,q) = -sum([p[i]*np.log(p[i]/q[i]) for i in range len(p[i])])
#                              metrics = ['accuracy'])
# model_training_first.fit(np.array([x[0] for _,x in enumerate(X_train)]),np.array([y[1] for _,y in enumerate(X_train)]) , epochs =20)#chay 20 lan
# model_training_first.save('model_CNN_20epochs.keras')
# import time

listResult = ['Fake','Fake','Fake','Fake','Real']
# cam = cv2.VideoCapture('000_003.mp4')
# cam = cv2.VideoCapture('009.mp4')

cam = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')  # tap tin detec face
models = models.load_model('model_CNN_20epochs.keras')
# while (True):
#     OK, frame = cam.read()
#     faces = face_detector.detectMultiScale(frame, 1.3, 5)
#     # time.sleep(0.5)
#     for (X, y, w, h) in faces:
#         roi = cv2.resize(frame[y + 2:y + h - 2, X + 2:X + w - 2], (64, 64))
#         # cv2.imwrite('img_roi/roi_{}.jpg'.format(count), roi)
#         result = np.argmax(models.predict(roi.reshape((-1,64,64,3))))
#         cv2.rectangle(frame, (X, y), (X + w, y + h), (0, 255, 0), 1)
#         cv2.putText(frame,listResult[result], (X+15,y-15), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255,255,0), 2)
#
#     cv2.imshow('FRAME', frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cam.release()
# cv2.destroyAllWindow()
#
