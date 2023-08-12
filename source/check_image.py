import cv2
import tensorflow as tf
import numpy as np
# import keras


def checkImage():
    from tensorflow.keras import models
    listResult = ['Fake', 'Fake', 'Fake', 'Fake', 'Real']
    listResult1 = ['deepfake','face2face','faceswap','neural textures', 'real']
    haar_path = '../haarcascades/haarcascade_frontalface_alt.xml'
    # img_path = './download.jpg'
    img_path = './cc.jpg'
    model_path = '../model_CNN_20epochs.keras'
    # model_path = '../test (1).keras'
    # model_path = '../model_faceforensics_20epochs.keras'
    face_detector = cv2.CascadeClassifier(haar_path)
    img = cv2.imread(img_path)
    faces = face_detector.detectMultiScale(img, 1.3, 5)
    models = models.load_model(model_path)
    result =4
    for x, y, w, h in faces:
        roi = cv2.resize(img[y + 2:y + h - 2, x + 2:x + w - 2], (64, 64))
        result = np.argmax(models.predict(roi.reshape((-1, 64, 64, 3))))

        arr = np.array(models.predict(roi.reshape((-1, 64, 64, 3))))
        max_value = np.max(arr)*100

        print(models.predict(roi.reshape((-1, 64, 64, 3))))
        # print()
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.putText(img, listResult[result], (x + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        img_output_path = './chong_eim.jpg'
        cv2.imwrite(img_output_path, img)
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.imshow('IMG',img)
        cv2.waitKey(0)
        # return
    return listResult[result],listResult1[result],img_output_path,max_value



#     cv2.imshow('IMG',img)
# cv2.waitKey(0)
print(checkImage())