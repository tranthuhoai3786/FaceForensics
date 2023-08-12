import cv2
# import dlib
import time
import os


'''
Đọc từng video có trong từng thư mục trong folder archive
Các frame detect khuôn mặt scale 1.3
Lưu các ảnh khuôn mặt vào các thư mục trong folder data
=> KQ: thư mục data chứa ảnh các khuôn mặt được detect
'''
# Khởi tạo bộ nhận diện khuôn mặt dlib
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')  # tap tin detect face
# detector = dlib.get_frontal_face_detector()

# Mở tệp video
TRAIN_DATA = 'archive/test/NeuralTextures'#thay tên tệp
X_train=[]
count = 0
def getData(dirData, listData,count):

    for filename in os.listdir(dirData):
        filename_path = os.path.join(dirData, filename)##archive/train/Deepfakes/100_077.mp4
        cam = cv2.VideoCapture(filename_path)  # camera tren lap
        count1=0
        while True:
            OK, frame = cam.read()
            time.sleep(0.25)
            faces = face_detector.detectMultiScale(frame, 1.3, 5)
            for (X, y, w, h) in faces:
                roi = cv2.resize(frame[y + 2:y + h - 2, X + 2:X + w - 2], (299, 299))
                cv2.imwrite('image/test/NeuralTextures/roi_{}.jpg'.format(count), roi)#thay tên tệp
                count += 1
                count1+=1
                cv2.rectangle(frame, (X, y), (X + w, y + h), (0, 255, 0), 1)

            # cv2.imshow('FRAME', frame)
            if count > 1:
                break

        cam.release()
        # cv2.destroyAllWindows()

    return
X_train = getData(TRAIN_DATA,X_train,0)
