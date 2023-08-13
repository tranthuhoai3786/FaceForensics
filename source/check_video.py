import cv2
import tensorflow as tf
import numpy as np
# import keras


def checkVideo():
    from tensorflow.keras import models
    listResult = ['Fake', 'Fake', 'Fake', 'Fake', 'Real']
    listResult1 = ['deepfake','face2face','faceswap','neural textures', 'real']
    haar_path = '../haarcascades/haarcascade_frontalface_alt.xml'
    video_path = './video.mp4'
    output_video_path = './video_out.mp4'
    # model_path = '../model_CNN_20epochs.keras'
    # model_path = '../test (1).keras'
    model_path = '../model_faceforensics_20epochs.keras'

    cap = cv2.VideoCapture(video_path)

    # Lấy thông tin về video (chiều rộng, chiều cao, số frame mỗi giây)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tạo đối tượng VideoWriter để lưu video mới
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng video mới là MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    face_detector = cv2.CascadeClassifier(haar_path)  # tap tin detec face
    models = models.load_model(model_path)
    while (cap.isOpened()):
        OK, frame = cap.read()
        if not OK :
            break
        faces = face_detector.detectMultiScale(frame, 1.3, 5)
        # time.sleep(0.5)
        for (X, y, w, h) in faces:
            roi = cv2.resize(frame[y + 2:y + h - 2, X + 2:X + w - 2], (64, 64))
            # cv2.imwrite('img_roi/roi_{}.jpg'.format(count), roi)
            result = np.argmax(models.predict(roi.reshape((-1, 64, 64, 3))))

            arr = np.array(models.predict(roi.reshape((-1, 64, 64, 3))))
            max_value = np.max(arr) * 100

            cv2.rectangle(frame, (X, y), (X + w, y + h), (0, 255, 0), 1)
            cv2.putText(frame, listResult[result], (X + 15, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


            out.write(frame)

            # video_output_path = './out.jpg'
            # cv2.imwrite(video_output_path, frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return listResult[result],listResult1[result],output_video_path,max_value





print(checkVideo())

