#deepface
from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

backends = ["opencv"," ssd","dlib","mtcnn","retinaface","mediapipe"]


image_path = "source/cc.jpg"
face = DeepFace.extract_faces(image_path, target_size = (128,128), detector_backend = "opencv")
plt.imshow(face)