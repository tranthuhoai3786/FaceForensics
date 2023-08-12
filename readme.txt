----------XỬ LÝ DATA------------
chia bộ train test. Mỗi folder gồm 5 folder gồm: Deepfakes, Face2Face, FaceSwap, NeuralTextures, youtube
Chia train test 90:10
----------CẤU TRÚC PRJ----------
-folder archive: chứa video dataset gồm 1000 video youtube, 1000 deepfakes, 1000 face2face, 1000 faceswap, 1000 neuraltextures
-foler data: chứa ảnh khuôn mặt được detect từ archive (size 64x64) đưa vào CNN
-folder haarcascades: detect khuôn mặt
-folder image : chứa ảnh khuôn mặt được detect từ archive (size 299x299) đưa vào Xception
+file videoToImage.py: detect khuôn mặt từ video thành ảnh
+file main.py: model CNN gồm có 3 layers
+file CNN.py: model CNN có 2 layers
+file Xception.py: model Xception (hiện tại thì chưa chạy được vì bị tràn bộ nhớ và không dùng được GPU)
+file .keras: model đã được training


-----------requiments------------------
python 3.10
Keras-Preprocessing	1.1.2
Markdown	3.4.3
MarkupSafe	2.1.3
Pillow	10.0.0
Werkzeug	2.3.6
absl-py	1.4.0
astunparse	1.6.3
cachetools	5.3.1
certifi	2023.5.7
charset-normalizer	3.2.0
flatbuffers	1.12
gast	0.4.0
google-auth	2.22.0
google-auth-oauthlib	0.4.6
google-pasta	0.2.0
grpcio	1.56.2
h5py	3.9.0
idna	3.4
keras	2.9.0
libclang	16.0.6
numpy	1.24.3
oauthlib	3.2.2
opencv-contrib-python	4.6.0.66
opt-einsum	3.3.0
packaging	23.1
pip	21.3.1	23.2.1
protobuf	3.19.6
pyasn1	0.5.0
pyasn1-modules	0.3.0
requests	2.31.0
requests-oauthlib	1.3.1
rsa	4.9
setuptools	68.0.0
six	1.16.0
tensorboard	2.9.1
tensorboard-data-server	0.6.1
tensorboard-plugin-wit	1.8.1
tensorflow	2.9.0
tensorflow-estimator	2.9.0
tensorflow-intel	2.13.0
tensorflow-io-gcs-filesystem	0.31.0
termcolor	2.3.0
typing-extensions	4.5.0
urllib3	1.26.16