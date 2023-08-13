from tensorflow.keras.models import load_model
import numpy as np
import librosa


def feature_extractor(file_name):
    audio, sample_rate = librosa.load(file_name)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis =0)
    return mfccs_scaled_features
def checkAudio():
    listResult = ['Real', 'Fake']
    model = load_model('../audio_classification.h5')
    filename = './file.mp3'
    predict_feature = feature_extractor(filename)
    predict_feature = predict_feature.reshape(1,-1)
    predictions = model.predict(predict_feature)

    arr = np.array(model.predict(predict_feature)[0])
    max_value = np.max(arr)*100

    print(predictions[0])

    # Tìm lớp có giá trị dự đoán cao nhất cho mỗi mẫu
    result = np.argmax(predictions, axis=-1)

    print(result[0])
    file_path = ''

    return listResult[result[0]], max_value

checkAudio()