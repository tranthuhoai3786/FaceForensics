# import speech_recognition as sr
#
# def count_speakers_in_audio(audio_file):
#     recognizer = sr.Recognizer()
#
#     with sr.AudioFile(audio_file) as source:
#         audio_data = recognizer.record(source)
#
#     # Sử dụng Google Web Speech API để nhận dạng giọng nói
#     try:
#         text = recognizer.recognize_google(audio_data)
#         print(text)
#         speakers = set(text.split())
#         num_speakers = len(speakers)
#         return num_speakers
#     except sr.UnknownValueError:
#         print("Không thể nhận dạng giọng nói.")
#         return 0
#     except sr.RequestError:
#         print("Không thể truy cập Google Web Speech API.")
#         return 0
#
# audio_file_path = "../ASVspoof2017_V2_dev/Real/D_1000001.wav"  # Thay đổi đường dẫn đến tệp âm thanh của bạn
# num_speakers = count_speakers_in_audio(audio_file_path)
# print(f"Số lượng người nói trong tệp âm thanh: {num_speakers}")







