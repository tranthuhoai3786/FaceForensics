import os
import telebot
import requests
from check_image import checkImage
from check_video import checkVideo
# from check_audio import checkAudio

API_KEY = '5676548221:AAEjZwiAqICj0RYSTEOxt5oubneBpBSz8dM'
bot = telebot.TeleBot(API_KEY)
# print(API_KEY)

@bot.message_handler(commands=['start'])
def start(message):
    bot.reply_to(message, "Xin chào! Để sử dụng bot, gõ /help để xem danh sách các lệnh.")

# Xử lý lệnh /help
@bot.message_handler(commands=['help'])
def help(message):
    # Hiển thị hướng dẫn và danh sách các lệnh của bot
    response = "Đây là danh sách các lệnh mà bot hỗ trợ:\n"
    response += "/start - Hiển thị lời chào từ bot.\n"
    response += "/Hello - Hiển thị tên bot.\n"
    response += "/help - Hiển thị danh sách các lệnh và giúp đỡ.\n"
    response += "/check_fake_image - Kiểm tra hình ảnh.\n"
    response += "/check_fake_audio - Kiểm tra âm thanh.\n"
    response += "/check_fake_video - Kiểm tra video.\n"
    # Thêm các lệnh khác và giúp đỡ khác tùy ý

    bot.send_message(message.chat.id, response)


@bot.message_handler(commands=['Hello'])
def hello(message):
    bot.send_message(message.chat.id, "Hi, tôi là Cindy Tran. Bot hỗ trợ nhận diện fake/real hình ảnh, âm thanh, video.")

@bot.message_handler(content_types=['photo'])
def down_load_image(message):

    photo_id = message.photo[-1].file_id#lay id
    # print(photo_id)
    file_info = bot.get_file(photo_id)#lay thong tin ve anh
    photo_url = f"https://api.telegram.org/file/bot{API_KEY}/{file_info.file_path}"
    try:
        response = requests.get(photo_url)#tai anh xuong
        response.raise_for_status()#kiem tra loi khi tai xuong
        # bot.reply_to(message,"tai thanh cong")
        with open("download.jpg", "wb") as f:
            f.write(response.content)
        print("Tải ảnh thành công và đã lưu vào download.jpg")
    except requests.exceptions.HTTPError as httperr:
        bot.reply_to(message,httperr)
    except requests.exceptions.ConnectionError as cnerr:
        bot.reply_to(message,cnerr)
    except requests.exceptions.Timeout as timeerr:
        bot.reply_to(message,timeerr)
    except requests.exceptions.RequestException as err:
        bot.reply_to(message,err)

        # bot.reply_to(message,"gui anh de kiem tra")

@bot.message_handler(commands=['check_fake_image'])
def check_deep_fake(message):
    result, sign, out_path, percent = checkImage()
    response = f"ảnh của bạn là ảnh {result}\n"
    response += f"dấu hiệu nhận biết là {sign}\n"
    response += f"tỉ lệ {result} là {percent}%\n"
    response += f"vùng {result} là "
    # bot.send_message(message.chat.id, f"ảnh của bạn là ảnh {result}")
    # bot.send_message(message.chat.id, f"dấu hiệu nhận biết là {sign}")
    # bot.send_message(message.chat.id, f"tỉ lệ {result} là {percent}%")
    bot .send_message(message.chat.id, response)
    try:
        with open(out_path, "rb") as photo:
            # bot.send_message(message.chat.id, f"vùng {result} là ")
            bot.send_photo(message.chat.id, photo)

    except FileNotFoundError:
        bot.reply_to(message, "Không tìm thấy ảnh.")

    except Exception as e:
        bot.reply_to(message, f"Có lỗi xảy ra: {e}")
    # bot.send_message(message.chat.id, f"vung fake la {out}")




# download video
@bot.message_handler(content_types=['video'])
def download_video(message):
    # Lấy ID của video gửi lên
    video_id = message.video.file_id

    # Sử dụng Telegram Bot API để lấy thông tin về video
    file_info = bot.get_file(video_id)

    # Tạo đường dẫn URL để tải xuống video
    video_url = f"https://api.telegram.org/file/bot{API_KEY}/{file_info.file_path}"

    try:
        # Tải video xuống từ URL
        response = requests.get(video_url)
        response.raise_for_status()  # Kiểm tra lỗi khi tải xuống

        # Lưu video xuống một tệp cụ thể
        with open("video.mp4", "wb") as f:
            f.write(response.content)

        print("Tải video thành công và đã lưu vào downloaded_video.mp4")

    except requests.exceptions.HTTPError as errh:
        # Xử lý lỗi HTTP
        print(f"Lỗi HTTP: {errh}")
    except requests.exceptions.ConnectionError as errc:
        # Xử lý lỗi kết nối
        print(f"Lỗi kết nối: {errc}")
    except requests.exceptions.Timeout as errt:
        # Xử lý lỗi timeout
        print(f"Lỗi timeout: {errt}")
    except requests.exceptions.RequestException as err:
        # Xử lý lỗi không xác định
        print(f"Lỗi không xác định: {err}")

@bot.message_handler(commands=['check_fake_video'])
def check_fake_video(message):
    result, sign, out_path, percent = checkVideo()
    response = f"video của bạn là video {result}\n"
    response += f"dấu hiệu nhận biết là {sign}\n"
    response += f"tỉ lệ {result} là {percent}%\n"
    response += f"vùng {result} là "
    # bot.send_message(message.chat.id, f"video của bạn là video {result}")
    # bot.send_message(message.chat.id, f"dấu hiệu nhận biết là {sign}")
    # bot.send_message(message.chat.id, f"tỉ lệ {result} là {percent}%")
    bot.send_message(message.chat.id, response)
    try:
        with open(out_path, "rb") as video:
            # bot.send_message(message.chat.id, f"vùng {result} là ")
            bot.send_video(message.chat.id, video)

    except FileNotFoundError:
        bot.reply_to(message, "Không tìm thấy video.")

    except Exception as e:
        bot.reply_to(message, f"Có lỗi xảy ra: {e}")

#download audio
# Xử lý tệp âm thanh
@bot.message_handler(content_types=['audio'])
def handle_audio(message):
    try:
        # Lấy thông tin về tệp âm thanh
        file_id = message.audio.file_id
        file_info = bot.get_file(file_id)
        file_path = file_info.file_path

        # Tải về tệp âm thanh
        downloaded_file = bot.download_file(file_path)

        # Lưu tệp âm thanh
        audio_save_path = "./file.mp3"
        with open(audio_save_path, 'wb') as f:
            f.write(downloaded_file)

        print("Đã tải về và lưu file.mp3 thành công")

    except Exception as e:
        # Xử lý lỗi
        bot.reply_to(message, f"Có lỗi xảy ra: {e}")


# @bot.message_handler(commands=['check_fake_audio'])
# def check_fake_audio(message):
#     result, percent = checkAudio()
#     response = f"âm thanh của bạn là âm thanh {result}\n"
#     response += f"tỉ lệ {result} là {percent}%\n"
#     bot.send_message(message.chat.id, f"âm thanh của bạn là âm thanh {result}")
#     bot.send_message(message.chat.id, f"tỉ lệ {result} là {percent}%")

#    # bot.send_message(message.chat.id, f"vung fake la {out}")
#     bot.send_message(message.chat.id, response)


bot.polling()#chay bot


