import logging
import telebot
from telebot import types
import logging
from config import *
from flask import Flask, request
import os
import shutil
from utils import stylize_image

server = Flask(__name__)
bot = telebot.TeleBot(token=API)
logger = telebot.logger
logger.setLevel(logging.DEBUG)

def clear_folder(message: types.Message):
    """Clears username folder"""
    username = message.from_user.first_name
    path = f"uploaded_images/{username}"

    shutil.rmtree(path)

def upload_image_on_server(message_info: types.Message):
    """Function takes photo from message and saves it."""
    if message_info.photo:
        folder_name = message_info.from_user.first_name

        if not os.path.exists(f"uploaded_images"):
            os.mkdir(f"uploaded_images")

        if not os.path.exists(f"uploaded_images/{folder_name}"):
            os.mkdir(f"uploaded_images/{folder_name}")

        image_info = bot.get_file(message_info.photo[-1].file_id)
        image = bot.download_file(file_path=image_info.file_path)

        with open(f"uploaded_images/{folder_name}/content.jpg", "wb") as file:
            file.write(image)

        bot.send_message(message_info.chat.id, f"""
        Content was recieved...
        """)

        choose_genre(message_info)

@bot.message_handler(commands=["start"])
def start(message: types.Message):
    bot.send_message(message.chat.id, """Hi, I'm StyleTransferBot. Let's begin!""")
    upload_content(message)

def upload_content(message: types.Message):
    content = bot.send_message(message.chat.id, """So, upload the image you want to stilize..""")
    bot.register_next_step_handler(content, upload_image_on_server)

def choose_genre(message: types.Message):
    genres = os.listdir("styles")
    pick_genre = types.InlineKeyboardMarkup(row_width=1)
    for genre in genres:
        pick_genre.add(types.InlineKeyboardButton(f"{str(genre).capitalize()}", callback_data=f"{genre}"))


    bot.send_message(message.chat.id, """
What genre are you interested in?         
    """, reply_markup=pick_genre)

@bot.callback_query_handler(func=lambda genre: True)
def choose_style(genre):
    global style 
    style = genre.data
    pics_path = os.listdir(f"styles/{style}")
    pics = []

    for pic_path in sorted(pics_path, key=lambda x: x.split(".")[0][-1], reverse=False):
        pic = types.InputMediaPhoto(open(f"styles/{style}/{pic_path}", "rb"), caption=pic_path.split(".")[0][-1])
        pics.append(pic)
    bot.send_media_group(genre.from_user.id, pics)

    pick_style = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    buttons = [str(i + 1) for i in range(len(pics_path))]
    pick_style.add(*buttons, "Back")

    style_num = bot.send_message(genre.from_user.id, f"""
You picked {str(style).capitalize()}
Now choose the picture, which style you want to apply...
    """, reply_markup=pick_style)

@bot.message_handler(content_types=["text"])
def stilize(message: types.Message):
    if message.text == "Back":
        choose_genre(message)
    if message.text == "Image":
        upload_content(message)
    if message.text == "Style":
        choose_genre(message)
    if message.text == "Finish":
        clear_folder(message)
        bot.send_message(message.chat.id, """Fine, sya! If you want to try again, type /start!""")

    if str(message.text).isdigit():
        number = message.text
        username = message.from_user.first_name
        model_path = f"models/{style}/{style}{number}.model"
        content_path = f"uploaded_images/{username}/content.jpg"
        output_path = f"uploaded_images/{username}/stilized.jpg"

        if not os.path.exists(model_path):
            bot.send_message(message.chat.id, """
Hmm... Actually that style doesn't exists for now.
Try another style...             
            """)
            choose_genre(message)
        
        else:
            bot.send_message(message.chat.id, f"""
You've just picked picture {number} in genre "{str(style).capitalize()}".
Wait until you get the picture... (Usually it takes around 10-15 seconds)
            """)
            stylize_image(content_path, model_path, output_path)
            bot.send_photo(message.chat.id, photo=open(output_path, "rb"))

            finish = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
            finish.add("Image", "Style", "Finish")

            answer = bot.send_message(message.chat.id, """That's it! Wanna change something?""", reply_markup=finish)


@server.route(f"/{API}", methods=["POST"])
def redirect_message():
    json_string = request.stream.read().decode("utf-8")
    update = types.Update.de_json(json_string)
    bot.process_new_updates([update])
    return "!", 200

if __name__ == "__main__":
    bot.remove_webhook()
    bot.set_webhook(url=APP_URL)
    server.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

