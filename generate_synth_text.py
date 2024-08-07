import os
import argparse
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import random
from tqdm import tqdm
from datetime import datetime, timedelta


class Config:

    def __init__(
        self,
        target_folder="output",
        fonts=["arial.ttf"],
        height=50,
        width=100,
        text_size=40,
        json_file=None,
        background_color="white",
        text_color="black",
        angle=0,
        x_offset=0,
        y_offset=0,
        count=1,
        vocabulary_count=1000,
        start_date=None,
        end_date=None,
        text_color_offset=(0, 0, 0),
        background_color_offset=(0, 0, 0),
        text_strings=[],
        vocabulary=None,
    ):
        self.target_folder = target_folder
        self.fonts = fonts
        self.height = height
        self.width = width
        self.text_size = text_size
        self.json_file = json_file
        self.background_color = background_color
        self.text_color = text_color
        self.angle = angle
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.count = count
        self.start_date = start_date
        self.end_date = end_date
        self.text_color_offset = text_color_offset
        self.background_color_offset = background_color_offset
        self.text_strings = text_strings
        self.vocabulary = vocabulary
        self.vocabulary_count = vocabulary_count


class TimeImageGenerator:
    def __init__(self, config):
        self.config = config
        self.ocr_data = []

    def random_color(self, base_color, offset):
        base_rgb = ImageColor.getrgb(base_color)
        offset_rgb = (
            random.randint(-offset[0], offset[0]),
            random.randint(-offset[1], offset[1]),
            random.randint(-offset[2], offset[2]),
        )
        new_rgb = tuple(
            max(0, min(255, base + off)) for base, off in zip(base_rgb, offset_rgb)
        )
        return new_rgb

    def generate_word_from_vocabulary(self, length):
        return "".join(random.choice(self.config.vocabulary) for _ in range(length))

    def generate_image(self, text):
        background_color_rgb = self.random_color(
            self.config.background_color, self.config.background_color_offset
        )
        text_color_rgb = self.random_color(
            self.config.text_color, self.config.text_color_offset
        )
        font_path = random.choice(self.config.fonts)
        font = ImageFont.truetype(font_path, self.config.text_size)

        image = Image.new(
            "RGB",
            (self.config.width, self.config.height),
            color=background_color_rgb,
        )
        draw = ImageDraw.Draw(image)

        text_x = 0
        text_y = 0

        if self.config.x_offset != 0:
            text_x += random.randint(0, self.config.x_offset)
        if self.config.y_offset != 0:
            text_y += random.randint(0, self.config.y_offset)

        draw.text(
            (text_x, text_y),
            text,
            font=font,
            fill=text_color_rgb,
        )

        angle = 0
        if self.config.angle != 0:
            angle = random.uniform(-self.config.angle, self.config.angle)
            image = image.rotate(angle, expand=1, fillcolor=background_color_rgb)

        # Обрезка изображения до исходного размера
        image = image.crop((0, 0, self.config.width, self.config.height))

        return image, angle

    def save_image(self, image, file_name, text, angle=0):
        os.makedirs(self.config.target_folder, exist_ok=True)
        file_path = os.path.join(self.config.target_folder, file_name)
        file_path = os.path.normpath(file_path)
        image.save(file_path)

        ocr_entry = {"file": file_path, "text": text}
        if self.config.angle != 0:
            ocr_entry["angle"] = angle
        self.ocr_data.append(ocr_entry)

    def generate_time_images(self):
        total_iterations = self.config.count * 24 * 60
        with tqdm(total=total_iterations, desc="Generating time images") as pbar:
            index = 0
            for _ in range(self.config.count):
                for hour in range(24):
                    for minute in range(60):
                        time_str = f"{hour:02}:{minute:02}"
                        image, angle = self.generate_image(time_str)
                        file_name = f"time_{index:06d}.png"
                        self.save_image(image, file_name, time_str, angle)
                        pbar.update(1)
                        index += 1

    def generate_date_images(self):
        start_date = datetime.strptime(self.config.start_date, "%d.%m.%Y")
        end_date = datetime.strptime(self.config.end_date, "%d.%m.%Y")
        total_days = (end_date - start_date).days + 1

        with tqdm(total=total_days * 2, desc="Generating date images") as pbar:
            index = 0
            for single_date in (start_date + timedelta(n) for n in range(total_days)):
                for date_format in [
                    "%d.%m.%Y",
                    "%d %m.%Y",
                    "%d%m%Y",
                    "%d %m%Y",
                    "%d %m %Y",
                    "%d-%m-%Y",
                ]:
                    date_str = single_date.strftime(date_format)
                    image, angle = self.generate_image(date_str)
                    file_name = f"date_{index:06d}.png"
                    self.save_image(image, file_name, date_str, angle)
                    pbar.update(1)
                    index += 1

    def generate_custom_text_images(self):
        with tqdm(
            total=len(self.config.text_strings) * 2,
            desc="Generating custom text images",
        ) as pbar:
            index = 0
            for _ in range(self.config.count):
                for text in self.config.text_strings:
                    words = text.split()
                    for word in words:
                        for i in range(1, len(word) + 1):
                            partial_text = word[:i]
                            image, angle = self.generate_image(partial_text)
                            file_name = f"word_{index:06d}.png"
                            index += 1
                            self.save_image(image, file_name, partial_text, angle)
                            pbar.update(1)
                        for i in range(len(word), 0, -1):
                            partial_text = word[:i]
                            image, angle = self.generate_image(partial_text)
                            file_name = f"word_{index:06d}.png"
                            index += 1
                            self.save_image(image, file_name, partial_text, angle)
                            pbar.update(1)

    def generate_vocabulary_images(self):
        with tqdm(
            total=self.config.count * self.config.vocabulary_count,
            desc="Generating vocabulary images",
        ) as pbar:
            index = 0
            for _ in range(self.config.count):
                for i in range(self.config.vocabulary_count):
                    word_length = random.randint(1, 9)
                    word = self.generate_word_from_vocabulary(word_length)
                    image, angle = self.generate_image(word)
                    file_name = f"vocab_{index:06d}.png"
                    self.save_image(image, file_name, word, angle)
                    pbar.update(1)
                    index += 1

    def save_ocr_data(self):
        if self.config.json_file:
            with open(self.config.json_file, "w", encoding="utf-8") as json_file:
                json.dump(
                    {"ocr_files": self.ocr_data},
                    json_file,
                    ensure_ascii=False,
                    indent=4,
                )


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    generator = TimeImageGenerator(config)
    if config.start_date and config.end_date:
        generator.generate_time_images()
        generator.generate_date_images()
    if config.text_strings:
        generator.generate_custom_text_images()
    if config.vocabulary:
        generator.generate_vocabulary_images()
    generator.save_ocr_data()
    return generator


def main():
    config = Config()
    config.target_folder = "/content/synth2/ocr"
    config.fonts = [
        "/content/fonts/Fira Sans Regular.ttf",
        "/content/fonts/Fira Sans Light.ttf",
        "/content/fonts/Fira Sans Thin.ttf",
        "/content/fonts/Fira Sans Condensed Thin.ttf",
    ]
    config.json_file = "/content/synth2/ocr.json"
    config.height = 50
    config.width = 200
    config.text_size = 40
    config.background_color = "rgb(181, 181, 181)"
    config.text_color = "rgb(139, 139, 139)"
    config.angle = 1
    config.start_date = "01.01.2020"
    config.end_date = "31.12.2030"
    config.x_offset = 0
    config.y_offset = 0
    config.count = 10
    config.text_color_offset = (5, 5, 5)
    config.background_color_offset = (5, 5, 5)
    config.text_strings = [
        "ПВХ PROPLEX L 1063 N ГОСТ 30673 18 072024 10 59 7л2",
        "ПВХ PROPLEX L 1.063 N ГОСТ 30673 24-07-24 16:39 2л1",
        "BERTASILVERECO BN_1 070 5 ГОСТ 30673 24.07.24 16:43 л16 1",
        "BERTASILV ERTASILVE RTASILVER TASILVERE ASILVEREC SILVERECO",
        "BERTASIL ERTASILV RTASILVE TASILVER ASILВЕРЕ SILVEREC ILVERECO",
        "BERTASI ERTASIL RTASILV TASILVE ASILВЕР SILVERE ILVEREC LVERECO",
        "BERTAS ERTASI RTASIL TASILV ASILВЕ SILVERE ILVERE LVEREC VERECO",
        "BERTA ERTAS RTASI TASIL ASILВЕ SILVE ILVER LVERE VEREC ERECO BERT",
        "ERT RTAS TASI ASIL SILV ILVE LVER VERE EREC RECO",
    ]

    config.vocabulary = "012.LN436785ВПХPл9-СГОТEO:RXB_CTASIVР"
    config.vocabulary_count = 20000

    parser = argparse.ArgumentParser(
        description="Утилита для создания синтетических текстовых данных для OCR"
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Папка для сохранения изображений",
    )
    parser.add_argument(
        "--fonts",
        nargs="+",
        default=config.fonts,
        help="Список путей к файлам шрифтов TTF",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.height,
        help="Высота выходного изображения в пикселях",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.width,
        help="Ширина выходного изображения в пикселях",
    )
    parser.add_argument(
        "--text_size",
        type=int,
        default=config.text_size,
        help="Размер текста в изображении",
    )
    parser.add_argument(
        "--json_file",
        type=str,
        default=config.json_file,
        help="Путь к JSON-файлу для сохранения данных OCR",
    )
    parser.add_argument(
        "--background_color",
        type=str,
        default=config.background_color,
        help="Цвет фона изображения",
    )
    parser.add_argument(
        "--text_color",
        type=str,
        default=config.text_color,
        help="Цвет текста изображения",
    )
    parser.add_argument(
        "--angle",
        type=float,
        default=config.angle,
        help="Максимальный угол поворота текста",
    )
    parser.add_argument(
        "--x_offset",
        type=int,
        default=config.x_offset,
        help="Максимальное случайное смещение текста по горизонтали",
    )
    parser.add_argument(
        "--y_offset",
        type=int,
        default=config.y_offset,
        help="Максимальное случайное смещение текста по вертикали",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=config.count,
        help="Количество повторений полного цикла генерации изображений",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=config.start_date,
        help="Начальная дата в формате ДД.ММ.ГГГГ",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=config.end_date,
        help="Конечная дата в формате ДД.ММ.ГГГГ",
    )
    parser.add_argument(
        "--text_color_offset",
        type=int,
        nargs=3,
        default=config.text_color_offset,
        help="Смещение цвета текста (r, g, b)",
    )
    parser.add_argument(
        "--background_color_offset",
        type=int,
        nargs=3,
        default=config.background_color_offset,
        help="Смещение цвета фона (r, g, b)",
    )
    parser.add_argument(
        "--text_strings",
        nargs="+",
        default=config.text_strings,
        help="Список текстовых строк для генерации изображений",
    )
    parser.add_argument(
        "--vocabulary",
        type=str,
        default=config.vocabulary,
        help="Словарь символов для генерации случайных слов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
