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
        json_file=None,
        background_color="white",
        text_color="black",
        angle=0,
        x_offset=0,
        y_offset=0,
        count=1,
        start_date=None,
        end_date=None,
        text_color_offset=(0, 0, 0),
        background_color_offset=(0, 0, 0),
    ):
        self.target_folder = target_folder
        self.fonts = fonts
        self.height = height
        self.width = width
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

    def generate_image(self, text):
        background_color_rgb = self.random_color(
            self.config.background_color, self.config.background_color_offset
        )
        text_color_rgb = self.random_color(
            self.config.text_color, self.config.text_color_offset
        )
        font_path = random.choice(self.config.fonts)
        font = ImageFont.truetype(font_path, self.config.height - 10)

        image = Image.new(
            "RGB",
            (self.config.width, self.config.height),
            color=background_color_rgb,
        )
        draw = ImageDraw.Draw(image)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_x = 0
        text_y = 0

        if self.config.x_offset != 0:
            text_x = random.randint(0, self.config.x_offset)
        if self.config.y_offset != 0:
            text_y = random.randint(0, self.config.y_offset)

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

        if self.config.json_file:
            ocr_entry = {"file": file_path, "text": text}
            if self.config.angle != 0:
                ocr_entry["angle"] = angle
            self.ocr_data.append(ocr_entry)

    def generate_time_images(self):
        total_iterations = self.config.count * 24 * 60
        with tqdm(total=total_iterations, desc="Generating time images") as pbar:
            for _ in range(self.config.count):
                for hour in range(24):
                    for minute in range(60):
                        time_str = f"{hour:02}:{minute:02}"
                        image, angle = self.generate_image(time_str)
                        file_name = f"time_{hour:02}_{minute:02}_{_}.png"
                        self.save_image(image, file_name, time_str, angle)
                        pbar.update(1)

    def generate_date_images(self):
        start_date = datetime.strptime(self.config.start_date, "%d.%m.%Y")
        end_date = datetime.strptime(self.config.end_date, "%d.%m.%Y")
        total_days = (end_date - start_date).days + 1

        with tqdm(total=total_days, desc="Generating date images") as pbar:
            for single_date in (start_date + timedelta(n) for n in range(total_days)):
                date_str = single_date.strftime("%d%m%Y")
                image, angle = self.generate_image(date_str)
                file_name = f"date_{date_str}.png"
                self.save_image(image, file_name, date_str, angle)
                pbar.update(1)

        if self.config.json_file:
            with open(self.config.json_file, "w", encoding="utf-8") as json_file:
                np.random.shuffle(self.ocr_data)
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
    generator.generate_time_images()
    if config.start_date and config.end_date:
        generator.generate_date_images()
    return generator


def main():
    config = Config()
    config.target_folder = "/content/synt/ocr"
    config.fonts = [
        "/content/fonts/Fira Sans Regular.ttf",
        "/content/fonts/Fira Sans Light.ttf",
        "/content/fonts/Fira Sans Thin.ttf",
        "/content/fonts/Fira Sans Condensed Thin.ttf",
    ]
    config.json_file = "/content/synt/ocr.json"
    config.height = 50
    config.width = 200
    config.background_color = "rgb(181, 181, 181)"
    config.text_color = "rgb(139, 139, 139)"
    config.angle = 0.7
    config.x_offset = 7
    config.y_offset = 5
    config.count = 5
    config.start_date = "01.01.2024"
    config.end_date = "31.12.2030"
    config.text_color_offset = (5, 5, 5)
    config.background_color_offset = (5, 5, 5)

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

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
