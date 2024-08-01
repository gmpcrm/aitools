import os
import argparse
import json
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np


class Config:
    def __init__(
        self,
        target_folder="output",
        font="arial.ttf",
        height=50,
        width=100,
        json_file=None,
        background_color="white",
        text_color="black",
    ):
        self.target_folder = target_folder
        self.font = font
        self.height = height
        self.width = width
        self.json_file = json_file
        self.background_color = background_color
        self.text_color = text_color


class TimeImageGenerator:
    def __init__(self, config):
        self.config = config
        self.font = ImageFont.truetype(config.font, config.height - 10)
        self.ocr_data = []

    def generate_time_images(self):
        if not os.path.exists(self.config.target_folder):
            os.makedirs(self.config.target_folder)

        background_color_rgb = ImageColor.getrgb(self.config.background_color)
        text_color_rgb = ImageColor.getrgb(self.config.text_color)

        for hour in range(24):
            for minute in range(60):
                time_str = f"{hour:02}:{minute:02}"
                image = Image.new(
                    "RGB",
                    (self.config.width, self.config.height),
                    color=background_color_rgb,
                )
                draw = ImageDraw.Draw(image)

                bbox = draw.textbbox((0, 0), time_str, font=self.font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                text_x = (self.config.width - text_width) // 2
                # text_y = (self.config.height - text_height) // 2 + offset
                text_y = 0
                text_x = 0

                draw.text(
                    (text_x, text_y), time_str, font=self.font, fill=text_color_rgb
                )

                file_name = f"{hour:02}_{minute:02}.png"
                file_path = os.path.join(self.config.target_folder, file_name)
                file_path = os.path.normpath(file_path)
                image.save(file_path)

                if self.config.json_file:
                    self.ocr_data.append({"file": file_path, "text": time_str})

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
    return generator


def main():
    config = Config()
    config.target_folder = "/content/synt/ocr"
    config.font = "/content/fonts/Fira Sans Regular.ttf"
    config.json_file = "/content/synt/ocr.json"
    config.height = 50
    config.width = 200
    config.background_color = "rgb(181, 181, 181)"
    config.text_color = "rgb(139, 139, 139)"

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
        "--font",
        type=str,
        default=config.font,
        help="Путь к файлу шрифта TTF",
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

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
