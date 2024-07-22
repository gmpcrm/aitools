import argparse
import os
from PIL import Image, ImageDraw, ImageFont


class Config:
    def __init__(
        self,
        font_folder="c:/fonts/mix1",
        font=None,
        text="PROPLEX 4л1 ГОСТ N 30673 28.06.2024 16:08",
        size=(1200, 60),
        target_folder="c:/proplex/font_images",
        subfolder=False,
    ):
        self.font = font
        self.text = text
        self.size = size
        self.font_folder = font_folder
        self.target_folder = target_folder
        self.subfolder = subfolder


class FontImageCreator:
    def __init__(self, config):
        self.config = config

    def create_image_with_text(self, font_path, font_name=None):
        width, height = self.config.size
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        try:
            font = ImageFont.truetype(font_path, 40)
        except IOError:
            print(f"Не удалось загрузить шрифт {font_path}")
            return

        # Использование textbbox для получения размеров текста
        try:
            textbbox = draw.textbbox((0, 0), self.config.text, font=font)
            textwidth = textbbox[2] - textbbox[0]
            textheight = textbbox[3] - textbbox[1]
            x = (width - textwidth) // 2
            y = (height - textheight) // 2

            draw.text((x, y), self.config.text, font=font, fill=(0, 0, 0))

            if font_name:
                file_name = f"{font_name}.png"
            else:
                file_name = f"{os.path.basename(font_path)}.png"

            image.save(os.path.join(self.config.target_folder, file_name))
            print(f"Сохранено изображение с шрифтом {font_path} как {file_name}")
        except Exception as e:
            print(f"Ошибка при создании изображения: {e}")

    def create_images(self):
        if not os.path.exists(self.config.target_folder):
            os.makedirs(self.config.target_folder)

        if self.config.font:
            font_path = os.path.join(self.config.font_folder, self.config.font)
            self.create_image_with_text(font_path, self.config.font)
        else:
            for root, _, files in os.walk(self.config.font_folder):
                for font_file in files:
                    if font_file.endswith(".ttf") or font_file.endswith(".otf"):
                        font_path = os.path.join(root, font_file)
                        self.create_image_with_text(font_path)
                if not self.config.subfolder:
                    break


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    creator = FontImageCreator(config)
    creator.create_images()
    return creator


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для создания изображений с текстом и шрифтами"
    )

    parser.add_argument(
        "--font_folder",
        default=config.font_folder,
        type=str,
        help="Папка с шрифтами",
    )
    parser.add_argument(
        "--font",
        type=str,
        help="Файл шрифта для использования",
    )
    parser.add_argument(
        "--text",
        default=config.text,
        type=str,
        help="Текст для отображения",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=config.size,
        help="Размер изображения (ширина высота)",
    )
    parser.add_argument(
        "--target_folder",
        default=config.target_folder,
        type=str,
        help="Папка для сохранения изображений",
    )
    parser.add_argument(
        "--subfolder",
        action="store_true",
        help="Сканировать подпапки в папке шрифтов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    config.subfolder = True
    run_config(config)


if __name__ == "__main__":
    main()
