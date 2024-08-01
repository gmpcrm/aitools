import os
import argparse
import shutil
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


class Config:
    def __init__(
        self,
        source_folder="~/data/source",
        target_folder="~/data/target",
        min_height=None,
        max_height=None,
        min_width=None,
        max_width=None,
        width_file="width_histogram.png",
        height_file="height_histogram.png",
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.min_height = min_height
        self.max_height = max_height
        self.min_width = min_width
        self.max_width = max_width
        self.width_file = width_file
        self.height_file = height_file


class SortImages:
    def __init__(self, config):
        self.config = config

    def get_image_size(self, img_path):
        try:
            with Image.open(img_path) as img:
                return img.size
        except Exception as e:
            print(f"Не удалось получить размеры изображения {img_path}: {e}")
            return None, None

    def get_image_sizes(self):
        widths = []
        heights = []

        for root, dirs, files in os.walk(self.config.source_folder):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                    width, height = self.get_image_size(os.path.join(root, file))
                    if width is not None and height is not None:
                        widths.append(width)
                        heights.append(height)

        return widths, heights

    def move_image_if_needed(self, img_path):
        width, height = self.get_image_size(img_path)
        if width is not None and height is not None:
            if (
                (self.config.min_height is not None and height < self.config.min_height)
                or (
                    self.config.max_height is not None
                    and height > self.config.max_height
                )
                or (self.config.min_width is not None and width < self.config.min_width)
                or (self.config.max_width is not None and width > self.config.max_width)
            ):
                try:
                    os.makedirs(self.config.target_folder, exist_ok=True)
                    shutil.move(
                        img_path,
                        os.path.join(
                            self.config.target_folder, os.path.basename(img_path)
                        ),
                    )
                    print(f"Перемещен {img_path} в {self.config.target_folder}")
                except Exception as e:
                    print(f"Не удалось переместить файл {img_path}: {e}")

    def plot_histogram(self, data, title, xlabel, ylabel, output_file):
        plt.figure()
        plt.hist(data, bins=30, alpha=0.7, color="blue", edgecolor="black")
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.savefig(output_file)
        plt.close()

    def process_images(self):
        for root, dirs, files in os.walk(self.config.source_folder):
            for file in files:
                if file.lower().endswith(("png", "jpg", "jpeg", "bmp", "gif")):
                    self.move_image_if_needed(os.path.join(root, file))

        widths, heights = self.get_image_sizes()
        self.plot_histogram(
            widths,
            "Гистограмма ширины изображений",
            "Ширина (пиксели)",
            "Частота",
            self.config.width_file,
        )
        self.plot_histogram(
            heights,
            "Гистограмма высоты изображений",
            "Высота (пиксели)",
            "Частота",
            self.config.height_file,
        )

        print(
            f"Гистограммы сохранены как '{self.config.width_file}' и '{self.config.height_file}'"
        )


def run(**kwargs):
    config = Config(**kwargs)
    processor = SortImages(config)
    processor.process_images()


def run_config(config):
    processor = SortImages(config)
    processor.process_images()
    return config


if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(
        description="Сканирует папку для изображений, строит гистограммы их ширины и высоты и перемещает изображения в целевую папку на основе заданных критериев размеров."
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Путь к исходной папке, содержащей изображения",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к целевой папке для перемещения изображений, соответствующих критериям размеров",
    )
    parser.add_argument(
        "--min_height",
        type=int,
        default=config.min_height,
        help="Минимальная высота изображений для перемещения",
    )
    parser.add_argument(
        "--max_height",
        type=int,
        default=config.max_height,
        help="Максимальная высота изображений для перемещения",
    )
    parser.add_argument(
        "--min_width",
        type=int,
        default=config.min_width,
        help="Минимальная ширина изображений для перемещения",
    )
    parser.add_argument(
        "--max_width",
        type=int,
        default=config.max_width,
        help="Максимальная ширина изображений для перемещения",
    )
    parser.add_argument(
        "--width_file",
        type=str,
        default=config.width_file,
        help="Имя файла для сохранения гистограммы ширины",
    )
    parser.add_argument(
        "--height_file",
        type=str,
        default=config.height_file,
        help="Имя файла для сохранения гистограммы высоты",
    )
    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)
