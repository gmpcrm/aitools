import os
import json
import shutil
import random
import argparse
import cv2
import numpy as np
from tqdm import tqdm


class Config:
    def __init__(
        self,
        source_files=[],
        target_folder="/content/ocr",
        target_json="/content/ocr.json",
        shuffle=False,
        preprocess=True,
        mean_color=True,
    ):
        self.source_files = source_files
        self.target_folder = target_folder
        self.target_json = target_json
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.mean_color = mean_color


class JSONProcessor:
    def __init__(self, config):
        self.config = config

    def read_json_files(self):
        data = []
        for file_path in self.config.source_files:
            with open(file_path, "r", encoding="utf-8") as file:
                data.extend(json.load(file)["ocr_files"])
        return data

    def normalize_path(self, path):
        return path.replace("\\", "/")

    def preprocess_image(self, image, target_size=(200, 50), threshold=170):

        if target_size == image.shape[:2]:
            return image

        # Преобразовать в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Найти пиксели, которые ярче порога
        mask = gray > threshold

        # Вычислить средний цвет топ 20% самых светлых пикселей
        if self.config.mean_color and np.any(mask):
            bright_pixels = image[mask]
            bright_pixels = bright_pixels.reshape(-1, 3)

            # Сортируем пиксели по яркости
            brightness = np.mean(bright_pixels, axis=1)
            sorted_indices = np.argsort(brightness)[
                ::-1
            ]  # от самого светлого к самому темному

            # Берем топ 20% самых светлых пикселей
            top_20_percent = bright_pixels[
                sorted_indices[: max(1, int(0.15 * len(sorted_indices)))]
            ]

            # Вычисляем средний цвет
            mean_color = np.mean(top_20_percent, axis=0)
            if np.isnan(mean_color).any():
                pad_color = (
                    181,
                    181,
                    181,
                )  # дефолтный цвет, если вычисление среднего цвета не удалось
            else:
                pad_color = (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))
        else:
            pad_color = (181, 181, 181)  # дефолтный цвет

        # Получить размеры исходного изображения
        old_size = image.shape[:2]  # (height, width)

        # Проверить, если изображение больше целевого размера, изменить его размер с сохранением пропорций
        if old_size[0] > target_size[1] or old_size[1] > target_size[0]:
            ratio = min(target_size[1] / old_size[0], target_size[0] / old_size[1])
            new_size = (
                int(old_size[1] * ratio),
                int(old_size[0] * ratio),
            )  # (width, height)
            image = cv2.resize(image, (new_size[0], new_size[1]))
            old_size = image.shape[:2]  # обновить размеры после изменения размера

        # Создать новое изображение с целевым размером и фоновым цветом
        new_image = np.full(
            (target_size[1], target_size[0], 3), pad_color, dtype=np.uint8
        )

        # Рассчитать смещение для выравнивания по левому краю
        y_offset = (target_size[1] - old_size[0]) // 2
        x_offset = (
            0  # смещение по горизонтали равно нулю для выравнивания по левому краю
        )

        # Вставить исходное изображение в новое изображение
        new_image[
            y_offset : y_offset + old_size[0], x_offset : x_offset + old_size[1]
        ] = image

        return new_image

    def copy_files_and_update_paths(self, data):
        if not os.path.exists(self.config.target_folder):
            os.makedirs(self.config.target_folder)

        index = 0
        result = []
        for entry in tqdm(data, desc="Копирование файлов"):
            source_file_path = entry["file"]
            if not os.path.exists(source_file_path):
                continue

            filename = f"{index:06d}.png"
            target_file_path = os.path.join(self.config.target_folder, filename)
            target_file_path = self.normalize_path(target_file_path)

            if self.config.preprocess:
                image = cv2.imread(source_file_path)
                image = self.preprocess_image(image)
                cv2.imwrite(target_file_path, image)
            else:
                shutil.copy(source_file_path, target_file_path)

            new_entry = {
                "file": target_file_path,
                "text": entry["text"],
            }
            result.append(new_entry)
            index += 1

        return result

    def save_json(self, data):
        with open(self.config.target_json, "w", encoding="utf-8") as file:
            json.dump({"ocr_files": data}, file, ensure_ascii=False, indent=4)

    def process(self):
        data = self.read_json_files()

        updated_data = self.copy_files_and_update_paths(data)
        if self.config.shuffle:
            random.shuffle(updated_data)

        self.save_json(updated_data)


def run(**kwargs):
    config = Config(**kwargs)
    processor = JSONProcessor(config)
    processor.process()


def run_config(config):
    processor = JSONProcessor(config)
    processor.process()
    return processor


def main():
    config = Config()
    base = "c:/proplex"
    config.source_files = [
        # f"{base}/synth/ocr.json",
        # f"{base}/label/ocr.json",
        # f"{base}/label1/ocr.json",
        # f"{base}/label2/ocr.json",
        # f"{base}/label3/ocr.json",
        # f"{base}/label4/ocr.json",
        # "/content/ocr.json",
        "/cluster/ocr.json",
        "/content/synth2/ocr.json",
    ]
    config.target_folder = f"/content/ocr012"
    config.target_json = f"/content/ocr012.json"
    config.shuffle = True
    config.preprocess = False
    config.mean_color = True

    parser = argparse.ArgumentParser(
        description="Утилита для обработки JSON файлов и копирования изображений."
    )

    parser.add_argument(
        "--source_files",
        nargs="+",
        default=config.source_files,
        help="Список исходных JSON файлов",
    )
    parser.add_argument(
        "--target_folder",
        default=config.target_folder,
        help="Папка для сохранения копий файлов",
    )
    parser.add_argument(
        "--target_json",
        default=config.target_json,
        help="Выходной результирующий JSON файл",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=config.shuffle,
        help="Перемешать результаты",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))
    run_config(config)


if __name__ == "__main__":
    main()
