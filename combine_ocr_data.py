import os
import json
import shutil
import random
import argparse
import cv2
import numpy as np
from pathlib import Path
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
        resize_height=50,
        resize_width=200,
        padding_color=(181, 181, 181),
        binary_threshold=True,
        remove_noise=True,
        remove_borders=True,
    ):
        self.source_files = source_files
        self.target_folder = target_folder
        self.target_json = target_json
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.mean_color = mean_color
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.padding_color = padding_color
        self.binary_threshold = binary_threshold
        self.remove_noise = remove_noise
        self.remove_borders = remove_borders


class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.result_data = []

    def normalize_path(self, path):
        return path.replace("\\", "/")

    def process_images(self):
        target_folder = Path(self.config.target_folder).expanduser()

        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)

        for file_path in self.config.source_files:
            json_path = Path(file_path).expanduser()

            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for files in tqdm(
                data["ocr_files"], desc=f"Обработка изображений из {file_path}"
            ):
                image_file = Path(files["file"])
                img = cv2.imread(str(image_file))

                if img is None:
                    print(f"Не удалось загрузить изображение: {image_file}")
                    continue

                gray_image = self.grayscale(img)
                if self.config.binary_threshold:
                    lower_thresh, upper_thresh = self.calculate_dynamic_thresholds(img)
                    _, im_bw = cv2.threshold(
                        gray_image, lower_thresh, upper_thresh, cv2.THRESH_BINARY
                    )
                else:
                    im_bw = gray_image

                if self.config.remove_noise:
                    im_bw = self.noise_removal(im_bw)

                if self.config.remove_borders:
                    im_bw = self.remove_borders_func(im_bw)

                if self.config.preprocess:
                    im_bw = self.preprocess_image(im_bw)

                output_file = target_folder / image_file.name
                rgb_image = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)
                cv2.imwrite(str(output_file), rgb_image)

                self.result_data.append(
                    {
                        "file": self.normalize_path(str(output_file)),
                        "text": files["text"],
                    }
                )

        if self.config.shuffle:
            random.shuffle(self.result_data)

        self.save_json(self.result_data)

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def calculate_dynamic_thresholds(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        max_intensity_value = np.argmax(hist)
        std_dev = np.std(gray_image)
        lower_thresh = max_intensity_value - std_dev
        upper_thresh = max_intensity_value + std_dev
        return int(lower_thresh), int(upper_thresh)

    def noise_removal(self, image):
        kernel = np.ones((1, 1), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        image = cv2.medianBlur(image, 3)
        return image

    def remove_borders_func(self, image):
        contours, _ = cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
        cnt = cntsSorted[-1]
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y : y + h, x : x + w]
        return crop

    def preprocess_image(self, image):
        target_size = (self.config.resize_width, self.config.resize_height)
        threshold = 170

        if target_size == image.shape[:2]:
            return image

        mask = image > threshold

        if self.config.mean_color and np.any(mask):
            bright_pixels = image[mask]

            brightness = np.mean(bright_pixels, axis=0)
            sorted_indices = np.argsort(brightness)[::-1]

            top_20_percent = bright_pixels[
                sorted_indices[: max(1, int(0.15 * len(sorted_indices)))]
            ]

            mean_color = np.mean(top_20_percent)
            if np.isnan(mean_color):
                pad_color = self.config.padding_color[0]
            else:
                pad_color = int(mean_color)
        else:
            pad_color = self.config.padding_color[0]

        old_size = image.shape[:2]

        if old_size[0] > target_size[1] or old_size[1] > target_size[0]:
            ratio = min(target_size[1] / old_size[0], target_size[0] / old_size[1])
            new_size = (
                int(old_size[1] * ratio),
                int(old_size[0] * ratio),
            )
            image = cv2.resize(image, (new_size[0], new_size[1]))
            old_size = image.shape[:2]

        new_image = np.full((target_size[1], target_size[0]), pad_color, dtype=np.uint8)

        y_offset = (target_size[1] - old_size[0]) // 2
        x_offset = 0

        new_image[
            y_offset : y_offset + old_size[0], x_offset : x_offset + old_size[1]
        ] = image

        return new_image

    def save_json(self, data):
        with open(self.config.target_json, "w", encoding="utf-8") as file:
            json.dump({"ocr_files": data}, file, ensure_ascii=False, indent=4)


def run(**kwargs):
    config = Config(**kwargs)
    processor = ImageProcessor(config)
    processor.process_images()


def run_config(config):
    processor = ImageProcessor(config)
    processor.process_images()
    return processor


def main():
    config = Config()

    base = "c:/proplex"
    config.source_files = [
        # f"{base}/synth/ocr.json",
        # f"{base}/label/ocr.json",
        # f"{base}/label1/ocr.json",
        f"{base}/label2/ocr.json",
        # f"{base}/label3/ocr.json",
        # f"{base}/label4/ocr.json",
    ]

    config.target_folder = f"/content/ocr"
    config.target_json = f"/content/ocr.json"
    config.shuffle = True
    config.resize_height = 50
    config.resize_width = 200
    config.padding_color = (181, 181, 181)
    config.binary_threshold = False
    config.remove_noise = False
    config.remove_borders = False
    config.preprocess = True
    config.mean_color = False

    parser = argparse.ArgumentParser(
        description="Утилита для обработки изображений и JSON файлов."
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
        type=str,
        help="Папка для сохранения обработанных изображений",
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
    parser.add_argument(
        "--preprocess",
        action="store_true",
        default=config.preprocess,
        help="Включить предобработку изображений (изменение размера и выравнивание)",
    )
    parser.add_argument(
        "--mean_color",
        action="store_true",
        default=config.mean_color,
        help="Использовать средний цвет самых ярких пикселей для заливки фона",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=config.resize_height,
        help="Высота целевого изображения после изменения размера",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=config.resize_width,
        help="Ширина целевого изображения после изменения размера",
    )
    parser.add_argument(
        "--padding_color",
        nargs=3,
        type=int,
        default=config.padding_color,
        help="Цвет заливки в формате RGB для фона, если не используется mean_color",
    )
    parser.add_argument(
        "--binary_threshold",
        action="store_true",
        default=config.binary_threshold,
        help="Применить бинарное пороговое значение к изображениям",
    )
    parser.add_argument(
        "--remove_noise",
        action="store_true",
        default=config.remove_noise,
        help="Удалить шум с изображений",
    )
    parser.add_argument(
        "--remove_borders",
        action="store_true",
        default=config.remove_borders,
        help="Удалить границы с изображений",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
