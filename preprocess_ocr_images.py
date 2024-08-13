import os
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
        source_folder="",
        target_folder="",
        shuffle=False,
        preprocess=True,
        mean_color=True,
        resize_height=50,
        resize_width=200,
        padding_color=(181, 181, 181),
        binary_threshold=True,
        remove_noise=True,
        remove_borders=True,
        thin_font=False,
        thick_font=False,
        deskew=False,
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.shuffle = shuffle
        self.preprocess = preprocess
        self.mean_color = mean_color
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.padding_color = padding_color
        self.binary_threshold = binary_threshold
        self.remove_noise = remove_noise
        self.remove_borders = remove_borders
        self.thin_font = thin_font
        self.thick_font = thick_font
        self.deskew = deskew


class ImageProcessor:
    def __init__(self, config):
        self.config = config

    def process_images(self):
        source_folder = Path(self.config.source_folder).expanduser()
        target_folder = Path(self.config.target_folder).expanduser()

        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)

        image_files = list(source_folder.glob("*.jpg")) + list(
            source_folder.glob("*.png")
        )

        if self.config.shuffle:
            random.shuffle(image_files)

        for image_file in tqdm(
            image_files, desc=f"Обработка изображений из {source_folder}"
        ):
            img = cv2.imread(str(image_file))

            if img is None:
                print(f"Не удалось загрузить изображение: {image_file}")
                continue

            if self.config.deskew:
                img = self.deskew(img)

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

            if self.config.thin_font:
                im_bw = self.thin_font(im_bw)

            if self.config.thick_font:
                im_bw = self.thick_font(im_bw)

            if self.config.preprocess:
                im_bw = self.preprocess_image(im_bw)

            output_file = target_folder / image_file.name
            rgb_image = cv2.cvtColor(im_bw, cv2.COLOR_GRAY2RGB)
            cv2.imwrite(str(output_file), rgb_image)

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

    def thin_font(self, image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.erode(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def thick_font(self, image):
        image = cv2.bitwise_not(image)
        kernel = np.ones((2, 2), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        image = cv2.bitwise_not(image)
        return image

    def deskew(self, image):
        angle = self.get_skew_angle(image)
        return self.rotate_image(image, -1.0 * angle)

    def get_skew_angle(self, cvImage) -> float:
        newImage = cvImage.copy()
        gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 0)
        thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilate = cv2.dilate(thresh, kernel, iterations=2)
        contours, _ = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largestContour = contours[0]
        minAreaRect = cv2.minAreaRect(largestContour)
        angle = minAreaRect[-1]
        if angle < -45:
            angle = 90 + angle
        return -1.0 * angle

    def rotate_image(self, cvImage, angle: float):
        newImage = cvImage.copy()
        (h, w) = newImage.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        newImage = cv2.warpAffine(
            newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return newImage


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = ImageProcessor(config)
    processor.process_images()
    return processor


def main():
    config = Config()
    config.source_folder = "/proplex/label1/yolo"
    config.target_folder = "/proplex/label1/yolo_processed"
    config.preprocess = False
    config.shuffle = True
    config.mean_color = True
    config.resize_height = 50
    config.resize_width = 200
    config.padding_color = (181, 181, 181)
    config.binary_threshold = True
    config.remove_noise = True
    config.remove_borders = True
    config.thin_font = False
    config.thick_font = False
    config.deskew = True
    parser = argparse.ArgumentParser(description="Утилита для обработки изображений.")

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Папка с исходными изображениями",
    )
    parser.add_argument(
        "--target_folder",
        default=config.target_folder,
        type=str,
        help="Папка для сохранения обработанных изображений",
    )
    parser.add_argument(
        "--shuffle",
        default=config.shuffle,
        action="store_true",
        help="Перемешать изображения перед обработкой",
    )
    parser.add_argument(
        "--preprocess",
        default=config.preprocess,
        action="store_true",
        help="Включить предобработку изображений (изменение размера и выравнивание)",
    )
    parser.add_argument(
        "--mean_color",
        default=config.mean_color,
        action="store_true",
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
        default=config.binary_threshold,
        action="store_true",
        help="Применить бинарное пороговое значение к изображениям",
    )
    parser.add_argument(
        "--remove_noise",
        default=config.remove_noise,
        action="store_true",
        help="Удалить шум с изображений",
    )
    parser.add_argument(
        "--remove_borders",
        default=config.remove_borders,
        action="store_true",
        help="Удалить границы с изображений",
    )
    parser.add_argument(
        "--thin_font",
        default=config.thin_font,
        action="store_true",
        help="Утоньшение шрифта на изображениях",
    )
    parser.add_argument(
        "--thick_font",
        default=config.thick_font,
        action="store_true",
        help="Утолщение шрифта на изображениях",
    )
    parser.add_argument(
        "--deskew",
        default=config.deskew,
        action="store_true",
        help="Исправление наклона изображения",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
