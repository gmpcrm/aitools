import argparse
import pytesseract
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class Config:
    def __init__(
        self,
        source_folder="~/data/source/",
        target_folder="~/data/target/",
        draw_boxes=False,
        lang="rus+eng",
        save_boxes=False,
        measure_time=False,
        scale=1.0,
        confidence=0.4,
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.draw_boxes = draw_boxes
        self.lang = lang
        self.save_boxes = save_boxes
        self.measure_time = measure_time
        self.confidence = confidence
        self.scale = scale


class OCRProcessor:
    def __init__(self, config):
        self.config = config
        self.prefix = "ocr"
        self.confidence = int(config.confidence * 100)
        self.source_folder = Path(config.source_folder).expanduser()
        self.target_folder = Path(config.target_folder).expanduser()

    def enhance_image(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)

        binary_image = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        return morphed_image

    def process_images(self):
        image_files = list(self.source_folder.glob("**/*.png")) + list(
            self.source_folder.glob("**/*.jpg")
        )
        for image_file in tqdm(image_files, desc="Обработка изображений"):
            self.process_image(image_file)

    def process_image(self, image_path):
        img = cv2.imread(str(image_path))
        source = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # source = self.enhance_image(source)
        height, width = source.shape[:2]

        if self.config.scale != 1.0:
            height = int(height * self.config.scale)
            width = int(width * self.config.scale)
            source = cv2.resize(
                source,
                (width, height),
                interpolation=cv2.INTER_CUBIC,
            )
        start_time = time.time()

        results = pytesseract.image_to_data(
            source, lang=self.config.lang, output_type=pytesseract.Output.DICT
        )
        elapsed_time = time.time() - start_time

        filtered_results = [
            {key: results[key][i] for key in results}
            for i in range(len(results["text"]))
            if int(results["conf"][i]) >= self.confidence
            and results["text"][i].strip() != ""
        ]

        if filtered_results:
            results_json = {
                "width": int(width),
                "height": int(height),
                "tesseract": [
                    {
                        "text": result["text"].strip(),
                        "confidence": int(result["conf"]) / 100,
                        "bbox": [
                            result["left"],
                            result["top"],
                            result["left"] + result["width"],
                            result["top"] + result["height"],
                        ],
                    }
                    for result in filtered_results
                ],
                "file": str(image_path),
            }

            if self.config.measure_time:
                results_json["time"] = elapsed_time

            if self.config.scale != 1.0:
                results_json["scale"] = self.config.scale

            self.save_results(image_path, results_json)

            if self.config.draw_boxes:
                self.draw_boxes(image_path, filtered_results)

            if self.config.save_boxes:
                self.save_boxes(image_path, filtered_results)

    def save_results(self, image_path, results_json):
        output_dir = self.target_folder / image_path.parent.relative_to(
            self.source_folder
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{image_path.stem}.{self.prefix}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=4)

    def draw_boxes(self, image_path, results):
        image = cv2.imread(str(image_path))
        for result in results:
            top_left = (result["left"], result["top"])
            bottom_right = (
                result["left"] + result["width"],
                result["top"] + result["height"],
            )
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        output_image_path = (
            self.target_folder / f"{image_path.stem}.{self.prefix}.box.png"
        )
        cv2.imwrite(str(output_image_path), image)

    def save_boxes(self, image_path, results):
        image = cv2.imread(str(image_path))
        for idx, result in enumerate(results):
            x_min, y_min, x_max, y_max = (
                result["left"],
                result["top"],
                result["left"] + result["width"],
                result["top"] + result["height"],
            )
            extracted_image = image[y_min:y_max, x_min:x_max]
            if extracted_image.size == 0:  # Check if the extracted image is empty
                continue
            output_image_path = (
                self.target_folder / f"{image_path.stem}.{self.prefix}.{idx:03d}.png"
            )
            cv2.imwrite(str(output_image_path), extracted_image)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = OCRProcessor(config)
    processor.process_images()
    return processor


def main():
    config = Config()
    config.source_folder = "c:/proplex/label3/yolo"
    config.target_folder = "c:/proplex/label3/ocr"
    config.confidence = 0.6
    config.draw_boxes = True
    config.save_boxes = True
    config.measure_time = True

    parser = argparse.ArgumentParser(
        description="Утилита для распознавания текста с изображений с использованием Tesseract OCR"
    )
    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Папка с исходными изображениями",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Папка для сохранения результатов",
    )
    parser.add_argument(
        "--draw_boxes",
        action="store_true",
        default=config.draw_boxes,
        help="Рисовать рамки вокруг обнаруженного текста",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=config.lang,
        help="Языки для распознавания, например 'eng' или 'rus'. Для нескольких языков используется '+': rus+eng",
    )
    parser.add_argument(
        "--save_boxes",
        action="store_true",
        default=config.save_boxes,
        help="Извлекать и сохранять боксы с текстом",
    )
    parser.add_argument(
        "--measure_time",
        action="store_true",
        default=config.measure_time,
        help="Засекать время выполнения Tesseract OCR и записывать его в JSON",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.confidence,
        help="Минимальный уровень уверенности для сохранения результатов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
