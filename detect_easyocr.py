import argparse
import easyocr
import cv2
import json
import os
from pathlib import Path
from tqdm import tqdm
import time


class Config:
    def __init__(
        self,
        source_folder="~/data/source/",
        target_folder="~/data/target/",
        draw_boxes=False,
        width_ths=0.01,
        add_margin=0.05,
        save_boxes=False,
        measure_time=False,
        confidence=0.4,
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.draw_boxes = draw_boxes
        self.width_ths = width_ths
        self.add_margin = add_margin
        self.save_boxes = save_boxes
        self.measure_time = measure_time
        self.confidence = confidence


class OCRProcessor:
    def __init__(self, config):
        self.config = config
        self.prefix = "ocr"
        self.reader = easyocr.Reader(["en", "ru"])
        self.source_folder = Path(config.source_folder).expanduser()
        self.target_folder = Path(config.target_folder).expanduser()

    def process_images(self):
        image_files = list(self.source_folder.glob("**/*.png")) + list(
            self.source_folder.glob("**/*.jpg")
        )
        for image_file in tqdm(image_files, desc="Обработка изображений"):
            self.process_image(image_file)

    def process_image(self, image_path):
        source = cv2.imread(str(image_path))
        height, width = source.shape[:2]

        start_time = time.time()

        results = self.reader.readtext(
            source,
            width_ths=self.config.width_ths,
            add_margin=self.config.add_margin,
            detail=1,
            paragraph=False,
        )

        elapsed_time = time.time() - start_time

        filtered_results = [
            result for result in results if result[2] >= self.config.confidence
        ]

        results_json = {
            "width": int(width),
            "height": int(height),
            "easyocr": [
                {
                    "text": result[1].strip(),
                    "confidence": float(result[2]),
                    "bbox": self.quad_to_bbox(result[0]),
                }
                for result in filtered_results
            ],
            "file": str(image_path),
        }

        if self.config.measure_time:
            results_json["time"] = elapsed_time

        self.save_results(image_path, results_json)

        if self.config.draw_boxes:
            self.draw_boxes(image_path, filtered_results)

        if self.config.save_boxes:
            self.save_boxes(image_path, filtered_results)

    def quad_to_bbox(self, quad):
        if len(quad) != 4:
            raise ValueError("Quadrilateral must have exactly 4 points")
        x_coords = [float(point[0]) for point in quad]
        y_coords = [float(point[1]) for point in quad]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    def save_results(self, image_path, results_json):
        output_dir = self.target_folder / image_path.parent.relative_to(
            self.source_folder
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{image_path.stem}.{self.prefix}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_json, f, ensure_ascii=False, indent=4)
        # print(f"Результаты сохранены в {output_file}")

    def draw_boxes(self, image_path, results):
        image = cv2.imread(str(image_path))
        for result in results:
            top_left = tuple(map(int, result[0][0]))
            bottom_right = tuple(map(int, result[0][2]))
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
        output_image_path = (
            self.target_folder / f"{image_path.stem}.{self.prefix}.box.png"
        )
        cv2.imwrite(str(output_image_path), image)
        # print(f"Изображение с рамками сохранено в {output_image_path}")

    def save_boxes(self, image_path, results):
        image = cv2.imread(str(image_path))
        for idx, result in enumerate(results):
            bbox = self.quad_to_bbox(result[0])
            x_min, y_min, x_max, y_max = map(int, bbox)
            extracted_image = image[y_min:y_max, x_min:x_max]
            output_image_path = (
                self.target_folder / f"{image_path.stem}.{self.prefix}.{idx:03d}.png"
            )
            cv2.imwrite(str(output_image_path), extracted_image)
            # print(f"Извлеченный бокс сохранен в {output_image_path}")


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = OCRProcessor(config)
    processor.process_images()
    return processor


def main():
    config = Config()

    parser = argparse.ArgumentParser(
        description="Утилита для распознавания текста с изображений с использованием EasyOCR"
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
        help="Рисовать рамки вокруг обнаруженного текста",
    )
    parser.add_argument(
        "--width_ths",
        type=float,
        default=config.width_ths,
        help="Порог ширины для EasyOCR",
    )
    parser.add_argument(
        "--add_margin",
        type=float,
        default=config.add_margin,
        help="Порог отступов вокруг текста для EasyOCR",
    )
    parser.add_argument(
        "--save_boxes",
        action="store_true",
        help="Извлекать и сохранять боксы с текстом",
    )
    parser.add_argument(
        "--measure_time",
        action="store_true",
        help="Засекать время выполнения EasyOCR и записывать его в JSON",
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
