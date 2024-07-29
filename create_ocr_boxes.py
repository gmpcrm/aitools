import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm


class Config:
    def __init__(
        self,
        source_folder="~/data/source",
        target_folder="~/data/target",
        ocr="easyocr",
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.ocr = ocr


class JSONProcessor:
    def __init__(self, config):
        self.config = config

    def load_json(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)

    def process_files(self):
        source_folder = Path(self.config.source_folder).expanduser()
        target_folder = Path(self.config.target_folder).expanduser()

        if not target_folder.exists():
            target_folder.mkdir(parents=True, exist_ok=True)

        for file_name in tqdm(os.listdir(source_folder), desc="Обработка файлов"):
            if file_name.endswith(".ocr.json"):
                child_json_path = source_folder / file_name
                child_json_data = self.load_json(child_json_path)

                # Извлечь путь к primary JSON из child JSON
                original_image_path = child_json_data["file"]
                primary_json_path = original_image_path.replace(
                    ".yolo.000.png", ".json"
                )
                primary_json_data = self.load_json(primary_json_path)
                original_image_path = primary_json_data["file"]

                # Извлечь bounding box из primary JSON
                primary_bbox = primary_json_data["yolo"][0]["bbox"]
                x_offset, y_offset, box_width, box_height = primary_bbox

                # Рассчитать корректированные bounding boxes
                corrected_bboxes = []
                for item in child_json_data[self.config.ocr]:
                    relative_bbox = item["bbox"]
                    absolute_bbox = [
                        x_offset + relative_bbox[0],
                        y_offset + relative_bbox[1],
                        x_offset + relative_bbox[2],
                        y_offset + relative_bbox[3],
                    ]
                    corrected_bboxes.append(absolute_bbox)

                # Создать результатирующий словарь
                result = {
                    "width": primary_json_data["width"],
                    "height": primary_json_data["height"],
                    "sliced_yolo_boxes": corrected_bboxes,
                    "file": original_image_path,
                }

                # Сохранить результат в целевой папке
                result_file_path = target_folder / (
                    file_name.replace(".json", ".box.json")
                )
                with open(result_file_path, "w", encoding="utf-8") as result_file:
                    json.dump(result, result_file, ensure_ascii=False, indent=4)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = JSONProcessor(config)
    processor.process_files()
    return processor


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для обработки JSON файлов и сохранения результатов."
    )

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к папке с исходными JSON файлами.",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к папке для сохранения результатов.",
    )
    parser.add_argument(
        "--ocr",
        type=str,
        default=config.ocr,
        help="Тип OCR модели: easyocr или tesseract.",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
