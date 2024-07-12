import argparse
import os
import json
from pathlib import Path
import shutil
import fnmatch
from PIL import Image


class Config:
    def __init__(self):
        self.source_folder = "c:/proplex/labels.florence"
        self.target_folder = "c:/proplex/labels.new"
        self.label_filter = ["text label", "inscription"]
        self.extract = True


def scan_and_copy_files(config):
    source_folder = Path(config.source_folder)
    target_folder = Path(config.target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            if fnmatch.fnmatch(file_name, "*.json"):
                if file_name == "results.json":
                    continue

                json_file_path = Path(root) / file_name
                with open(json_file_path, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                        florence_results = data.get("florence_results")
                        if not florence_results:
                            print(
                                f"Нет данных 'florence_results' в файле: {json_file_path}"
                            )
                            break

                        first_key = next(iter(florence_results))
                        result_data = florence_results.get(first_key)
                        if (
                            not isinstance(result_data, dict)
                            or "labels" not in result_data
                        ):
                            print(f"Неверный формат JSON в файле: {json_file_path}")
                            break

                        labels = result_data["labels"]
                        bboxes = result_data["bboxes"]
                        original_file_name = data["file"]
                        for index, label in enumerate(labels):
                            if any(f in label for f in config.label_filter):
                                image_file_name = file_name.replace(
                                    ".json", f".florence.{index:03}.png"
                                )
                                image_file_path = source_folder / image_file_name
                                if image_file_path.exists():
                                    if config.extract and bboxes:
                                        bbox = bboxes[index]
                                        extract_and_save_image(
                                            original_file_name,
                                            bbox,
                                            target_folder
                                            / image_file_name.replace(
                                                ".png", ".bbox.png"
                                            ),
                                        )
                                    shutil.copy(
                                        image_file_path,
                                        target_folder / image_file_name,
                                    )
                                    print(f"Скопировано: {image_file_name}")
                                else:
                                    print(
                                        f"Файл изображения не найден: {image_file_path}"
                                    )
                    except json.JSONDecodeError:
                        print(f"Ошибка чтения JSON файла: {json_file_path}")


def extract_and_save_image(image_path, bbox, target_path):
    with Image.open(image_path) as img:
        left, top, right, bottom = bbox
        cropped_img = img.crop((left, top, right, bottom))
        cropped_img.save(target_path)


if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для копирования файлов на основе JSON меток"
    )

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к исходной папке",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к папке для сохранения скопированных файлов",
    )
    parser.add_argument(
        "--label_filter",
        nargs="+",
        default=config.label_filter,
        help="Список фильтров для меток",
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        default=config.extract,
        help="Вырезать изображения по bounding box",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    scan_and_copy_files(config)
