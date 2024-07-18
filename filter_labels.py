import argparse
import os
import json
from pathlib import Path
import shutil
import fnmatch
from PIL import Image
from tqdm import tqdm


class Config:
    def __init__(
        self,
        source_folder="~/data/labels.florence",
        target_folder="~/data/labels.new",
        bbox_folder="~/data/labels.bbox",
        label_filter=["extract inscription text"],
        filter_match="substr",
        extract_bbox=True,
        save_json=False,
        query="<CAPTION_TO_PHRASE_GROUNDING>",
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.label_filter = label_filter
        self.extract_bbox = extract_bbox
        self.save_json = save_json
        self.bbox_folder = bbox_folder
        self.query = query
        self.filter_match = filter_match


class LabelFileProcessor:
    def __init__(self, config):
        self.config = config
        self.source_folder = Path(config.source_folder).expanduser()
        self.target_folder = Path(config.target_folder).expanduser()
        self.target_folder.mkdir(parents=True, exist_ok=True)
        if self.config.bbox_folder:
            self.bbox_folder = Path(config.bbox_folder).expanduser()
            self.bbox_folder.mkdir(parents=True, exist_ok=True)

    def scan_and_copy_files(self):
        json_files = []
        for root, _, files in os.walk(self.source_folder):
            for file_name in files:
                if fnmatch.fnmatch(file_name, "*.json") and file_name != "results.json":
                    json_files.append(Path(root) / file_name)

        for json_file_path in tqdm(json_files, desc="Processing JSON files"):
            with open(json_file_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    florence_results = data.get("florence_results")
                    if not florence_results:
                        print(
                            f"Нет данных 'florence_results' в файле: {json_file_path}"
                        )
                        continue

                    result_data = florence_results[0].get(self.config.query)
                    if not isinstance(result_data, dict) or "labels" not in result_data:
                        print(f"Неверный формат JSON в файле: {json_file_path}")
                        continue

                    labels = result_data["labels"]
                    bboxes = result_data["bboxes"]
                    yolo_box = data["yolo_box"]
                    original_file_name = data["file"]

                    if not os.path.exists(original_file_name):
                        original_file_name = (
                            self.source_folder
                            / json_file_path.name.replace(".json", ".png")
                        )
                    if not os.path.exists(original_file_name):
                        print(f"Файл изображения не найден: {original_file_name}")
                        continue

                    for index, label in enumerate(labels):
                        if any(
                            (
                                f in label
                                if self.config.filter_match == "substr"
                                else f == label
                            )
                            for f in self.config.label_filter
                        ):
                            image_file_name = json_file_path.name.replace(
                                ".json", f".florence.{index:03}.png"
                            )
                            image_file_path = self.source_folder / image_file_name
                            if image_file_path.exists():
                                if self.config.extract_bbox and bboxes:
                                    bbox = bboxes[index]
                                    bbox_target_folder = (
                                        self.bbox_folder
                                        if self.config.bbox_folder
                                        else self.target_folder
                                    )
                                    bbox[0] += yolo_box[0]
                                    bbox[1] += yolo_box[1]
                                    bbox[2] += yolo_box[0]
                                    bbox[3] += yolo_box[1]
                                    self.extract_and_save_image(
                                        original_file_name,
                                        bbox,
                                        bbox_target_folder
                                        / image_file_name.replace(".png", ".bbox.png"),
                                    )
                                shutil.copy(
                                    image_file_path,
                                    self.target_folder / image_file_name,
                                )
                                if self.config.save_json:
                                    json_target_path = (
                                        self.target_folder
                                        / image_file_name.replace(".png", ".json")
                                    )
                                    self.save_filtered_json(
                                        data, json_target_path, index
                                    )
                                # print(f"Скопировано: {image_file_name}")
                            else:
                                print(f"Файл изображения не найден: {image_file_path}")
                except json.JSONDecodeError:
                    print(f"Ошибка чтения JSON файла: {json_file_path}")

    def extract_and_save_image(self, image_path, bbox, target_path):
        with Image.open(image_path) as img:
            left, top, right, bottom = bbox
            cropped_img = img.crop((left, top, right, bottom))
            cropped_img.save(target_path)

    def save_filtered_json(self, data, json_target_path, index):
        query_data = data["florence_results"][0][self.config.query]
        filtered_data = {
            "width": data.get("width"),
            "height": data.get("height"),
            "yolo_box": data.get("yolo_box"),
            "yolo_confidence": data.get("yolo_confidence"),
            "florence_results": {
                self.config.query: {
                    "bboxes": [query_data["bboxes"][index]],
                    "labels": [query_data["labels"][index]],
                }
            },
            "file": data["file"],
        }
        with open(json_target_path, "w", encoding="utf-8") as json_out:
            json.dump(filtered_data, json_out, ensure_ascii=False, indent=4)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = LabelFileProcessor(config)
    processor.scan_and_copy_files()
    return processor


def main():
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
        "--extract_bbox",
        action="store_true",
        default=config.extract_bbox,
        help="Вырезать изображения по bounding box",
    )
    parser.add_argument(
        "--bbox_folder",
        default=config.bbox_folder,
        help="Путь к папке для сохранения вырезанных изображений",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        default=config.save_json,
        help="Сохранять JSON данные для каждой картинки",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=config.query,
        help="Ключ для получения данных из florence",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
