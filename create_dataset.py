import argparse
import json
import os
from pathlib import Path
import re
from PIL import Image
from tqdm import tqdm
import random


class Config:
    def __init__(
        self,
        source_folder="~/data/florence",
        data_folder="~/data/florence.new",
        dataset_folder="~/data/dataset",
        valid=20,  # Default validation percentage
        class_id=0,  # Default class ID for YOLO
        shuffle=False,  # Default shuffle option
    ):
        self.source_folder = source_folder
        self.data_folder = data_folder
        self.dataset_folder = dataset_folder
        self.valid = valid
        self.class_id = class_id
        self.shuffle = shuffle


class YOLODatasetCreator:
    def __init__(self, config):
        self.config = config

    def convert_bbox_to_yolo(self, size, bbox):
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        x_center = (bbox[0] + bbox[2]) / 2.0
        y_center = (bbox[1] + bbox[3]) / 2.0
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        x_center = x_center * dw
        width = width * dw
        y_center = y_center * dh
        height = height * dh
        return (x_center, y_center, width, height)

    def save_yolo_bbox(self, label_file, data, bbox):
        if "yolo_box" in data:
            yolo_box = data["yolo_box"]
            bbox = [
                bbox[0] + yolo_box[0],
                bbox[1] + yolo_box[1],
                bbox[2] + yolo_box[0],
                bbox[3] + yolo_box[1],
            ]

        yolo_bbox = self.convert_bbox_to_yolo((data["width"], data["height"]), bbox)

        with open(label_file, "a", encoding="utf-8") as label_file:
            label_file.write(
                f"{self.config.class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
            )

    def create_dataset(self):
        dataset_folder = Path(self.config.dataset_folder).expanduser()
        train_images_folder = dataset_folder / "train" / "images"
        train_labels_folder = dataset_folder / "train" / "labels"
        valid_images_folder = dataset_folder / "valid" / "images"
        valid_labels_folder = dataset_folder / "valid" / "labels"

        train_images_folder.mkdir(parents=True, exist_ok=True)
        train_labels_folder.mkdir(parents=True, exist_ok=True)
        valid_images_folder.mkdir(parents=True, exist_ok=True)
        valid_labels_folder.mkdir(parents=True, exist_ok=True)

        file_patterns = [
            re.compile(r"^(.*)\.florence\.(\d{3})\.png$"),
            re.compile(r"^(.*)\.yolo\.(\d{3})\.png$"),
            re.compile(r"^(.*)\.(\d{3})\.png$"),
            re.compile(r"^(.*\.ocr\.box)\.json$"),
        ]

        counter = 0
        valid_interval = 100 // self.config.valid if self.config.valid > 0 else 0

        files = [
            f
            for f in os.listdir(Path(self.config.data_folder).expanduser())
            if f.endswith(".png")
        ]

        if not files:
            files = [
                f
                for f in os.listdir(Path(self.config.data_folder).expanduser())
                if f.endswith(".json")
            ]

        if self.config.shuffle:
            random.shuffle(files)

        for file_name in tqdm(files, desc="Processing files"):
            match = None
            for pattern in file_patterns:
                match = pattern.match(file_name)
                if match:
                    break
            if not match:
                continue

            original_name = match.group(1)
            index_str = match.group(2) if len(match.groups()) > 1 else "0"
            index = int(index_str)

            json_file_name = f"{original_name}.json"
            json_file_path = (
                Path(self.config.source_folder).expanduser() / json_file_name
            )

            original_img_path = (
                Path(self.config.source_folder).expanduser() / f"{original_name}.png"
            )

            if not json_file_path.exists():
                print(f"JSON файл {json_file_path} не найден")
                continue

            with open(json_file_path, "r", encoding="utf-8") as json_file:
                data = json.load(json_file)

            if "file" in data and os.path.exists(data["file"]):
                original_img_path = Path(data["file"])

            if not original_img_path.exists():
                print(f"Файл {original_img_path} не найден")
                continue

            img = Image.open(original_img_path)
            output_img_path = train_images_folder / f"{original_name}.jpg"
            label_file = train_labels_folder / f"{original_name}.txt"

            if valid_interval != 0 and counter % valid_interval == 0:
                output_img_path = valid_images_folder / f"{original_name}.jpg"
                label_file = valid_labels_folder / f"{original_name}.txt"

            img.save(output_img_path, "JPEG")
            bboxes = (
                data.get("florence_results", [{}])[0].get("bboxes", [])
                if "florence_results" in data
                else data.get("bboxes", [])
            )
            if "sliced_yolo_boxes" in data:
                bboxes = data["sliced_yolo_boxes"]
            if "yolo" in data and index < len(data["yolo"]):
                bboxes = [data["yolo"][index]["bbox"]]

            for bbox in bboxes:
                self.save_yolo_bbox(label_file, data, bbox)

            counter += 1


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    creator = YOLODatasetCreator(config)
    creator.create_dataset()
    return creator


def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Утилита для создания YOLO датасета")

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к исходной папке",
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        default=config.data_folder,
        help="Путь к папке с отфильтрованными данными",
    )
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default=config.dataset_folder,
        help="Путь к папке для сохранения датасета",
    )
    parser.add_argument(
        "--valid",
        type=int,
        default=config.valid,
        help="Процент данных для валидации",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=config.class_id,
        help="Идентификатор класса для YOLO",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Перемешать файлы перед обработкой",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
