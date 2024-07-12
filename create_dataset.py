import argparse
import json
import os
from pathlib import Path
import re
from PIL import Image


class Config:
    def __init__(self):
        self.source_folder = "c:\\proplex\\florence"
        self.data_folder = "c:\\proplex\\florence.good"
        self.dataset_folder = "c:\\proplex\\dataset"
        self.valid_percent = 20  # Default validation percentage
        self.class_id = 0  # Default class ID for YOLO


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

    def create_dataset(self):
        dataset_folder = Path(self.config.dataset_folder)
        train_images_folder = dataset_folder / "train" / "images"
        train_labels_folder = dataset_folder / "train" / "labels"
        valid_images_folder = dataset_folder / "valid" / "images"
        valid_labels_folder = dataset_folder / "valid" / "labels"

        train_images_folder.mkdir(parents=True, exist_ok=True)
        train_labels_folder.mkdir(parents=True, exist_ok=True)
        valid_images_folder.mkdir(parents=True, exist_ok=True)
        valid_labels_folder.mkdir(parents=True, exist_ok=True)

        file_pattern = re.compile(r"^(.*)\.florence\.(\d+)\.png$")
        counter = 0
        valid_interval = 100 // self.config.valid_percent

        for file_name in os.listdir(self.config.data_folder):
            if file_name.endswith(".png"):
                match = file_pattern.match(file_name)
                if not match:
                    continue

                original_name = match.group(1)
                index_str = match.group(2)
                index = int(index_str)

                json_file_name = f"{original_name}.json"
                json_file_path = Path(self.config.source_folder) / json_file_name
                original_img_path = (
                    Path(self.config.source_folder) / f"{original_name}.png"
                )

                if json_file_path.exists() and original_img_path.exists():
                    with open(json_file_path, "r", encoding="utf-8") as json_file:
                        data = json.load(json_file)

                    img = Image.open(original_img_path)
                    output_img_path = train_images_folder / f"{original_name}.jpg"
                    label_file_path = train_labels_folder / f"{original_name}.txt"

                    if counter % valid_interval == 0:
                        output_img_path = valid_images_folder / f"{original_name}.jpg"
                        label_file_path = valid_labels_folder / f"{original_name}.txt"

                    img.save(output_img_path, "JPEG")

                    florence_results = next(
                        iter(data["florence_results"].values()), None
                    )
                    if florence_results:
                        bboxes = florence_results["bboxes"]
                        if index < len(bboxes):
                            bbox = bboxes[index]
                            yolo_bbox = self.convert_bbox_to_yolo(
                                (data["width"], data["height"]), bbox
                            )

                            with open(
                                label_file_path, "a", encoding="utf-8"
                            ) as label_file:
                                label_file.write(
                                    f"{self.config.class_id} {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n"
                                )

                    counter += 1
                    print(f"Добавлен файл: {file_name}")


if __name__ == "__main__":
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
        default=config.valid_percent,
        help="Процент данных для валидации",
    )
    parser.add_argument(
        "--class_id",
        type=int,
        default=config.class_id,
        help="Идентификатор класса для YOLO",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))
    creator = YOLODatasetCreator(config)
    creator.create_dataset()
