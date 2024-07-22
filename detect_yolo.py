import argparse
import json
import os
from pathlib import Path

import cv2
import torch
from tqdm import tqdm
from ultralytics import YOLOv10
import numpy as np


class Config:
    def __init__(
        self,
        source_folder="~/data/video",
        target_folder="~/data/output",
        boxes_folder="~/data/boxes",
        models=["detectprofile.pt", "detectlabel.pt"],
        model_path="~/models",
        classes=[[0], [0]],
        subfolder=False,
        save_boxes=[False, True],
        draw_boxes=[True, True],
        save_json=[False, True],
        verbose=False,
        format="png",
        extensions=["*.jpg", "*.jpeg", "*.png"],
        borders=[0.0, 20.0],
        confidiences=[0.5, 0.7],
        max_confidience=False,
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.boxes_folder = boxes_folder
        self.models = models
        self.model_path = model_path
        self.classes = classes
        self.subfolder = subfolder
        self.save_boxes = save_boxes
        self.draw_boxes = draw_boxes
        self.save_json = save_json
        self.verbose = verbose
        self.format = format
        self.extensions = extensions
        self.borders = borders
        self.confidiences = confidiences
        self.max_confidience = max_confidience


class YOLODetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = [
            YOLOv10(os.path.join(config.model_path, model)).to(self.device)
            for model in config.models
        ]

    def detect_objects(self, model, image):
        results = model(image, verbose=self.config.verbose)[0]
        return results

    def add_border(self, x1, y1, x2, y2, border_ratio, width, height):
        box_width = x2 - x1
        box_height = y2 - y1
        border_x = int(box_width * border_ratio)
        border_y = int(box_height * border_ratio)
        x1 = max(0, x1 - border_x)
        y1 = max(0, y1 - border_y)
        x2 = min(width, x2 + border_x)
        y2 = min(height, y2 + border_y)
        return [x1, y1, x2, y2]

    def process_image_recursive(
        self,
        image_cv,
        original_filepath,
        parent_image=None,
        model_idx=0,
        bbox_prefix=None,
        x_offset=0,
        y_offset=0,
        original_size=(0, 0),
    ):
        if model_idx >= len(self.models):
            return

        if model_idx == 0:
            original_size = (image_cv.shape[1], image_cv.shape[0])
            bbox_prefix = []

        results = self.detect_objects(self.models[model_idx], image_cv)

        original_filename = Path(original_filepath).stem
        original_width, original_height = original_size

        image_meta = {
            "width": original_width,
            "height": original_height,
            "model": self.config.models[model_idx],
            "yolo": [],
            "file": original_filepath,
        }

        filtered_boxes = [
            box
            for box in results.boxes.data.tolist()
            if box[4] >= self.config.confidiences[model_idx]
            and int(box[5]) in self.config.classes[model_idx]
        ]

        if self.config.max_confidience:
            filtered_boxes = sorted(filtered_boxes, key=lambda x: x[4], reverse=True)[
                :1
            ]

        for idx, box in enumerate(filtered_boxes):
            x1, y1, x2, y2, conf, class_id = box
            bordered_bbox = self.add_border(
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                self.config.borders[model_idx] / 100.0,
                original_width,
                original_height,
            )
            cropped_image = image_cv[
                bordered_bbox[1] : bordered_bbox[3],
                bordered_bbox[0] : bordered_bbox[2],
            ]
            bbox = [
                bordered_bbox[0] + x_offset,
                bordered_bbox[1] + y_offset,
                bordered_bbox[2] + x_offset,
                bordered_bbox[3] + y_offset,
            ]

            image_meta["yolo"].append(
                {"class_id": int(class_id), "confidence": conf, "bbox": bbox}
            )

            if self.config.draw_boxes[model_idx] and parent_image is not None:
                cv2.rectangle(
                    parent_image,
                    (bbox[0], bbox[1]),
                    (bbox[2], bbox[3]),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    parent_image,
                    f"{int(class_id)}: {conf:.2f}",
                    (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )

            if self.config.save_boxes[model_idx]:
                image_bbox_identifier = bbox_prefix + ["yolo"] + [f"{idx:03d}"]
                image_bbox_identifier_str = ".".join(image_bbox_identifier)
                box_image_name = f"{original_filename}.{image_bbox_identifier_str}.{self.config.format}"
                box_image_path = Path(self.config.target_folder) / box_image_name
                cv2.imwrite(str(box_image_path), cropped_image)

            self.process_image_recursive(
                cropped_image,
                original_filepath,
                parent_image,
                model_idx + 1,
                bbox_prefix + [f"{idx:03d}"],
                x_offset + bordered_bbox[0],
                y_offset + bordered_bbox[1],
                original_size,
            )

        if image_meta["yolo"] and self.config.save_json[model_idx]:
            bbox_identifier_str = ".".join(bbox_prefix)
            json_file_name = f"{original_filename}{'.' if bbox_prefix else ''}{bbox_identifier_str}.json"
            json_file_path = Path(self.config.target_folder) / json_file_name
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(image_meta, f, ensure_ascii=False, indent=4)

    def process_folder(self):
        source_path = Path(self.config.source_folder)
        target_path = Path(self.config.target_folder)
        boxes_path = Path(self.config.boxes_folder)
        target_path.mkdir(parents=True, exist_ok=True)
        boxes_path.mkdir(parents=True, exist_ok=True)

        if self.config.subfolder:
            image_files = []
            for ext in self.config.extensions:
                image_files.extend(source_path.rglob(ext))
        else:
            image_files = []
            for ext in self.config.extensions:
                image_files.extend(source_path.glob(ext))

        for image_path in tqdm(image_files, desc="Обработка изображений"):
            image_cv = cv2.imread(str(image_path))
            parent_image = image_cv.copy() if any(self.config.draw_boxes) else None
            self.process_image_recursive(
                image_cv, str(image_path), parent_image=parent_image
            )

            if parent_image is not None:
                draw_image_name = f"{image_path.stem}.boxes.{self.config.format}"
                draw_image_path = boxes_path / draw_image_name
                cv2.imwrite(str(draw_image_path), parent_image)


def run(**kwargs):
    config = Config(**kwargs)
    return run_config(config)


def run_config(config):
    detector = YOLODetector(config)
    detector.process_folder()
    return detector


def main():
    config = Config()

    parser = argparse.ArgumentParser(
        description="Утилита для детекции объектов с использованием YOLO"
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
        "--boxes_folder",
        type=str,
        default=config.boxes_folder,
        help="Папка для сохранения изображений с боксами",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=config.models,
        help="Файлы моделей YOLO",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=config.model_path,
        help="Путь к файлу модели YOLO",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        action="append",
        default=config.classes,
        help="Классы YOLO для каждой модели",
    )
    parser.add_argument(
        "--subfolder",
        action="store_true",
        default=config.subfolder,
        help="Включать подпапки",
    )
    parser.add_argument(
        "--save_boxes",
        type=bool,
        nargs="+",
        default=config.save_boxes,
        help="Сохранять боксы для каждой модели",
    )
    parser.add_argument(
        "--draw_boxes",
        type=bool,
        nargs="+",
        default=config.draw_boxes,
        help="Рисовать боксы для каждой модели",
    )
    parser.add_argument(
        "--save_json",
        type=bool,
        nargs="+",
        default=config.save_json,
        help="Сохранять JSON для каждой модели",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.verbose,
        help="Подробный вывод",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg"],
        default=config.format,
        help="Формат для сохранения изображений",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=config.extensions,
        help="Расширения файлов для обработки (например, *.jpg *.png)",
    )
    parser.add_argument(
        "--borders",
        type=float,
        nargs="+",
        default=config.borders,
        help="Процент увеличения размера ограничивающей рамки для каждой модели",
    )
    parser.add_argument(
        "--confidiences",
        type=float,
        nargs="+",
        default=config.confidiences,
        help="Минимальная уверенность для каждой модели",
    )
    parser.add_argument(
        "--max_confidience",
        action="store_true",
        default=config.max_confidience,
        help="Сохранять только максимальную уверенность для каждого класса",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
