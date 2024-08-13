import argparse
import gc
import json
import time
from pathlib import Path
import cv2
import torch
import numpy as np
import tensorflow as tf
from ultralytics import YOLOv10
from tqdm import tqdm


# Конфигурационный класс
class Config:
    def __init__(
        self,
        input_folder="",
        yolo_weight="",
        ocr_weights="",
        confidence=0.5,
        resize_width=200,
        resize_height=50,
        padding_color=(181, 181, 181),
        mean_color=True,
        vocabulary="*02137O64PX.TBC5L8:ERN-Г9Пл_SAIV ",
    ):
        self.input_folder = input_folder
        self.yolo_weight = yolo_weight
        self.ocr_weights = ocr_weights
        self.confidence = confidence
        self.resize_width = resize_width
        self.resize_height = resize_height
        self.padding_color = padding_color
        self.mean_color = mean_color
        self.vocabulary = vocabulary


class YOLOOCRProcessor:
    def __init__(self, config):
        self.config = config
        self.yolo_model = YOLOv10(config.yolo_weight)
        custom_objects = {
            "LeakyReLU": tf.keras.layers.LeakyReLU,
            "LSTM": tf.keras.layers.LSTM,
            "Bidirectional": tf.keras.layers.Bidirectional,
            "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
        }
        self.ocr_model = tf.keras.models.load_model(
            config.ocr_weights, custom_objects=custom_objects, compile=False
        )

    def detect_objects(self, image):
        results = self.yolo_model(image)[0]
        filtered_boxes = [
            box
            for box in results.boxes.data.tolist()
            if box[4] >= self.config.confidence
        ]
        return filtered_boxes

    def preprocess_image(self, image):
        target_size = (self.config.resize_width, self.config.resize_height)
        threshold = 170

        if target_size == image.shape[:2]:
            return image

        # Применяем маску по каждому каналу отдельно
        mask = (image > threshold).any(axis=-1)

        if self.config.mean_color and np.any(mask):
            bright_pixels = image[mask]

            # Проверка, что bright_pixels не пуст и имеет нужную размерность
            if bright_pixels.ndim == 2 and bright_pixels.shape[0] > 0:
                brightness = np.mean(bright_pixels, axis=0)
                sorted_indices = np.argsort(brightness.mean(axis=-1))[::-1]

                top_20_percent = bright_pixels[
                    sorted_indices[: max(1, int(0.15 * len(sorted_indices)))]
                ]

                mean_color = np.mean(top_20_percent, axis=0)
                if np.isnan(mean_color).any():
                    pad_color = self.config.padding_color
                else:
                    pad_color = mean_color.astype(int).tolist()
            else:
                pad_color = self.config.padding_color
        else:
            pad_color = self.config.padding_color

        old_size = image.shape[:2]

        if old_size[0] > target_size[1] or old_size[1] > target_size[0]:
            ratio = min(target_size[1] / old_size[0], target_size[0] / old_size[1])
            new_size = (
                int(old_size[1] * ratio),
                int(old_size[0] * ratio),
            )
            image = cv2.resize(image, (new_size[0], new_size[1]))
            old_size = image.shape[:2]

        new_image = np.full(
            (target_size[1], target_size[0], 3), pad_color, dtype=np.uint8
        )

        y_offset = (target_size[1] - old_size[0]) // 2
        x_offset = 0

        new_image[
            y_offset : y_offset + old_size[0], x_offset : x_offset + old_size[1]
        ] = image

        return new_image

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results_indx = tf.sparse.to_dense(
            tf.nn.ctc_greedy_decoder(
                tf.transpose(pred, perm=[1, 0, 2]), input_len, blank_index=0
            )[0][0]
        ).numpy()
        results_char = []
        for result in results_indx:
            result_char = []
            for indx in result:
                if indx != 0:  # Предполагается, что 0 это blank_index
                    result_char.append(self.config.vocabulary[indx])
                else:
                    result_char.append("")  # Явно игнорируем blank символ
            results_char.append("".join(result_char))
        return results_char

    def get_plates(self, plates):
        imgs = np.zeros((len(plates), 200, 50, 3), dtype=np.uint8)
        for i, plate in enumerate(plates):
            plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
            plate = cv2.resize(plate, (50, 200))
            imgs[i, :, :, :] = plate

        pred_logits = self.ocr_model.predict_on_batch(imgs.astype(np.float32) / 255.0)
        pred_labels_chars = self.decode_batch_predictions(pred_logits)
        return pred_labels_chars

    def process_image(self, image_path):
        image_cv = cv2.imread(str(image_path))

        boxes = self.detect_objects(image_cv)
        boxes = sorted(boxes, key=lambda box: box[0])

        plates = []
        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            cropped_image = image_cv[int(y1) : int(y2), int(x1) : int(x2)]
            plate = self.preprocess_image(cropped_image)
            plates.append(plate)

        texts = self.get_plates(plates)

        ocr_results = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2, conf, class_id = box
            ocr_results.append(
                {
                    "image": str(image_path),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "text": texts[i],
                    "confidence": conf,
                    "class_id": int(class_id),
                }
            )

        combined_text = " ".join(texts)
        print(f"Текст {combined_text}")

        return ocr_results

    def process_images(self):
        input_folder = Path(self.config.input_folder)
        image_paths = list(input_folder.glob("*.png"))

        all_ocr_results = []

        for image_path in image_paths:
            start_time = time.time()
            ocr_results = self.process_image(image_path)
            all_ocr_results.extend(ocr_results)

            end_time = time.time()
            processing_time = end_time - start_time

            print(
                f"Time for processing {image_path.name}: {processing_time:.2f} seconds"
            )

        output_json = input_folder / "results.json"
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(all_ocr_results, f, ensure_ascii=False, indent=4)

        print(f"Результаты сохранены в {output_json}")


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = YOLOOCRProcessor(config)
    processor.process_images()
    return processor


def main():
    config = Config()
    config.input_folder = r"c:\proplex\test"
    config.yolo_weight = r"c:\projects\models\yolov10_textbox00710s2.pt"
    config.ocr_weights = r"c:\projects\models\EfficientNetV2L_ocr_092028.keras"

    parser = argparse.ArgumentParser(
        description="YOLO и OCR утилита для детекции и распознавания текста"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        default=config.input_folder,
        help="Путь к папке с изображениями",
    )
    parser.add_argument(
        "--yolo_weight",
        type=str,
        default=config.yolo_weight,
        help="Путь к весам модели YOLO",
    )
    parser.add_argument(
        "--ocr_weights",
        type=str,
        default=config.ocr_weights,
        help="Путь к весам модели OCR",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.confidence,
        help="Минимальный порог уверенности для YOLO",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=config.resize_width,
        help="Ширина изменения размера изображения перед OCR",
    )
    parser.add_argument(
        "--resize_height",
        type=int,
        default=config.resize_height,
        help="Высота изменения размера изображения перед OCR",
    )
    parser.add_argument(
        "--padding_color",
        type=int,
        nargs=3,
        default=config.padding_color,
        help="Цвет отступов при изменении размера изображения (три целых числа)",
    )
    parser.add_argument(
        "--mean_color",
        action="store_true",
        default=config.mean_color,
        help="Использовать средний цвет ярких пикселей для отступов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
