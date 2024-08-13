import argparse
import threading
import time
import tkinter as tk
from collections import defaultdict
from pathlib import Path

import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
from ultralytics import YOLOv10
import torch
import numpy as np
import tensorflow as tf


class Config:
    def __init__(self, source="", model="", ocr_weights="", fps=20, device="cpu"):
        self.source = source
        self.model = model
        self.ocr_weights = ocr_weights
        self.fps = fps
        self.device = device


class ObjectDetectionApp(threading.Thread):
    def __init__(self, root, config):
        super().__init__()
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Детекция объектов с YOLOv10 и OCR")
        self.config = config
        self.cap = cv2.VideoCapture(self.config.source)
        self.device = torch.device(config.device)
        self.model = YOLOv10(self.config.model).to(self.device)
        self.custom_objects = {
            "LeakyReLU": tf.keras.layers.LeakyReLU,
            "LSTM": tf.keras.layers.LSTM,
            "Bidirectional": tf.keras.layers.Bidirectional,
            "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
        }
        self.ocr_model = tf.keras.models.load_model(
            self.config.ocr_weights, custom_objects=self.custom_objects, compile=False
        )
        self.label = tk.Label(root)
        self.label.pack(fill=tk.BOTH, expand=True)
        self.running = True
        self.confidence = 0.4
        self.text_offset_x = 0
        self.text_offset_y = 35
        self.font = ImageFont.truetype(
            "/proplex/fonts/Fira Sans Regular.ttf",
            18,
        )
        self.vocabulary = "*02137O64PX.TBC5L8:ERN-Г9Пл_SAIV "

    def run(self):
        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()
            if ret:
                width = frame.shape[1]
                height = frame.shape[0]
                pil_image = self.draw_boxes(frame, width, height)
                self.schedule_frame_update(pil_image)
            else:
                self.running = False
                break

            elapsed_time = time.time() - start_time
            delay = max(1, int((1 / self.config.fps - elapsed_time) * 1000))
            time.sleep(delay / 1000)

    def resize_frame_to_label(self, frame):
        label_width = self.label.winfo_width()
        label_height = self.label.winfo_height()
        frame = cv2.resize(frame, (label_width, label_height))
        return frame

    def detect_objects(self, frame):
        results = self.model(frame, verbose=False)[0]
        filtered_boxes = [
            box for box in results.boxes.data.tolist() if box[4] >= self.confidence
        ]
        return filtered_boxes

    def preprocess_image(self, image):
        target_size = (200, 50)
        threshold = 170

        mask = (image > threshold).any(axis=-1)
        pad_color = [181, 181, 181]

        if np.any(mask):
            bright_pixels = image[mask]
            mean_color = np.mean(bright_pixels, axis=0)
            if not np.isnan(mean_color).any():
                pad_color = mean_color.astype(int).tolist()

        old_size = image.shape[:2]

        if old_size[0] > target_size[1] or old_size[1] > target_size[0]:
            ratio = min(target_size[1] / old_size[0], target_size[0] / old_size[1])
            new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))
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
                if indx != 0:
                    result_char.append(self.vocabulary[indx])
                else:
                    result_char.append("")
            results_char.append("".join(result_char))
        return results_char

    def get_plates(self, plates):
        pred_labels_chars = []
        if plates:
            imgs = np.zeros((len(plates), 200, 50, 3), dtype=np.uint8)
            for i, plate in enumerate(plates):
                plate = cv2.rotate(plate, cv2.ROTATE_90_CLOCKWISE)
                plate = cv2.resize(plate, (50, 200))
                imgs[i, :, :, :] = plate

            pred_logits = self.ocr_model.predict_on_batch(
                imgs.astype(np.float32) / 255.0
            )
            pred_labels_chars = self.decode_batch_predictions(pred_logits)
        return pred_labels_chars

    def process_image(self, frame):
        boxes = self.detect_objects(frame)
        plates = []
        boxes = sorted(boxes, key=lambda box: box[0])

        for box in boxes:
            x1, y1, x2, y2, conf, class_id = box
            cropped_image = frame[int(y1) : int(y2), int(x1) : int(x2)]
            plate = self.preprocess_image(cropped_image)
            plates.append(plate)

        texts = self.get_plates(plates)
        combined_text = " ".join(texts)
        print(f"Текст: {combined_text}")

        return boxes, texts

    def draw_boxes(self, frame, width, height):
        boxes, texts = self.process_image(frame)
        frame = self.resize_frame_to_label(frame)

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        if texts:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2, confidence, class_id = box[:6]
                x1 = int(x1 / width * frame.shape[1])
                y1 = int(y1 / height * frame.shape[0])
                x2 = int(x2 / width * frame.shape[1])
                y2 = int(y2 / height * frame.shape[0])
                label = texts[i]

                draw.rectangle([x1, y1, x2, y2], outline="green", width=2)

                text_position = (x1 + self.text_offset_x, y1 + self.text_offset_y)
                draw.text(text_position, label, fill="green", font=self.font)

        return pil_image

    def schedule_frame_update(self, pil_image):
        if self.running:
            self.root.after(0, self.update_frame, pil_image)

    def update_frame(self, pil_image):
        if self.running:
            imgtk = ImageTk.PhotoImage(image=pil_image)
            self.label.imgtk = imgtk
            self.label.config(image=imgtk)

    def on_closing(self):
        self.running = False
        self.cap.release()
        self.check_thread()

    def check_thread(self):
        if self.is_alive():
            self.root.after(100, self.check_thread)
        else:
            cv2.destroyAllWindows()
            self.root.destroy()


def main():
    config = Config()
    config.source = (
        r"c:\proplex\label1\video\kamera 2 polozhitel'no 5 minut s markirovkoj.mp4"
    )
    config.model = r"c:\projects\models\yolov10_textbox00710s2.pt"
    config.ocr_weights = r"c:\projects\models\EfficientNetV2L_ocr_092028.keras"

    parser = argparse.ArgumentParser(description="Детекция объектов с YOLOv10 и OCR")
    parser.add_argument(
        "--source",
        type=str,
        default=config.source,
        help="Путь к видеофайлу",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model,
        help="Путь к весам модели YOLOv10",
    )
    parser.add_argument(
        "--ocr_weights",
        type=str,
        default=config.ocr_weights,
        help="Путь к весам модели OCR",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.fps,
        help="FPS для вывода на экран",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        help="Устройство для выполнения модели [cpu, cuda]",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    root = tk.Tk()
    app = ObjectDetectionApp(root, config)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.start()
    root.mainloop()


if __name__ == "__main__":
    main()
