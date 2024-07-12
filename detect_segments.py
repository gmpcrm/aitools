import json
import os
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLOv10
import argparse

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from common.logger import AIDesLogger
from tqdm import tqdm


class VideoProcessingConfig:
    def __init__(
        self,
        fps,
        device,
        output_file,
        results_file,
        class_id,
        model_weights,
        confidence,
        border,
        gap,
        save_images,
        save_boxes,
    ):
        self.fps = fps
        self.device = device
        self.output_file = output_file
        self.results_file = results_file
        self.class_id = class_id
        self.model_weights = model_weights
        self.confidence = confidence
        self.border = border
        self.gap = gap
        self.save_images = save_images
        self.save_boxes = save_boxes


class VideoFrameDetector:
    def __init__(self, video_path, config):
        self.device = config.device
        self.logger = AIDesLogger("vext", log_to_console=True, log_to_file=True)
        self.video_path = video_path
        self.fps = config.fps
        self.results_file = config.results_file
        self.class_id = config.class_id
        self.model_weights = config.model_weights
        self.save_boxes = config.save_boxes
        self.save_images = config.save_images
        self.save_all_boxes = False
        self.use_cuda = self.check_cuda()

        self.cpu_used = 1  # Single thread
        self.start_index = 0

        self.confidence = config.confidence
        self.model_path = os.path.join(parent_path, "models")
        self.border = config.border
        self.reduce_factor = 0

        self.yolo_model = None
        self.names = {}

        if self.save_images or self.save_boxes:
            self.output_path = os.path.join(
                os.path.dirname(video_path),
                "output",
                os.path.splitext(os.path.basename(video_path))[0],
            )
            os.makedirs(self.output_path, exist_ok=True)

    def init_models(self):
        if self.model_weights:
            model_path = os.path.join(self.model_path, self.model_weights)
            self.yolo_model = YOLOv10(model=model_path, verbose=False)
            if self.device == "cuda":
                self.yolo_model = self.yolo_model.to(self.device)

    def predict(self, frame, verbose=False):
        result = self.yolo_model(frame, verbose=verbose)
        self.register_names(result[0].names)
        return result

    def register_names(self, names):
        if isinstance(names, dict):
            for key, value in names.items():
                self.names[key] = value

    def get_class_name(self, class_id):
        return self.names.get(class_id, "unknown")

    def filter_boxes(self, boxes):
        return [
            box
            for box in boxes
            if int(box[5]) == self.class_id and box[4] > self.confidence
        ]

    def draw_boxes_on_image(
        self,
        pil_image,
        bbox,
        label,
        outline_color="red",
        outline_width=3,
        text_color="red",
    ):
        draw = ImageDraw.Draw(pil_image)
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width)
        draw.text((x1, y1), label, fill=text_color)
        return pil_image

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = int(video_fps / self.fps)

        extracted_count = 0
        yolo_results = []

        try:
            if frame_interval == 0:
                print(f"Ошибка обработки файла: {self.video_path}")
                return total_frames, extracted_count, yolo_results

            for frame_number in tqdm(
                range(0, total_frames, frame_interval), desc="Обработка кадров"
            ):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = cap.read()
                if not ret:
                    break

                extracted_count += 1
                if extracted_count < self.start_index:
                    continue

                if self.reduce_factor > 1:
                    frame = frame[:: self.reduce_factor, :: self.reduce_factor, :]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                model_result = self.predict(rgb_frame, verbose=False)[0]
                boxes = model_result.boxes.data.tolist()
                filtered_boxes = self.filter_boxes(boxes)

                seconds_frame = int(extracted_count / self.fps)
                if filtered_boxes:
                    yolo_results.append([frame_number, seconds_frame, filtered_boxes])

                if self.save_images:
                    image_path = os.path.join(
                        self.output_path, f"frame_{seconds_frame:06d}.png"
                    )
                    cv2.imwrite(image_path, frame)

                if self.save_boxes:
                    for box_index, box in enumerate(filtered_boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        border_x = int((x2 - x1) * self.border)
                        border_y = int((y2 - y1) * self.border)
                        x1, y1 = max(0, x1 - border_x), max(0, y1 - border_y)
                        x2, y2 = min(width, x2 + border_x), min(height, y2 + border_y)

                        cropped_image = rgb_frame[y1:y2, x1:x2]
                        pil_image = Image.fromarray(cropped_image)

                        box_path = os.path.join(self.output_path, "boxes")
                        os.makedirs(box_path, exist_ok=True)

                        box_path = os.path.join(
                            box_path, f"frame_{seconds_frame:06d}.{box_index}.png"
                        )
                        pil_image.save(box_path, format="PNG")
        finally:
            cap.release()

        return total_frames, extracted_count, yolo_results

    def check_cuda(self):
        if self.device == "cpu":
            print("Использование CPU")
            return False

        print("CUDA доступен.")
        return True

    def process_video(self):
        frame_count, extracted_count, yolo_results = self.extract_frames()
        return frame_count, extracted_count, yolo_results

    def add_segment(self, segments, segment):
        start_time = min(seg[1] for seg in segment)
        end_time = max(seg[1] for seg in segment)
        segments.append((start_time, end_time - start_time))

    def get_segments(self, video_path, results, gap=0):
        if not results:
            return []

        results = sorted(
            [
                (frame_number, seconds_frame, boxes)
                for frame_number, seconds_frame, boxes in results
                if any(self.filter_boxes(boxes))
            ],
            key=lambda x: x[1],
        )

        longest = []
        current = [results[0]]

        for i in range(1, len(results)):
            if results[i][1] <= results[i - 1][1] + 1 + gap:
                current.append(results[i])
            else:
                self.add_segment(longest, current)
                current = [results[i]]

        if current:
            self.add_segment(longest, current)

        return sorted(longest, key=lambda x: x[0])


def process_video(video_path, config):
    detector = VideoFrameDetector(video_path, config)
    detector.init_models()

    start_time = time.time()
    frames, seconds, yolo_results = detector.process_video()
    end_time = time.time()

    segments = detector.get_segments(video_path, yolo_results, gap=config.gap)
    processing_time = end_time - start_time
    fps = frames / processing_time

    print(f"Файл: {video_path}")
    print(f"Всего кадров в видео: {frames}")
    print(f"Извлечено кадров: {seconds}")
    print(f"Время выполнения: {processing_time:.2f} секунд")
    print(f"FPS (кадров в секунду): {fps:.2f}")

    return frames, seconds, yolo_results, segments


def save_segments(video_path, segments, output_file):
    with open(output_file, "a") as f:
        f.write(f'"{video_path}",')
        for start, length in segments:
            f.write(f"{int(start)}:{int(length)},")
        f.write("\n")


def save_results(video_path, results, results_file):
    with open(results_file, "a") as f:
        json.dump({video_path: results}, f, indent=4)
        f.write("\n")


def read_segments_files(segments_file):
    processed_files = set()
    if os.path.exists(segments_file):
        with open(segments_file, "r") as f:
            for line in f:
                if line.strip():
                    video_path = line.split(",")[0].strip('"')
                    processed_files.add(video_path)
    return processed_files


def process_videos(video_paths, config):
    processed_files = read_segments_files(config.output_file)
    for video_path in video_paths:
        if video_path in processed_files:
            print(f"Файл {video_path} уже обработан, пропуск")
            continue

        frame_count, extracted_count, results, segments = process_video(
            video_path, config
        )
        save_segments(video_path, segments, config.output_file)
        save_results(video_path, results, config.results_file)


def get_video_files_from_folder(folder_path):
    return [str(p) for p in Path(folder_path).glob("*.mp4")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Поиск максимально длительных сегментов видео с помощью YOLOv10."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="c:\\video",
        help="Папка, содержащая исходные видеофайлы",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.1,
        help="Кадры в секунду для детекции (по умолчанию: 0.1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство для использования (cpu или cuda)",
    )
    parser.add_argument(
        "--segments",
        type=str,
        default="segments.txt",
        help="Файл для сохранения результатов сегментов",
    )
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Файл для сохранения детализированных результатов",
    )
    parser.add_argument(
        "--class_id", type=int, default=0, help="YOLO id класса для фильтрации объектов"
    )
    parser.add_argument(
        "--save_images",
        action="store_true",
        help="Сохранить обрабатываемые кадры с видео",
    )
    parser.add_argument(
        "--save_boxes",
        action="store_true",
        help="Сохранить обрезанные области с обнаруженными объектами",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="yolov10s.pt",
        help="Путь к весам модели YOLOv10",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Минимальная уверенность для фильтрации объектов",
    )
    parser.add_argument(
        "--border",
        type=float,
        default=0.20,
        help="Процент увеличения границы вокруг обнаруженного объекта",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=20,
        help="Разрешенный промежуток, в секундах, между кадрами для сегментации",
    )

    args = parser.parse_args()

    video_files = get_video_files_from_folder(args.source)
    config = VideoProcessingConfig(
        fps=args.fps,
        device=args.device,
        output_file=args.segments,
        results_file=args.results,
        class_id=args.class_id,
        model_weights=args.model_weights,
        confidence=args.confidence,
        border=args.border,
        gap=args.gap,
        save_images=args.save_images,
        save_boxes=args.save_boxes,
    )

    process_videos(video_files, config)

    cv2.destroyAllWindows()
