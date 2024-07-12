import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLOv10
from tqdm import tqdm

parent_path = str(Path(__file__).resolve().parents[1])
sys.path.append(parent_path)


class Config:
    def __init__(self):
        self.input_dir = "c:/video/2024-05-10"
        self.output_dir = "c:/video/florence"
        self.models_path = "models"
        self.yolo_model = "yolov10s.pt"
        self.florence_model = "microsoft/Florence-2-large"
        self.yolo_id = 0
        self.general_prompt = None
        self.forcemask = []
        self.class_prompts_default = []
        self.class_prompts = {}
        self.device = "cpu"
        self.fps = 1
        self.confidence = 0.7
        self.border = 0.20
        self.debug = True
        self.query = "<CAPTION_TO_PHRASE_GROUNDING>"
        self.mode = "images"
        self.subfolder = True
        self.save_original = True
        self.save_boxes = False
        self.draw_boxes = False
        self.verbose = False
        self.scale = 1.0


class ObjectDetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.get_device(config.device))
        self.processed_files = []

        yolo_model_path = os.path.join(
            parent_path,
            config.models_path,
            config.yolo_model,
        )
        self.yolo_model = YOLOv10(yolo_model_path).to(self.device)
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            config.florence_model,
            trust_remote_code=True,
            use_flash_attention_2=False,
        ).to(self.device)

        self.florence_processor = AutoProcessor.from_pretrained(
            config.florence_model,
            trust_remote_code=True,
        )

        # Загрузка уже обработанных файлов
        results_file_path = Path(config.output_dir) / "results.json"
        if results_file_path.exists():
            with open(results_file_path, "r", encoding="utf-8") as f:
                self.processed_files = json.load(f)

    def get_device(self, device):
        if device == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def process_yolo_results(self, yolo_results, rgb_frame):
        results = []
        images = []
        for box in yolo_results.boxes.data.tolist():
            x1, y1, x2, y2, conf, class_id = box
            if int(class_id) == self.config.yolo_id and conf > self.config.confidence:
                result, images = self.process_detection(x1, y1, x2, y2, conf, rgb_frame)
                results.append(result)

        return results, images

    def process_detection(self, x1, y1, x2, y2, conf, rgb_frame):
        width = rgb_frame.shape[1]
        height = rgb_frame.shape[0]

        border_x = int((x2 - x1) * self.config.border)
        border_y = int((y2 - y1) * self.config.border)
        x1, y1 = max(0, x1 - border_x), max(0, y1 - border_y)
        x2, y2 = min(width, x2 + border_x), min(height, y2 + border_y)
        cropped_image = rgb_frame[int(y1) : int(y2), int(x1) : int(x2)]

        pil_image = Image.fromarray(cropped_image)
        images = [pil_image]

        florence_result = self.florence_detect(pil_image)

        result = {
            "width": width,
            "height": height,
            "yolo_box": [x1, y1, x2, y2],
            "yolo_confidence": conf,
            "florence_results": florence_result,
        }
        return result, images

    def resize_image(self, image):
        if self.config.scale != 1:
            width = int(image.shape[1] * self.config.scale)
            height = int(image.shape[0] * self.config.scale)
            return cv2.resize(image, (width, height))
        return image

    def process_image(self, image):
        image = self.resize_image(image)
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.config.yolo_id == -1:
            return self.process_no_yolo(rgb_frame)
        else:
            yolo_results = self.yolo_model(rgb_frame, verbose=self.config.verbose)[0]
            return self.process_yolo_results(yolo_results, rgb_frame)

    def process_no_yolo(self, rgb_frame):
        pil_image = Image.fromarray(rgb_frame)
        florence_result = self.florence_detect(pil_image)

        width = rgb_frame.shape[1]
        height = rgb_frame.shape[0]

        result = {
            "width": width,
            "height": height,
            "yolo_box": [0, 0, width, height],
            "yolo_confidence": 0,
            "florence_results": florence_result,
        }

        return [result], [pil_image]

    def process_video(self, video_path):
        video_name = Path(video_path).stem
        output_dir = Path(self.config.output_dir) / video_name
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps / self.config.fps)

        results = []

        for frame_number in tqdm(
            range(0, total_frames, frame_interval), desc="Processing Video"
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                break

            image_results, images = self.process_image(frame)

            for idx, result in enumerate(image_results):
                result["frame"] = frame_number
                result["file"] = str(video_path)
                self.save_results(
                    images[idx],
                    result,
                    frame_number,
                    idx,
                    output_dir,
                    is_video=True,
                    base_name=video_name,
                )
                self.processed_files.append(result)
                self.save_processed_files()

            results.extend(image_results)

        cap.release()
        return results

    def draw_boxes_on_image(
        self,
        pil_image,
        bbox,
        label,
        outline_color="red",
        outline_width=3,
        text_color="red",
        copy=True,
    ):
        if copy:
            pil_image = pil_image.copy()

        draw = ImageDraw.Draw(pil_image)

        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=outline_color, width=outline_width)
        draw.text((x1, y1), label, fill=text_color)

        return pil_image

    def is_file_processed(self, file_path):
        return any(entry["file"] == str(file_path) for entry in self.processed_files)

    def process_images(self, input_folder):
        results = []
        glob_pattern = "**/*" if self.config.subfolder else "*"
        image_files = [
            f
            for f in Path(input_folder).glob(glob_pattern)
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]

        total_images = len(image_files)
        for idx, image_file in enumerate(
            tqdm(image_files, desc="Processing Images"), 1
        ):
            if any(
                mask in str(image_file) or fnmatch.fnmatch(image_file, mask)
                for mask in self.config.forcemask
            ):

                print(f"Файл {image_file} принудительно обрабатывается заново")
            elif self.is_file_processed(image_file):
                print(f"Файл {image_file} уже обработан")
                continue

            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Не удалось прочитать изображение: {image_file}")
                continue

            image_results, images = self.process_image(image)
            for img_idx, result in enumerate(image_results):
                result["file"] = str(image_file)
                self.save_results(
                    images[img_idx],
                    result,
                    idx,
                    img_idx,
                    Path(self.config.output_dir),
                    is_video=False,
                    base_name=image_file.stem,
                )
                self.processed_files.append(result)
                self.save_processed_files()
            results.extend(image_results)

        return results

    def florence_detect(self, image):
        return self.predict(self.config.query, image, self.config.general_prompt)

    def predict(self, task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.florence_processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(self.device)

        generated_ids = self.florence_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.florence_processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.florence_processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )

        return parsed_answer

    def save_results(
        self,
        pil_image,
        result,
        frame_number,
        index,
        output_dir,
        is_video=False,
        base_name="",
    ):
        if is_video:
            image_path = (
                output_dir / f"{base_name}.frame_{frame_number:06d}.{index:03d}.png"
            )
        else:
            image_path = output_dir / f"{base_name}.{index:03d}.png"

        if self.config.save_original:
            pil_image.save(image_path, "PNG")

        json_path = image_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=4)

        florence_results = result["florence_results"].get(self.config.query, [])
        if (
            self.config.draw_boxes or self.config.save_boxes
        ) and "bboxes" in florence_results:
            fboxes = florence_results.get("bboxes")
            flabels = florence_results.get("labels")
            if fboxes and flabels:
                for bbox_index, (label, bbox) in enumerate(zip(flabels, fboxes)):
                    florence_image = pil_image.crop(bbox)
                    if self.config.draw_boxes:
                        florence_image = self.draw_boxes_on_image(
                            pil_image, bbox, label
                        )

                    florence_image_path = image_path.with_name(
                        f"{image_path.stem}.florence.{bbox_index:03d}.png"
                    )

                    florence_image.save(florence_image_path, "PNG")

    def save_processed_files(self):
        results_file_path = Path(self.config.output_dir) / "results.json"
        with open(results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=4)


def process_data(config):
    detector = ObjectDetector(config)

    if config.mode == "video":
        for video_file in Path(config.input_dir).glob("*.mp4"):
            print(f"Обработка видео: {video_file}")
            detector.process_video(video_file)
    else:
        detector.process_images(config.input_dir)


def main():
    config = Config()

    parser = argparse.ArgumentParser(
        description="Детектор объектов с использованием YOLOv10 и Microsoft Florence2"
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=config.input_dir if config.debug else None,
        required=not config.debug,
        help="Директория с входными файлами",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=config.output_dir if config.debug else None,
        required=not config.debug,
        help="Директория для сохранения результатов",
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        default=config.yolo_model,
        help="Путь к весам модели YOLO",
    )
    parser.add_argument(
        "--florence_model",
        type=str,
        default=config.florence_model,
        help="Название или путь к модели Florence",
    )
    parser.add_argument(
        "--yolo_id",
        type=int,
        default=config.yolo_id,
        help="ID класса YOLO для детекции",
    )
    parser.add_argument(
        "--general_prompt",
        type=str,
        default=config.general_prompt if config.debug else None,
        required=not config.debug,
        help="Общий промпт для детекции",
    )
    parser.add_argument(
        "--class_prompts",
        type=str,
        nargs="+",
        default=config.class_prompts_default if config.debug else None,
        required=not config.debug,
        help='Промпты для классов в формате "class_id=prompt"',
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.device,
        choices=["cpu", "cuda"],
        help="Устройство для вычислений",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.fps,
        help="Кадров в секунду для детекции",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.confidence,
        help="Минимальная уверенность для детекций YOLO",
    )
    parser.add_argument(
        "--border",
        type=float,
        default=config.border,
        help="Процент увеличения размера ограничивающей рамки",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=config.query,
        help="Запрос к модели Microsoft Florence",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=config.mode,
        choices=["video", "images"],
        help="Режим обработки: video или images",
    )
    parser.add_argument(
        "--subfolder",
        action="store_true",
        default=config.subfolder,
        help="Обрабатывать подпапки во входной директории",
    )
    parser.set_defaults(subfolder=config.subfolder)

    parser.add_argument(
        "--save_original",
        action="store_true",
        help="Сохранять оригинальные изображения",
    )
    parser.set_defaults(save_boxes=config.save_boxes)

    parser.add_argument(
        "--save_boxes",
        action="store_true",
        help="Сохранять боксы из результатов Florence",
    )
    parser.set_defaults(save_boxes=config.save_boxes)

    parser.add_argument(
        "--draw_boxes",
        action="store_true",
        help="Рисовать боксы из результатов Florence",
    )
    parser.set_defaults(save_boxes=config.save_boxes)

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Выводить отладочную информацию YOLOv10",
    )
    parser.set_defaults(verbose=config.verbose)

    parser.add_argument(
        "--scale",
        type=float,
        default=config.scale,
        help="Масштабирование изображений перед обработкой",
    )

    parser.add_argument(
        "--forcemask",
        nargs="+",
        default=config.forcemask,
        help="Список файловых масок для принудительной обработки",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    config.class_prompts = {}
    for prompt in args.class_prompts:
        class_id, prompt_text = prompt.split("=")
        config.class_prompts[prompt_text] = int(class_id)

    process_data(config)


if __name__ == "__main__":
    main()