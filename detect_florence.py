import argparse
import fnmatch
import json
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLOv10
from tqdm import tqdm


class Config:
    def __init__(
        self,
        source_folder="~/data/video",
        target_folder="~/data/florence",
        models_path="~/models",
        yolo_model="yolov10s.pt",
        florence_model="microsoft/Florence-2-large",
        yolo_id=0,
        general_prompt=None,
        forcemask=None,
        class_prompts=None,
        device="cpu",
        fps=1,
        confidence=0.7,
        border=0.20,
        debug=True,
        query="<CAPTION_TO_PHRASE_GROUNDING>",
        mode="images",
        subfolder=True,
        save_original=True,
        save_boxes=False,
        save_yolo=False,
        draw_boxes=False,
        verbose=False,
        scale=1.0,
        draw_yolo_boxes=False,
        yolo_slice=1,
        yolo_slice_overlap=0.0,
    ):
        if forcemask is None:
            forcemask = []
        if class_prompts is None:
            class_prompts = []

        self.source_folder = source_folder
        self.target_folder = target_folder
        self.models_path = models_path
        self.yolo_model = yolo_model
        self.florence_model = florence_model
        self.yolo_id = yolo_id
        self.general_prompt = general_prompt
        self.forcemask = forcemask
        self.class_prompts = class_prompts
        self.device = device
        self.fps = fps
        self.confidence = confidence
        self.border = border
        self.debug = debug
        self.query = query
        self.mode = mode
        self.subfolder = subfolder
        self.save_original = save_original
        self.save_boxes = save_boxes
        self.save_yolo = save_yolo
        self.draw_boxes = draw_boxes
        self.verbose = verbose
        self.scale = scale
        self.draw_yolo_boxes = draw_yolo_boxes
        self.yolo_slice = yolo_slice
        self.yolo_slice_overlap = yolo_slice_overlap


class FlorenceDetector:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(self.get_device(config.device))
        self.processed_files = []

        self.source_folder = Path(config.source_folder).expanduser()
        self.models_path = Path(config.models_path).expanduser()
        self.target_folder = Path(config.target_folder).expanduser()
        self.target_folder.mkdir(parents=True, exist_ok=True)

        yolo_model_path = self.models_path / self.config.yolo_model
        if self.config.yolo_id != -1:
            self.yolo_model = YOLOv10(str(yolo_model_path)).to(self.device)

        if self.config.query:
            self.florence_model = AutoModelForCausalLM.from_pretrained(
                config.florence_model,
                trust_remote_code=True,
                use_flash_attention_2=False,
            ).to(self.device)

            self.florence_processor = AutoProcessor.from_pretrained(
                config.florence_model,
                trust_remote_code=True,
            )

        results_file_path = self.target_folder / "results.json"
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
                result, image = self.process_detection(x1, y1, x2, y2, conf, rgb_frame)
                results.append(result)
                images.extend(image)

        return results, images

    def process_detection(self, x1, y1, x2, y2, conf, rgb_frame):
        width = rgb_frame.shape[1]
        height = rgb_frame.shape[0]

        box_width = int(x2 - x1)
        box_height = int(y2 - y1)

        border_x = int(box_width * self.config.border)
        border_y = int(box_height * self.config.border)
        x1, y1 = max(0, x1 - border_x), max(0, y1 - border_y)
        x2, y2 = min(width, x2 + border_x), min(height, y2 + border_y)
        cropped_image = rgb_frame[int(y1) : int(y2), int(x1) : int(x2)]

        sliced_yolo_boxes = []
        if self.config.yolo_slice != 0:
            if self.config.yolo_slice == -1:
                overlap = self.config.yolo_slice_overlap / 100.0
                if box_width >= box_height:
                    slice_size = box_height
                    num_slices = int((box_width / (slice_size * (1 - overlap))) + 1)
                else:
                    slice_size = box_width
                    num_slices = int((box_height / (slice_size * (1 - overlap))) + 1)
            else:
                num_slices = self.config.yolo_slice
                slice_size = min(box_width, box_height)

            sliced_images = []
            if box_width >= box_height:  # нарезка по ширине
                offset = (
                    (box_width - slice_size * num_slices) // (num_slices - 1)
                    if num_slices > 1
                    else 0
                )
                for i in range(num_slices):
                    slice_x1 = x1 + i * (slice_size + offset)
                    slice_x2 = slice_x1 + slice_size
                    if slice_x2 > x2:
                        slice_x2 = x2
                        slice_x1 = slice_x2 - slice_size
                    slice_y1 = y1
                    slice_y2 = y2
                    sliced_yolo_boxes.append([slice_x1, slice_y1, slice_x2, slice_y2])
            else:  # нарезка по высоте
                offset = (
                    (box_height - slice_size * num_slices) // (num_slices - 1)
                    if num_slices > 1
                    else 0
                )
                for j in range(num_slices):
                    slice_y1 = y1 + j * (slice_size + offset)
                    slice_y2 = slice_y1 + slice_size
                    if slice_y2 > y2:
                        slice_y2 = y2
                        slice_y1 = slice_y2 - slice_size
                    slice_x1 = x1
                    slice_x2 = x2
                    sliced_yolo_boxes.append([slice_x1, slice_y1, slice_x2, slice_y2])
            pil_image = Image.fromarray(rgb_frame)
            images = [pil_image]
        else:
            pil_image = Image.fromarray(cropped_image)
            images = [pil_image]

        if self.config.query:
            florence_results = []
            for image in images:
                florence_result = self.florence_detect(image)
                florence_results.append(florence_result)
        else:
            florence_results = []

        yolo_box = [x1, y1, x2, y2]

        if self.config.save_yolo or self.config.draw_yolo_boxes:
            if not self.config.save_boxes:
                pil_image = Image.fromarray(rgb_frame)
            if self.config.yolo_slice != 0 and self.config.draw_yolo_boxes:
                for sliced_box in sliced_yolo_boxes:
                    pil_image = self.draw_boxes_on_image(
                        pil_image, sliced_box, f"YOLO Slice", copy=False
                    )
            else:
                pil_image = self.draw_boxes_on_image(
                    pil_image, yolo_box, f"YOLO: {conf:.2f}"
                )

        result = {
            "width": width,
            "height": height,
            "yolo_box": yolo_box,
            "yolo_confidence": conf,
            "sliced_yolo_boxes": sliced_yolo_boxes,
        }

        if florence_results:
            result["florence_results"] = florence_results

        return result, [pil_image]

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
        if self.config.query:
            florence_result = self.florence_detect(pil_image)
        else:
            florence_result = {}

        caption_query = "<CAPTION>"
        caption_query = "<DETAILED_CAPTION>"
        caption_query = None
        if caption_query:
            captions = []
            bboxes = florence_result.get(self.config.query, {}).get("bboxes", [])
            for bbox in bboxes:
                florence_image = pil_image.crop(bbox)
                florence_caption = self.predict(caption_query, florence_image)
                florence_caption = florence_caption.get(caption_query, "").strip()
                captions.append(florence_caption)
            florence_result[self.config.query][caption_query] = captions

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
        video_name = video_path.stem
        output_dir = self.target_folder / video_name
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
                    Path(self.target_folder),
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

        if self.config.query:
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
        results_file_path = self.target_folder / "results.json"
        with open(results_file_path, "w", encoding="utf-8") as f:
            json.dump(self.processed_files, f, ensure_ascii=False, indent=4)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    detector = FlorenceDetector(config)
    if config.mode == "video":
        for video_file in detector.source_folder.glob("*.mp4"):
            print(f"Обработка видео: {video_file}")
            detector.process_video(video_file)
    else:
        detector.process_images(detector.source_folder)
    return detector


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Детектор объектов с использованием YOLOv10 и Microsoft Florence2"
    )

    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Директория с входными файлами",
    )

    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Директория для сохранения результатов",
    )

    parser.add_argument(
        "--models_path",
        type=str,
        default=config.models_path,
        help="Путь к моделям",
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
        default=config.general_prompt,
        help="Общий промпт для детекции",
    )
    parser.add_argument(
        "--class_prompts",
        type=str,
        nargs="+",
        default=config.class_prompts,
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

    parser.add_argument(
        "--save_original",
        action="store_true",
        default=config.save_original,
        help="Сохранять оригинальные изображения",
    )

    parser.add_argument(
        "--save_yolo",
        action="store_true",
        default=config.save_yolo,
        help="Сохранять боксы из результатов YOLO",
    )

    parser.add_argument(
        "--save_boxes",
        action="store_true",
        default=config.save_boxes,
        help="Сохранять боксы из результатов Florence",
    )

    parser.add_argument(
        "--draw_boxes",
        action="store_true",
        default=config.draw_boxes,
        help="Рисовать боксы из результатов Florence",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=config.verbose,
        help="Выводить отладочную информацию YOLO",
    )

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

    parser.add_argument(
        "--draw_yolo_boxes",
        action="store_true",
        default=config.draw_yolo_boxes,
        help="Рисовать боксы из результатов YOLO",
    )

    parser.add_argument(
        "--yolo_slice",
        type=int,
        default=config.yolo_slice,
        help="Количество квадратных кусков для нарезки результатов YOLO",
    )

    parser.add_argument(
        "--yolo_slice_overlap",
        type=float,
        default=config.yolo_slice_overlap,
        help="Процент перекрытия для нарезки YOLO",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
