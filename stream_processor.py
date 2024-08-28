import asyncio
from pathlib import Path
import json

import cv2
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from ultralytics import YOLOv10
from PIL import Image, ImageDraw
from tqdm.asyncio import tqdm


async def source_video_file_stream(result_stream, folder_path):
    video_files = list(Path(folder_path).rglob("*.mp4"))
    for video_file in video_files:
        result_stream["file"] = str(video_file)
        yield result_stream


async def source_video_stream(result_stream, fps=-1.0):
    async for results in result_stream:
        video_path = results["file"]
        cap = cv2.VideoCapture(video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = int(video_fps / fps) if fps > 0 else 1

        frame_count = 0

        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    results["frame"] = frame
                    results["frame_index"] = frame_count
                    yield results
                    await asyncio.sleep(0)

                frame_count += 1
                pbar.update(1)

        cap.release()


async def detection_yolo_stream(
    result_stream,
    model_path,
    confidence=0.7,
    class_ids=None,
    device="cpu",
    border=0.20,
):
    model = YOLOv10(model_path).to(torch.device(device))

    async for results in result_stream:
        frame = results["frame"]
        detections = model(frame, verbose=False)
        yolo_results = []
        for detection in detections[0].boxes:
            x1, y1, x2, y2 = detection.xyxy[0].int().tolist()
            conf = float(detection.conf)
            class_id = int(detection.cls)
            if conf > confidence and (class_ids is None or class_id in class_ids):
                # Увеличиваем размер bbox с учетом border
                box_width = int(x2 - x1)
                box_height = int(y2 - y1)
                border_x = int(box_width * border)
                border_y = int(box_height * border)
                x1, y1 = max(0, x1 - border_x), max(0, y1 - border_y)
                x2, y2 = min(frame.shape[1], x2 + border_x), min(
                    frame.shape[0], y2 + border_y
                )

                cropped_image = frame[y1:y2, x1:x2]
                yolo_results.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "cropped_image": cropped_image,
                        "conf": conf,
                        "class_id": class_id,
                    }
                )

        results["yolo"] = yolo_results
        yield results
        await asyncio.sleep(0)


async def detection_florence_stream(
    result_stream, model_name, task_prompt, text_input=None, device="cpu"
):
    device = torch.device(device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_flash_attention_2=False,
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    async for results in result_stream:
        yolo_results = results.get("yolo", [])

        florence_results = []
        for yolo_result in yolo_results:
            cropped_image = yolo_result["cropped_image"]
            inputs = processor(
                text=task_prompt + (text_input or ""),
                images=cropped_image,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )

            generated_text = processor.batch_decode(
                outputs,
                skip_special_tokens=False,
            )[0]

            parsed_answer = processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(cropped_image.shape[1], cropped_image.shape[0]),
            )

            result_answer = parsed_answer.get(task_prompt, [])

            florence_results.append(
                {
                    "yolo_bbox": yolo_result["bbox"],
                    "yolo_conf": yolo_result["conf"],
                    "yolo_class_id": yolo_result["class_id"],
                    "cropped_image": cropped_image,
                    "answer": result_answer,
                }
            )

        results["florence"] = florence_results
        yield results
        await asyncio.sleep(0)


async def draw_yolo_stream(result_stream):
    async for results in result_stream:
        yolo_frame = results["frame"].copy()

        for yolo_result in results.get("yolo", []):
            x1, y1, x2, y2 = yolo_result["bbox"]
            cv2.rectangle(yolo_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        results["yolo_frame"] = yolo_frame

        yield results
        await asyncio.sleep(0)


def save_yolo_results(frame, results, output_path, idx):
    yolo_image = results.get("yolo_frame", frame)
    cv2.imwrite(str(output_path / f"frame_{idx:06d}.yolo.png"), yolo_image)


def save_florence_results(frame, results, output_path, idx, draw_boxes=True):
    florence_results = results.get("florence", [])
    for result_idx, florence_result in enumerate(florence_results):
        florence_answer = florence_result["answer"]
        cropped_image = florence_result["cropped_image"]

        # Сохранение cropped image без обводки
        cropped_image_path = (
            output_path / f"frame_{idx:06d}.cropped.{result_idx:03d}.png"
        )
        cv2.imwrite(str(cropped_image_path), cropped_image)

        # Сохранение результата в JSON
        json_result = {
            "yolo_bbox": florence_result["yolo_bbox"],
            "yolo_conf": florence_result["yolo_conf"],
            "yolo_class_id": florence_result["yolo_class_id"],
            "bboxes": florence_answer["bboxes"],
            "labels": florence_answer["labels"],
            "file": results.get("file"),
            "frame_index": results.get("frame_index"),
        }
        json_path = output_path / f"frame_{idx:06d}.florence.{result_idx:03d}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, ensure_ascii=False, indent=4)

        if draw_boxes:
            for box_index, (bbox, label) in enumerate(
                zip(florence_answer["bboxes"], florence_answer["labels"])
            ):
                cropped_copy = cropped_image.copy()
                pil_image = Image.fromarray(
                    cv2.cvtColor(cropped_copy, cv2.COLOR_BGR2RGB)
                )
                draw = ImageDraw.Draw(pil_image)

                draw.rectangle(bbox, outline="red", width=3)
                draw.text((bbox[0], bbox[1]), label, fill="red")

                florence_image_path = (
                    output_path
                    / f"frame_{idx:06d}.florence.{result_idx:03d}.{box_index:03d}.png"
                )

                pil_image.save(florence_image_path, "PNG")


async def save_results_stream(stream_results, output_folder, draw_boxes=True):
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    idx = 0

    async for results in stream_results:
        frame = results["frame"]
        save_yolo_results(frame, results, output_path, idx)
        save_florence_results(frame, results, output_path, idx, draw_boxes)
        idx += 1
        await asyncio.sleep(0)


async def main():
    models = "c:/projects/models"
    yolo_model = f"{models}/yolov10s.pt"
    yolo_classes = [0]
    florence_model = "microsoft/Florence-2-large"
    base = "c:/kitchen/clips/cam-1001"
    video_path = f"{base}/2024-05-10/cam-1001-2024-05-10_01-19-26.0.mp4"
    output_folder = f"{base}/2024-05-10.florence"

    florence_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    florence_text = "locate gloves on people hands"

    stream = source_video_file_stream({}, folder_path=base)
    stream = source_video_stream(stream, fps=1)
    stream = detection_yolo_stream(
        stream,
        model_path=yolo_model,
        class_ids=yolo_classes,
        border=0.20,
    )
    stream = detection_florence_stream(
        stream,
        model_name=florence_model,
        task_prompt=florence_prompt,
        text_input=florence_text,
    )
    stream = draw_yolo_stream(stream)
    await save_results_stream(stream, output_folder, draw_boxes=True)


# Запуск всего пайплайна
asyncio.run(main())
