import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


class Config:
    def __init__(self, source_folder="~/data/source", target_file="~/data/target.json"):
        self.source_folder = source_folder
        self.target_file = target_file


class OCRProcessor:
    def __init__(self, config):
        self.config = config

    def process_files(self):
        ocr_files = []
        source_folder = Path(self.config.source_folder).expanduser()
        json_files = [f for f in os.listdir(source_folder) if f.endswith(".json")]
        for file_name in tqdm(json_files, desc="Processing files"):
            if file_name.endswith(".json"):
                json_path = source_folder / file_name
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                base_file_name = os.path.splitext(file_name)[0]
                ocr_data = data.get("tesseract", data.get("easyocr", []))
                for index, item in enumerate(ocr_data):
                    png_file_name = f"{base_file_name}.{index:03d}.png"
                    png_file_path = source_folder / png_file_name

                    text = item["text"]
                    if png_file_path.exists() and text:
                        ocr_files.append(
                            {
                                "file": str(png_file_path),
                                "text": text,
                            }
                        )

        with open(self.config.target_file, "w", encoding="utf-8") as f:
            json.dump({"ocr_files": ocr_files}, f, ensure_ascii=False, indent=4)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    processor = OCRProcessor(config)
    processor.process_files()
    return processor


def main():
    config = Config()
    parser = argparse.ArgumentParser(description="Утилита для обработки OCR данных")

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к исходной папке",
    )
    parser.add_argument(
        "--target_file",
        default=config.target_file,
        type=str,
        help="Полный путь к новому JSON файлу",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
