import argparse
import json
import os
from pathlib import Path


class Config:
    def __init__(
        self,
        source_files=[],
        source_vocabulary="",
        target_vocabulary="",
        target_folder="",
    ):
        self.source_files = source_files
        self.source_vocabulary = source_vocabulary
        self.target_vocabulary = target_vocabulary
        self.target_folder = target_folder


class TextReplacer:
    def __init__(self, source_vocabulary, target_vocabulary):
        self.translation_map = str.maketrans(source_vocabulary, target_vocabulary)

    def replace_text(self, text):
        return text.translate(self.translation_map)


def process_files(config):
    replacer = TextReplacer(config.source_vocabulary, config.target_vocabulary)
    target_folder = Path(config.target_folder).expanduser()
    target_folder.mkdir(parents=True, exist_ok=True)

    for file_path in config.source_files:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data.get("ocr_files", []):
            item["text"] = replacer.replace_text(item["text"])

        target_path = target_folder / Path(file_path).name
        with open(target_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Файл обработан и сохранен: {target_path}")


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    process_files(config)
    return config


def main():
    config = Config()
    config.source_files = [
        "/content/ocr.json",
        "/proplex/dataset012/ocr012.json",
    ]
    config.source_vocabulary = "АВЕКМНОРСТУХаекорсух"
    config.target_vocabulary = "ABEKMHOPCTYXaekopcyx"
    config.target_folder = "/content/fixed"

    parser = argparse.ArgumentParser(
        description="Утилита для замены символов в JSON-файлах OCR"
    )

    parser.add_argument(
        "--source_files",
        nargs="+",
        default=config.source_files,
        help="Пути к исходным JSON файлам",
    )
    parser.add_argument(
        "--source_vocabulary",
        default=config.source_vocabulary,
        help="Исходный словарь символов",
    )
    parser.add_argument(
        "--target_vocabulary",
        default=config.target_vocabulary,
        help="Целевой словарь символов",
    )
    parser.add_argument(
        "--target_folder",
        default=config.target_folder,
        help="Папка для сохранения обработанных файлов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
