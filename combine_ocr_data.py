import os
import json
import shutil
import random
import argparse
from tqdm import tqdm


class Config:
    def __init__(
        self,
        source_files=[],
        target_folder="/content/ocr",
        target_json="/content/ocr.json",
        shuffle=False,
    ):
        self.source_files = source_files
        self.target_folder = target_folder
        self.target_json = target_json
        self.shuffle = shuffle


class JSONProcessor:
    def __init__(self, config):
        self.config = config

    def read_json_files(self):
        data = []
        for file_path in self.config.source_files:
            with open(file_path, "r", encoding="utf-8") as file:
                data.extend(json.load(file)["ocr_files"])
        return data

    def normalize_path(self, path):
        return path.replace("\\", "/")

    def copy_files_and_update_paths(self, data):
        if not os.path.exists(self.config.target_folder):
            os.makedirs(self.config.target_folder)

        for entry in tqdm(data, desc="Копирование файлов"):
            source_file_path = entry["file"]
            filename = os.path.basename(source_file_path)
            target_file_path = os.path.join(self.config.target_folder, filename)
            target_file_path = self.normalize_path(target_file_path)

            shutil.copy(source_file_path, target_file_path)
            entry["file"] = target_file_path

        return data

    def save_json(self, data):
        with open(self.config.target_json, "w", encoding="utf-8") as file:
            json.dump({"ocr_files": data}, file, ensure_ascii=False, indent=4)

    def process(self):
        data = self.read_json_files()

        if self.config.shuffle:
            random.shuffle(data)

        updated_data = self.copy_files_and_update_paths(data)
        self.save_json(updated_data)


def run(**kwargs):
    config = Config(**kwargs)
    processor = JSONProcessor(config)
    processor.process()


def run_config(config):
    processor = JSONProcessor(config)
    processor.process()
    return processor


def main():
    config = Config()
    base = "c:/proplex"
    config.source_files = [
        f"{base}/synth/ocr.json",
        f"{base}/label/ocr.json",
        f"{base}/label1/ocr.json",
        f"{base}/label2/ocr.json",
        f"{base}/label3/ocr.json",
        f"{base}/label4/ocr.json",
    ]
    config.target_folder = f"/content/ocr"
    config.target_json = f"/content/ocr.json"
    config.shuffle = True

    parser = argparse.ArgumentParser(
        description="Утилита для обработки JSON файлов и копирования изображений."
    )

    parser.add_argument(
        "--source_files",
        nargs="+",
        default=config.source_files,
        help="Список исходных JSON файлов",
    )
    parser.add_argument(
        "--target_folder",
        default=config.target_folder,
        help="Папка для сохранения копий файлов",
    )
    parser.add_argument(
        "--target_json",
        default=config.target_json,
        help="Выходной результирующий JSON файл",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=config.shuffle,
        help="Перемешать результаты",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))
    run_config(config)


if __name__ == "__main__":
    main()
