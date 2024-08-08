import json
import shutil
from collections import Counter
from tqdm import tqdm
import argparse
from pathlib import Path


class Config:
    def __init__(
        self,
        source_file="~/data/ocr.json",
        target_folder="~/data/processed",
        min_count=1,
        max_count=10,
        move=False,
    ):
        self.source_file = source_file
        self.target_folder = target_folder
        self.min_count = min_count
        self.max_count = max_count
        self.move = move


class OCRProcessor:
    def __init__(self, config):
        self.config = config
        self.cluster_map = {}
        self.existing_data = []

    def load_data(self):
        with open(self.config.source_file, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data["ocr_files"]

    def load_existing_data(self):
        target_file = Path(self.config.target_folder, "ocr.json")
        if target_file.exists():
            with open(target_file, "r", encoding="utf-8") as file:
                existing_data = json.load(file)
                self.existing_data = existing_data.get("ocr_files", [])
                for item in self.existing_data:
                    text = item["text"]
                    cluster_folder = Path(item["file"]).parent
                    if text not in self.cluster_map:
                        self.cluster_map[text] = cluster_folder

    def count_unique_texts(self, data):
        texts = [item["text"] for item in data]
        unique_counts = Counter(texts)
        return unique_counts

    def create_folders(self):
        Path(self.config.target_folder, "min").mkdir(parents=True, exist_ok=True)

    def normalize_path(self, path):
        return path.replace("\\", "/")

    def move_files(self, data, unique_counts):
        result_data = self.existing_data.copy()

        for item in tqdm(data, desc="Перемещение файлов"):
            text = item["text"]
            count = unique_counts[text]
            old_path = Path(item["file"])
            if not old_path.exists():
                continue

            if count <= self.config.min_count:
                new_path = Path(self.config.target_folder, "min", old_path.name)
                if self.config.move:
                    shutil.move(str(old_path), str(new_path))
            elif count >= self.config.max_count:
                if text not in self.cluster_map:
                    cluster_index = len(self.cluster_map) + 1
                    cluster_folder = Path(
                        self.config.target_folder, f"cluster_{cluster_index:04d}"
                    )
                    cluster_folder.mkdir(parents=True, exist_ok=True)
                    self.cluster_map[text] = cluster_folder
                new_path = Path(self.cluster_map[text], old_path.name)
                if self.config.move:
                    shutil.move(str(old_path), str(new_path))
            else:
                new_path = old_path

            result_data.append(
                {
                    "file": self.normalize_path(str(new_path)),
                    "old_file": self.normalize_path(str(old_path)),
                    "text": text,
                }
            )

        return result_data

    def save_result(self, result_data):
        result_file_path = Path(self.config.target_folder, "ocr.json")
        with open(result_file_path, "w", encoding="utf-8") as result_file:
            json.dump(
                {"ocr_files": result_data}, result_file, ensure_ascii=False, indent=4
            )

    def run(self):
        print("Загрузка данных из файла...")
        data = self.load_data()

        print("Загрузка существующих данных из целевой папки...")
        self.load_existing_data()

        print("Подсчет уникальных текстов...")
        unique_counts = self.count_unique_texts(data)

        print(f"Создание папок в {self.config.target_folder}...")
        self.create_folders()

        print("Перемещение файлов в соответствующие папки...")
        result_data = self.move_files(data, unique_counts)

        print(
            f"Сохранение результатов в {Path(self.config.target_folder, 'ocr.json')}..."
        )
        self.save_result(result_data)

        print("Процесс завершен.")


def run(**kwargs):
    config = Config(**kwargs)
    return run_config(config)


def run_config(config):
    processor = OCRProcessor(config)
    processor.run()
    return processor


def main():
    config = Config()
    config.source_file = "/content/ocr.json"
    config.target_folder = "/cluster"
    config.move = True
    config.min_count = 1
    config.max_count = 300

    parser = argparse.ArgumentParser(
        description="Утилита для обработки и перемещения OCR файлов"
    )

    parser.add_argument(
        "--source_file",
        default=config.source_file,
        type=str,
        help="Путь к исходному JSON файлу",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к целевой папке",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=config.min_count,
        help="Минимальное количество вхождений текста",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=config.max_count,
        help="Максимальное количество вхождений текста",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        default=config.move,
        help="Перемещать файлы вместо копирования",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
