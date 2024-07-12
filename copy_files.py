import argparse
import os
from pathlib import Path
import shutil
import fnmatch


class Config:
    def __init__(self):
        self.source_folder = "florence"
        self.target_folder = "florence.new"
        self.ignore_folders = ["florence.good", "florence.bad"]
        self.mask = "*.florence.*"
        self.ignored_files = set()

    def load_ignored_files(self):
        for folder in self.ignore_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                for root, _, files in os.walk(folder_path):
                    for file_name in files:
                        self.ignored_files.add(file_name)


def copy_files(config):
    source_folder = Path(config.source_folder)
    target_folder = Path(config.target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    for root, _, files in os.walk(source_folder):
        for file_name in files:
            if fnmatch.fnmatch(file_name, config.mask):
                if file_name in config.ignored_files:
                    print(f"Пропущено: {file_name}")
                else:
                    source_file = Path(root) / file_name
                    shutil.copy(source_file, target_folder)
                    print(f"Скопировано: {file_name}")


if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser(description="Утилита для копирования файлов")

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к исходной папке",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к папке для сохранения скопированных файлов",
    )
    parser.add_argument(
        "--ignore_folders",
        nargs="+",
        default=config.ignore_folders,
        help="Список папок, которые нужно игнорировать",
    )
    parser.add_argument(
        "--mask",
        type=str,
        default=config.mask,
        help="Маска для сканирования файлов",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))
    config.load_ignored_files()
    copy_files(config)
