import os
import shutil
import argparse
from datetime import datetime


class VideoFileSorter:
    def __init__(self, source_folder, target_folder):
        self.source_folder = source_folder
        self.target_folder = target_folder

    def extract_metadata(self, filename):
        parts = filename.split("-")
        camera = f"cam-{parts[1]}"
        date_str = "-".join(parts[2:5])[:10]  # Извлекаем полную дату YYYY-MM-DD
        return camera, date_str

    def create_folder(self, path):
        os.makedirs(path, exist_ok=True)

    def sort_files(self):
        for filename in os.listdir(self.source_folder):
            if filename.endswith(".mp4"):
                camera, date = self.extract_metadata(filename)

                camera_folder = os.path.join(self.target_folder, camera)
                self.create_folder(camera_folder)

                date_folder = os.path.join(camera_folder, date)
                self.create_folder(date_folder)

                source_path = os.path.join(self.source_folder, filename)
                target_path = os.path.join(date_folder, filename)

                shutil.move(source_path, target_path)
                print(f"Перемещен файл {filename} в {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Сортировка видеофайлов по папкам в соответствии с камерой и датой."
    )
    parser.add_argument(
        "--source_folder",
        required=True,
        help="Исходная папка, содержащая видеофайлы",
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        help="Целевая папка для отсортированных файлов",
    )
    args = parser.parse_args()

    sorter = VideoFileSorter(args.source_folder, args.target_folder)
    sorter.sort_files()


if __name__ == "__main__":
    main()
