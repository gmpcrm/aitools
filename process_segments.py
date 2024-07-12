import argparse
import os
import shutil


class SegmentProcessor:
    def __init__(self, segments_file, delete_folder, new_segments_file, mintime):
        self.segments_file = segments_file
        self.delete_folder = delete_folder
        self.new_segments_file = new_segments_file
        self.mintime = mintime

    def read_segments(self):
        """Чтение файла с сегментами."""
        with open(self.segments_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
        return lines

    def process_segments(self, segments):
        """Обработка сегментов, фильтрация по минимальной длительности."""
        processed_segments = []
        files_to_delete = []

        for line in segments:
            parts = line.strip().split(",")
            file_path = parts[0].strip('"')

            # Проверка существования файла
            if not os.path.exists(file_path):
                print(f"Файл {file_path} не существует, пропускаем.")
                continue

            segment_data = parts[1:]

            # Если нет данных о сегментах, добавляем файл в список для удаления
            if not segment_data:
                files_to_delete.append(file_path)
                continue

            new_segments = []
            for segment in segment_data:
                if segment:
                    start, duration = map(int, segment.split(":"))
                    if duration > self.mintime:
                        new_segments.append(f"{start}:{duration}")

            # Если есть новые сегменты, добавляем их в список, иначе добавляем файл в список для удаления
            if new_segments:
                processed_segments.append(f'"{file_path}",{",".join(new_segments)}')
            else:
                files_to_delete.append(file_path)

        return processed_segments, files_to_delete

    def write_segments(self, segments):
        """Запись обработанных сегментов в новый файл."""
        with open(self.new_segments_file, "w", encoding="utf-8") as file:
            for segment in segments:
                file.write(segment + "\n")

    def delete_files(self, files):
        """Перемещение файлов в папку для удаления."""
        if not os.path.exists(self.delete_folder):
            os.makedirs(self.delete_folder)
        for file in files:
            try:
                print(f"Перемещение {file} в {self.delete_folder}")
                shutil.move(file, self.delete_folder)
            except FileNotFoundError:
                print(f"Файл {file} не найден, пропускаем.")

    def run(self):
        segments = self.read_segments()
        new_segments, files_to_delete = self.process_segments(segments)
        self.write_segments(new_segments)
        self.delete_files(files_to_delete)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обработка видео сегментов и фильтрация по длительности."
    )
    parser.add_argument(
        "--segments",
        default="segments.txt",
        type=str,
        help="Путь к файлу с сегментами",
    )
    parser.add_argument(
        "--delete_folder",
        type=str,
        default="delete_files/",
        help="Папка для перемещения файлов на удаление",
    )
    parser.add_argument(
        "--newsegments",
        default="segments_split.txt",
        type=str,
        help="Файл для записи новых сегментов",
    )
    parser.add_argument(
        "--mintime",
        type=int,
        default=300,
        help="Минимальная длительность сегментов, секунд",
    )

    args = parser.parse_args()

    processor = SegmentProcessor(
        args.segments, args.delete_folder, args.newsegments, args.mintime
    )
    processor.run()
