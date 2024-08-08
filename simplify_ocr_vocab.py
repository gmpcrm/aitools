import os
import json
import argparse


class TextReplacer:
    def __init__(self, source_vocabulary, target_vocabulary):
        self.translation_map = str.maketrans(source_vocabulary, target_vocabulary)

    def replace_text(self, text):
        return text.translate(self.translation_map)


def process_files(source_files, source_vocabulary, target_vocabulary, target_folder):
    replacer = TextReplacer(source_vocabulary, target_vocabulary)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file_path in source_files:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        for item in data.get("ocr_files", []):
            item["text"] = replacer.replace_text(item["text"])

        target_path = os.path.join(target_folder, os.path.basename(file_path))
        with open(target_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Файл обработан и сохранен: {target_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Утилита для замены символов в JSON-файлах OCR."
    )
    parser.add_argument(
        "--source_files",
        nargs="+",
        required=True,
        help="Пути к исходным JSON файлам.",
    )
    parser.add_argument(
        "--source_vocabulary",
        required=True,
        help="Исходный словарь символов.",
    )
    parser.add_argument(
        "--target_vocabulary",
        required=True,
        help="Целевой словарь символов.",
    )
    parser.add_argument(
        "--target_folder",
        required=True,
        help="Папка для сохранения обработанных файлов.",
    )

    args = parser.parse_args()

    process_files(
        args.source_files,
        args.source_vocabulary,
        args.target_vocabulary,
        args.target_folder,
    )


if __name__ == "__main__":
    main()
