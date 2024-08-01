import pandas as pd
import json
from collections import Counter
import argparse
import cv2


class Config:
    def __init__(self, source_files=[], max_length=13):
        self.source_files = source_files
        self.max_length = max_length


class OCRVocabulary:
    def __init__(self, config):
        self.config = config
        self.valid_chars = "-0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ:. _()л"
        self.data = self.load_data()
        self.letter_counts = self.get_letter_counts()
        self.real_vocab = self.create_real_vocab()
        self.missing_chars = self.get_missing_chars()
        self.full_vocab = self.real_vocab + self.missing_chars
        self.max_text_length = self.get_max_text_length()
        self.min_max_dimensions = self.get_min_max_dimensions()

    def load_data(self):
        all_data = []
        for json_file in self.config.source_files:
            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                all_data.extend(data["ocr_files"])
        return all_data

    def get_letter_counts(self):
        all_text = "".join([item["text"] for item in self.data])
        letter_counter = Counter(all_text)
        df = pd.DataFrame(letter_counter.items(), columns=["letter", "count"])
        df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        return df

    def create_real_vocab(self):
        filtered_vocab = [
            char for char in self.letter_counts["letter"] if char in self.valid_chars
        ]
        return "".join(filtered_vocab)

    def get_letter_counts_df(self):
        return self.letter_counts

    def get_real_vocab(self):
        return self.real_vocab

    def get_missing_chars(self):
        used_chars = set(self.letter_counts["letter"])
        missing_chars = [char for char in self.valid_chars if char not in used_chars]
        return "".join(missing_chars)

    def get_full_vocab(self):
        return self.full_vocab

    def get_max_text_length(self):
        return max(len(item["text"]) for item in self.data)

    def get_min_max_dimensions(self):
        heights = []
        widths = []
        for item in self.data:
            image_path = item.get("file")
            if image_path:
                try:
                    img = cv2.imread(image_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        heights.append(height)
                        widths.append(width)
                    else:
                        print(f"Не удалось прочитать изображение: {image_path}")
                except Exception as e:
                    print(f"Ошибка при чтении файла {image_path}: {e}")

        min_height = min(heights) if heights else None
        max_height = max(heights) if heights else None
        min_width = min(widths) if widths else None
        max_width = max(widths) if widths else None
        return min_height, max_height, min_width, max_width

    def get_texts_longer_than(self, length):
        long_texts = [item["text"] for item in self.data if len(item["text"]) > length]
        unique_long_texts = set(long_texts)
        return unique_long_texts, len(unique_long_texts)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    ocr_vocab = OCRVocabulary(config)
    letter_counts_df = ocr_vocab.get_letter_counts_df()
    real_vocab = ocr_vocab.get_real_vocab()
    missing_chars = ocr_vocab.get_missing_chars()
    full_vocab = ocr_vocab.get_full_vocab()
    max_text_length = ocr_vocab.get_max_text_length()
    min_height, max_height, min_width, max_width = ocr_vocab.get_min_max_dimensions()

    print(letter_counts_df)
    print("Реальный словарь:", real_vocab)
    print("Отсутствующие символы:", missing_chars)
    print("Полный словарь:", full_vocab)
    print("Максимальная длина текста:", max_text_length)
    print(f"Минимальная высота: {min_height}, Максимальная высота: {max_height}")
    print(f"Минимальная ширина: {min_width}, Максимальная ширина: {max_width}")

    if config.max_length is not None:
        unique_long_texts, count = ocr_vocab.get_texts_longer_than(config.max_length)
        print(f"Уникальные тексты длиннее {config.max_length}:")
        for text in unique_long_texts:
            print(text)
        print(f"Количество уникальных текстов длиннее {config.max_length}: {count}")

    return ocr_vocab


def main():
    config = Config()
    base = "c:/proplex"
    source_files = [
        f"{base}/label/ocr.json",
        f"{base}/label1/ocr.json",
        f"{base}/label2/ocr.json",
        f"{base}/label3/ocr.json",
        f"{base}/label4/ocr.json",
    ]

    config.source_files = source_files
    config.max_length = 9

    parser = argparse.ArgumentParser(description="Анализ OCR словаря")

    parser.add_argument(
        "--source_files",
        nargs="+",
        default=config.source_files,
        help="Пути к JSON файлам с данными OCR",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=config.max_length,
        help="Максимальная длина текста для фильтрации уникальных значений",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
