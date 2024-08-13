import os
import re
import shutil
import json
import pandas as pd
from collections import defaultdict

vocab = "-0127436LNP.ВХПСГОТ58лOER-X: B_9CTASIVР"

chars = {
    ",": ".",
    "*": ".",
    "/": "7",
    "Z": "7",
    "Л": "л",
    "б": "6",
    "n": "л",
    "$": "S",
    "!": "I",
    "А": "A",
    "Е": "E",
    "К": "K",
    "З": "S",
    "]": "I",
    "|": "I",
    "(": "С",
    "F": "Г",
    "[": "I",
    "ж": "л",
    "K": "K",
    "m": "N",
    "l": "I",
    "И": "N",
    "о": "О",
}

invalid_vocab = []


def move_invalid_json_files(source_folder, invalid_json_folder):
    """
    Перемещает JSON-файлы без соответствующих PNG-файлов в указанную папку.

    :param source_folder: Путь к папке с JSON-файлами
    :param invalid_json_folder: Путь к папке для невалидных JSON-файлов
    """
    if not os.path.exists(invalid_json_folder):
        os.makedirs(invalid_json_folder)

    png_files = {f for f in os.listdir(source_folder) if f.endswith(".png")}

    for json_file in os.listdir(source_folder):
        if json_file.endswith(".json"):
            base_name = re.sub(r"\.json$", "", json_file)
            # Проверяем наличие хотя бы одного соответствующего PNG-файла
            pattern = re.compile(re.escape(base_name) + r"\.\d{3}\.png")
            if not any(pattern.match(png_file) for png_file in png_files):
                # print(f"Перемещение файла {json_file} в папку {invalid_json_folder} из-за отсутствия соответствующих PNG-файлов.")
                shutil.move(
                    os.path.join(source_folder, json_file),
                    os.path.join(invalid_json_folder, json_file),
                )


def find_missing_chars(text, vocab):
    missing_chars = [c for c in text if c not in vocab]
    return missing_chars


def move_invalid_png_files(source_folder, invalid_png_folder, max_length=13):
    """
    Перемещает PNG-файлы, для которых нет соответствующего JSON-файла или в которых текст пустой или длина текста превышает max_length.

    :param source_folder: Путь к папке с PNG и JSON файлами
    :param invalid_png_folder: Путь к папке для невалидных PNG-файлов
    :param max_length: Максимальная допустимая длина текста
    """
    if not os.path.exists(invalid_png_folder):
        os.makedirs(invalid_png_folder)

    for json_file in os.listdir(source_folder):
        if json_file.endswith(".json"):
            with open(
                os.path.join(source_folder, json_file), "r", encoding="utf-8"
            ) as file:
                try:
                    json_data = json.load(file)
                    base_name = re.sub(r"\.json$", "", json_file)
                    ocr_data = json_data.get("tesseract", json_data.get("easyocr", []))
                    for i, item in enumerate(ocr_data):
                        png_file = f"{base_name}.{i:03d}.png"
                        png_path = os.path.join(source_folder, png_file)
                        text = item.get("text", "")
                        if text == "" or len(text) > max_length:
                            if os.path.exists(png_path):
                                print(
                                    f"Перемещение файла {png_file} в папку {invalid_png_folder} из-за отсутствия текста или превышения максимальной длины."
                                )
                                shutil.move(
                                    png_path, os.path.join(invalid_png_folder, png_file)
                                )

                        missing_chars = find_missing_chars(text, vocab)
                        if missing_chars:
                            print(
                                f"Перемещение файла {png_file} в папку {invalid_png_folder} из-за наличия недопустимых символов: {missing_chars}"
                            )
                            shutil.move(
                                png_path, os.path.join(invalid_png_folder, png_file)
                            )

                            for c in missing_chars:
                                if c not in invalid_vocab:
                                    invalid_vocab.append(c)

                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON в файле: {json_file}")
                except Exception as e:
                    print(f"Произошла ошибка при обработке файла {json_file}: {e}")


def extract_text_statistics(json_folder, replace_dict):
    """
    Извлекает текст, уверенность и bounding boxes из всех JSON-файлов в указанной папке и
    считает количество вхождений, а также статистику ширины и высоты bounding boxes.

    :param json_folder: Путь к папке с JSON-файлами
    :param replace_dict: Словарь с заменами для текстов
    :return: DataFrame с текстом, уверенностью и bounding boxes
    """
    label_counts = defaultdict(int)
    bbox_stats = defaultdict(lambda: {"widths": [], "heights": []})

    def apply_replacements(text, replacements):
        for old, new in replacements.items():
            if text == old:
                text = new
                break
        return text

    parent_folder = os.path.basename(os.path.dirname(json_folder))
    replacements = replace_dict.get(parent_folder, {})
    for json_file in os.listdir(json_folder):
        if json_file.endswith(".json"):
            with open(
                os.path.join(json_folder, json_file), "r", encoding="utf-8"
            ) as file:
                try:
                    json_data = json.load(file)
                    modified = False
                    ocr_data = json_data.get("tesseract", json_data.get("easyocr", []))
                    for item in ocr_data:
                        text = item.get("text", "")
                        new_text = apply_replacements(text, replacements)
                        if new_text != text:
                            item["text"] = new_text
                            modified = True
                        else:
                            new_text = "".join(chars.get(char, char) for char in text)
                            if new_text != text:
                                item["text"] = new_text
                                modified = True

                        bbox = item.get("bbox", [0, 0, 0, 0])
                        label_counts[new_text] += 1
                        width = bbox[2] - bbox[0]
                        height = bbox[3] - bbox[1]
                        bbox_stats[new_text]["widths"].append(width)
                        bbox_stats[new_text]["heights"].append(height)
                    if modified:
                        new_json_path = os.path.join(json_folder, "new")
                        if not os.path.exists(new_json_path):
                            os.makedirs(new_json_path)
                        with open(
                            os.path.join(new_json_path, json_file),
                            "w",
                            encoding="utf-8",
                        ) as new_file:
                            json.dump(json_data, new_file, ensure_ascii=False, indent=4)
                except json.JSONDecodeError:
                    print(f"Ошибка декодирования JSON в файле: {json_file}")
                except Exception as e:
                    print(f"Произошла ошибка при обработке файла {json_file}: {e}")

    data = []
    replaced = 0
    for label, count in label_counts.items():
        if count >= 10:
            if bbox_stats[label]["widths"] and bbox_stats[label]["heights"]:
                min_width = min(bbox_stats[label]["widths"])
                max_width = max(bbox_stats[label]["widths"])
                min_height = min(bbox_stats[label]["heights"])
                max_height = max(bbox_stats[label]["heights"])
            else:
                min_width = max_width = min_height = max_height = None
            data.append([label, count, min_width, max_width, min_height, max_height])
        else:
            replacements[label] = ""
            replaced += 1

    df = pd.DataFrame(
        data,
        columns=[
            "label",
            "count",
            "min width",
            "max width",
            "min height",
            "max height",
        ],
    )
    df = df.sort_values(by="count", ascending=False)
    df.to_clipboard(index=False)
    return df, replaced


labelreplace = {
    "28062024": "28 062024",
    "ЕСО": "ECO",
    "ECO": "ECO",
    "№": "N",
    "30573": "30673",
    "MBX": "ПВХ",
    "52024": "62024",
    "6^2": "6л2",
    "28052024": "28 062024",
    "C1063N": "C1.063N",
    "С1.063": "С1.063",
    "16°11": "16 10",
    "М": "N",
    "C1083N": "C1.063N",
    "16:12": "16 12",
    "пвх": "ПВХ",
    "30873": "30673",
    "16:11": "16 11",
    "16°12": "16 12",
    "С1053": "С1.063",
    "6A2": "6л2",
    "С1.053": "С1.063",
    "5^2": "6л2",
    "С1.063№": "C1.063N",
    "м": "N",
    "И": "N",
    "С1063": "С1.063",
    "ИВХ": "ПВХ",
    "16`12": "16 12",
    "16°10": "16 10",
    "542": "6л2",
    "642": "6л2",
    "6a2": "6л2",
    "ИЗХ": "ПВХ",
    "пах": "ПВХ",
    "гост": "ГОСТ",
    "—": "",
    "|": "",
    "©": "",
    "=": "",
    "С": "",
    "—^": "",
    "i": "",
    "_": "",
    "A2": "",
    "Е": "",
}

label1replace = {
    "1063": "1.063",
    "742": "7л2",
    "7/2": "7л2",
    "07202L": "072024",
    "[": "L",
    "ГВХ": "ПВХ",
    "PPOPLEX": "PROPLEX",
    "7.2": "7л2",
    "106": "1.06",
    "18 07202L": "18 072024",
    ")": "0",
    "PRC": "PRO",
    "2L": "24",
}

label2replace = {
    "гост": "ГОСТ",
    "1063": "1.063",
    "пвх": "ПВХ",
    "16.43": "16:43",
    "16.4": "16:4",
    "ПВх": "ПВХ",
    "16.42": "16:42",
    "16-42": "16:42",
    "16-39": "16:39",
    "L1063": "L 1.063",
    "Li063": "L 1.063",
    "16.41": "16:41",
    "16.40": "16:40",
    "ГОСт": "ГОСТ",
    "16.44": "16:44",
    "тост": "ГОСТ",
    "ГОСI": "ГОСТ",
    "ГОСТ": "ГОСТ",
    "16-41": "16:41",
    "16.39": "16:39",
    "1063 N": "1.063 N",
    "02-07-24": "24-07-24",
    "1640": "16:40",
    "16 39": "16:39",
    "16-43": "16:43",
    "16-40": "16:40",
    "пвХ": "ПВХ",
    "16 42": "16:42",
    "1641": "16:41",
    "24-07-22": "24-07-24",
    "11063": "L 1.063",
    "271": "2л1",
    "гоСт": "ГОСТ",
    "Пвх": "ПВХ",
    "ГоСт": "ГОСТ",
    "LOCI": "ГОСТ",
    "IOCI": "ГОСТ",
    "ПвХ": "ПВХ",
    "пэх": "ПВХ",
    "Гост": "ГОСТ",
    "м": "N",
    "2z": "2л",
    "Zл1": "2л1",
    "7063": "1.063",
    "30623": "30673",
    "1-063": "1.063",
    "2x": "2л",
    "Z71": "2л1",
    "76.43": "16:43",
    "ПЕХ": "ПВХ",
    "2д1": "2л1",
    "гест": "ГОСТ",
    "1643": "16:43",
    "госI": "ГОСТ",
    "Fост": "ГОСТ",
    "БOCI": "ГОСТ",
    "46.43": "16:43",
    "1€-39": "16:39",
    "16 41": "16:41",
    "2z-07-22": "24-07-24",
    "L1063 N": "L 1.063 N",
    "IBX": "ПВХ",
    "PROPLEX L1063": "PROPLEX L 1.063",
    "2z-07-24": "24-07-24",
    "04-07-24": "24-07-24",
    "16 44": "16:44",
    "ГОст": "ГОСТ",
    "27-07-24": "24-07-24",
    "ГОСТ 30673": "ГОСТ 30673",
    "PRCPLEX": "PROPLEX",
    "пых": "ПВХ",
    "[OCI": "ГОСТ",
    "24-07-2-": "24-07-24",
    "ГВх": "ПВХ",
    "PRSPLEX": "PROPLEX",
    "2zi": "2л1",
    "ГО": "ГО",
    "2z1": "2л1",
    "ост": "ОСТ",
    "ГOСI": "ГОСТ",
}

label3replace = {
    "|": "1",
    "116": "л16",
    "716": "л16",
    "filo": "л16",
    "flo": "л16",
    "№": "",
    "nl6": "л16",
    "RIASILVERECO": "RTASILVERECO",
    ".": "",
    "гост": "ГОСТ",
    ">": "",
    "|.": "",
    "Л16": "л16",
    "ВМ_1.": "BN_1.",
    "ВN_1.": "BN_1.",
}

label4replace = {
    "ГВХ": "ПВХ",
    "7/2": "7л2",
    "7.2": "7л2",
    "07202L": "072024",
    "18 07202L": "18 072024",
    "PPOPLEX": "PROPLEX",
    "[": "L",
}
synht3replace = {
    "ТA3": "TAS",
    "ТA3I": "TASI",
    "ВEВТ": "BERT",
    "ВТA": "RTA",
    "ВEВТ": "BERT",
    "ВEВТAS": "BERTAS",
    "TOC": "ГОС",
    "ВEВТAS": "BERTAS",
    "ВEВТA": "BERTA",
}

replace = {
    "label": labelreplace,
    "label1": label1replace,
    "label2": label2replace,
    "label3": label3replace,
    "label4": label4replace,
    "synth3": synht3replace,
}

for base in ["synth", "label", "label1", "label2", "label3", "label4"]:
    source_folder = f"c:/proplex/{base}/ocr"

    # Перемещаем невалидные JSON и PNG-файлы
    invalid_json_folder = f"c:/proplex/{base}/ocr.json.removed"
    invalid_png_folder = f"c:/proplex/{base}/ocr.png.removed"

    df, replaced = extract_text_statistics(source_folder, replace)
    move_invalid_json_files(source_folder, invalid_json_folder)
    move_invalid_png_files(source_folder, invalid_png_folder, max_length=9)
    move_invalid_json_files(source_folder, invalid_json_folder)
    move_invalid_png_files(source_folder, invalid_png_folder, max_length=9)

    df, replaced = extract_text_statistics(source_folder, replace)
    print(df.to_string(index=False))
    print(replaced)

    if replaced > 0:
        df, replaced = extract_text_statistics(source_folder, replace)
        print(df.to_string(index=False))
        print(replaced)

if invalid_vocab:
    print("Invalid vocab:", invalid_vocab)
