import argparse
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os
import fnmatch


class Config:
    def __init__(
        self,
        source_folder="~/data/json_files/",
        query="<CAPTION_TO_PHRASE_GROUNDING>",
        output_file=None,
        stats=False,
    ):
        self.source_folder = source_folder
        self.output_file = output_file
        self.query = query
        self.stats = stats


class DataFrameCreator:
    def __init__(self, config):
        self.config = config
        self.source_folder = Path(config.source_folder).expanduser()
        self.output_file = (
            Path(config.output_file).expanduser() if config.output_file else None
        )

    def create_dataframe(self):
        data = []

        for root, _, files in os.walk(self.source_folder):
            for file_name in tqdm(files, desc="Обработка JSON файлов"):
                if fnmatch.fnmatch(file_name, "*.json"):
                    if file_name == "results.json":
                        continue

                    json_path = Path(root) / file_name
                    with open(json_path, "r") as file:
                        json_data = json.load(file)

                        bboxes = json_data["florence_results"][self.config.query][
                            "bboxes"
                        ]
                        labels = json_data["florence_results"][self.config.query][
                            "labels"
                        ]

                        for bbox, label in zip(bboxes, labels):
                            data.append(
                                {
                                    "label": label,
                                    "left": bbox[0],
                                    "top": bbox[1],
                                    "right": bbox[2],
                                    "bottom": bbox[3],
                                }
                            )

        df = pd.DataFrame(data)

        if self.config.stats:
            df["width"] = df["right"] - df["left"]
            df["height"] = df["bottom"] - df["top"]

            df = df.groupby("label").agg(
                {"width": ["min", "max", "mean"], "height": ["min", "max", "mean"]}
            )

            df.columns = [
                "min_width",
                "max_width",
                "avg_width",
                "min_height",
                "max_height",
                "avg_height",
            ]

            df = df.reset_index()

        if self.output_file:
            df.to_csv(self.output_file, index=False, float_format="%.2f")
            print(f"Данные сохранены в {self.output_file}")

        return df


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    creator = DataFrameCreator(config)
    return creator.create_dataframe()


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для создания DataFrame из JSON файлов"
    )

    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Исходная папка с JSON файлами",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=config.query,
        help="Запрос используемый для блока данных",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=config.output_file,
        help="Путь к выходному CSV файлу",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Вычислить и вывести статистику",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
