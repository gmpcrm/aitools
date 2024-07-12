import os
import shutil
import fastdup
import argparse
from pathlib import Path


class Config:
    def __init__(
        self,
        source_folder="~/data/frames/",
        target_folder="~/data/nodupes/",
        fastdup_work_dir="/tmp/fastdup_work/",
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.fastdup_work_dir = fastdup_work_dir


class DuplicateRemover:
    def __init__(self, config):
        self.config = config
        self.fd = fastdup.create(
            work_dir=self.config.fastdup_work_dir, input_dir=self.config.source_folder
        )

    def run_fastdup(self):
        self.fd.run(overwrite=True)

    def get_files_to_remove(self):
        invalid_instances_df = self.fd.invalid_instances()
        list_of_broken_images = invalid_instances_df.filename.to_list()
        print("Количество битых:", len(list_of_broken_images))

        outliers_df = self.fd.outliers()
        list_of_outliers = outliers_df[
            outliers_df.distance < 0.68
        ].filename_outlier.tolist()
        print("Количество выбросов:", len(list_of_outliers))

        stats_df = self.fd.img_stats()
        dark_images = stats_df[stats_df["mean"] < 13]
        list_of_dark_images = dark_images.filename.to_list()
        print("Количество тёмных:", len(list_of_dark_images))

        bright_images = stats_df[stats_df["mean"] > 221]
        list_of_bright_images = bright_images.filename.to_list()
        print("Количество ярких:", len(list_of_bright_images))

        blurry_images = stats_df[stats_df["blur"] < 30]
        list_of_blurry_images = blurry_images.filename.to_list()
        print("Количество размытых:", len(list_of_blurry_images))

        connected_components_grouped_df = self.fd.connected_components_grouped()
        list_of_duplicates = []
        for file_list in connected_components_grouped_df.files:
            list_of_duplicates.extend(file_list[1:])
        print("Количество дубликатов:", len(list_of_duplicates))

        files_to_del = set(
            list_of_duplicates
            + list_of_broken_images
            + list_of_outliers
            + list_of_dark_images
            + list_of_bright_images
            + list_of_blurry_images
        )
        print("Общее количество изображений для удаления:", len(files_to_del))

        return files_to_del

    def remove_files(self, files_to_del):
        count = 0
        for file_name_to_del in files_to_del:
            file_path = os.path.join(self.config.source_folder, file_name_to_del)
            if os.path.exists(file_path):
                os.remove(file_path)
                count += 1
        print(f"Из необходимых {len(files_to_del)} удалено {count} файлов.")

    def move_remaining_files(self):
        source_folder = Path(self.config.source_folder)
        target_folder = Path(self.config.target_folder)
        target_folder.mkdir(parents=True, exist_ok=True)

        count = 0
        for root, _, files in os.walk(source_folder):
            for file in files:
                source_path = os.path.join(root, file)
                target_path = os.path.join(target_folder, file)
                shutil.move(source_path, target_path)
                count += 1
        print(f"Перемещено в {target_folder} {count} файлов.")


def run(
    source_folder="~/data/frames/",
    target_folder="~/data/nodupes/",
    fastdup_work_dir="/tmp/fastdup_work/",
):
    config = Config(source_folder, target_folder, fastdup_work_dir)
    run_config(config)


def run_config(config):
    remover = DuplicateRemover(config)
    remover.run_fastdup()
    files_to_remove = remover.get_files_to_remove()
    remover.remove_files(files_to_remove)
    remover.move_remaining_files()


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для удаления дубликатов файлов"
    )

    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Путь к исходной папке",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Путь к папке для сохранения файлов без дубликатов",
    )
    parser.add_argument(
        "--fastdup_work_dir",
        type=str,
        default=config.fastdup_work_dir,
        help="Рабочая папка для хранения артефактов fastdup",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
