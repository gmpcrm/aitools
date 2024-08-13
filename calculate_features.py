import argparse
import gzip
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os
import shutil
import pandas as pd
from tqdm import tqdm  # Импортируем tqdm


class Config:
    def __init__(
        self,
        source_file="~/data/features/features.gz",
        target_file="~/data/features/distances.gz",
        distances="cos",
        similarity=0.95,
        deleted_folder=None,
        statistics_file=None,
    ):
        self.source_file = source_file
        self.target_file = target_file
        self.distances = distances
        self.similarity = similarity
        self.deleted_folder = deleted_folder
        self.statistics_file = statistics_file


class DistanceCalculator:
    def __init__(self, config):
        self.config = config

    def load_embeddings(self):
        with gzip.open(self.config.source_file, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings

    def calculate_distances(self, embeddings):
        features = [e[0].flatten() for e in embeddings]  # Flatten each embedding
        metric = self.config.distances
        if metric == "cos":
            distances = 1 - cosine_similarity(features)
        else:
            distances = pairwise_distances(features, metric=metric)
        return distances

    def save_distances(self, distances):
        with gzip.open(self.config.target_file, "wb") as f:
            pickle.dump(distances, f)

    def load_distances(self):
        if not os.path.exists(self.config.target_file):
            return None

        with gzip.open(self.config.target_file, "rb") as f:
            distances = pickle.load(f)

        return distances

    def remove_duplicates(self, embeddings, distances):
        to_delete = set()
        for i in tqdm(range(len(distances)), desc="Calculating duplicates"):
            for j in range(i + 1, len(distances)):
                if distances[i][j] < self.config.similarity:
                    to_delete.add(j)

        if self.config.deleted_folder:
            if not os.path.exists(self.config.deleted_folder):
                os.makedirs(self.config.deleted_folder)

            deleted = 0
            for idx in to_delete:
                file_path = embeddings[idx][2]  # Получаем путь к файлу из эмбеддингов
                file_name = os.path.basename(file_path)
                target_path = os.path.join(self.config.deleted_folder, file_name)
                if os.path.exists(
                    file_path
                ):  # Проверка существования файла перед перемещением
                    shutil.move(file_path, target_path)
                    print(f"Файл {file_name} перемещен в {self.config.deleted_folder}")
                    deleted += 1

            print(f"Удалено {deleted} дубликатов")
        else:
            print(
                f"Найдено {len(to_delete)} дубликатов, но удаление не выполнено, так как папка для удаления не указана."
            )

    def save_statistics(self, embeddings, distances):
        file_names = [os.path.basename(e[2]) for e in embeddings]
        df = pd.DataFrame(distances, index=file_names, columns=file_names)
        df.to_csv(self.config.statistics_file)
        print(f"Статистика сохранена в {self.config.statistics_file}")

    def process(self):
        embeddings = self.load_embeddings()
        print(f"Всего файлов в эмбеддингах: {len(embeddings)}")
        distances = self.load_distances()
        if distances is None:
            distances = self.calculate_distances(embeddings)
            self.save_distances(distances)
            print(f"Расстояния сохранены в {self.config.target_file}")
        self.remove_duplicates(embeddings, distances)
        if self.config.statistics_file:
            self.save_statistics(embeddings, distances)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    calculator = DistanceCalculator(config)
    calculator.process()
    return calculator


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для расчета расстояний между эмбеддингами и удаления дубликатов"
    )

    parser.add_argument(
        "--source_file",
        default=config.source_file,
        type=str,
        help="Путь к gzip файлу с эмбеддингами",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=config.target_file,
        help="Путь к gzip файлу для сохранения расстояний",
    )
    parser.add_argument(
        "--distances",
        type=str,
        choices=["cos", "euclidean", "manhattan"],
        default=config.distances,
        help="Тип расстояния: cos, euclidean, manhattan",
    )
    parser.add_argument(
        "--similarity",
        type=float,
        default=config.similarity,
        help="Порог схожести для удаления дубликатов",
    )
    parser.add_argument(
        "--deleted_folder",
        type=str,
        default=config.deleted_folder,
        help="Папка для перемещения удаленных дубликатов",
    )
    parser.add_argument(
        "--statistics_file",
        type=str,
        default=None,
        help="Путь к CSV файлу для сохранения статистики",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
