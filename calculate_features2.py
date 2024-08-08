import argparse
import gzip
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import os
import shutil
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import cv2
import gc
import time
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image


class Config:
    def __init__(
        self,
        source_folder="~/data/source",
        distances="cos",
        max_count=300,
        move=False,
        statistics=False,
        width=299,
        height=299,
    ):
        self.source_folder = source_folder
        self.distances = distances
        self.max_count = max_count
        self.move = move
        self.statistics = statistics
        self.width = width
        self.height = height


class DistanceCalculator:
    def __init__(self, config):
        self.config = config
        self.base_model = InceptionV3(weights="imagenet", include_top=False)
        self.feature_extraction_model = Model(
            inputs=self.base_model.input, outputs=self.base_model.layers[-2].output
        )

    def extract_features_from_frames(self, frames):
        batch = []
        for frame in frames:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.width, self.config.height))
            x = image.img_to_array(img)
            x = preprocess_input(x)
            batch.append(x)

        batch = np.array(batch)
        features = self.feature_extraction_model.predict(batch, verbose=0)
        del batch
        gc.collect()

        return features

    def process_batches(self, batch_frames, features_list):
        if batch_frames:
            frames, frame_nums, paths = zip(*batch_frames)
            features = self.extract_features_from_frames(frames)
            for feature, frame_num, path in zip(features, frame_nums, paths):
                features_list.append((feature, frame_num, Path(path)))
            batch_frames.clear()

    def extract_features_from_images(self, image_folder, batch_size=128):
        image_files = [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]

        features_list = []
        batch_frames = []

        for idx, image_file in enumerate(
            tqdm(image_files, desc="Обработка изображений")
        ):
            frame = cv2.imread(image_file)
            if frame is None:
                continue
            batch_frames.append((frame, 0, image_file))
            if len(batch_frames) == batch_size:
                self.process_batches(batch_frames, features_list)

        self.process_batches(batch_frames, features_list)

        return features_list

    def save_features(self, folder, cluster_index):
        features = self.extract_features_from_images(folder)
        features_path = folder.parent / f"cluster_{cluster_index:04d}.features.gz"
        with gzip.open(features_path, "wb") as f:
            pickle.dump(features, f)
        return features

    def load_features(self, cluster_path, cluster_index):
        features_path = cluster_path.parent / f"cluster_{cluster_index:04d}.features.gz"
        if not features_path.exists():
            print(f"Считаем новые признаки")
            return self.save_features(cluster_path, cluster_index)
        with gzip.open(features_path, "rb") as f:
            print(f"Загружаем признаки")
            features = pickle.load(f)
        return features

    def calculate_distances(self, features):
        feature_vectors = [f[0].flatten() for f in features]
        metric = self.config.distances
        if metric == "cos":
            distances = 1 - cosine_similarity(feature_vectors)
        else:
            distances = pairwise_distances(feature_vectors, metric=metric)
        return distances

    def save_distances(self, distances, cluster_index, source_folder):
        distances_path = source_folder / f"cluster_{cluster_index:04d}.distances.gz"
        with gzip.open(distances_path, "wb") as f:
            pickle.dump(distances, f)

    def load_distances(self, cluster_path, cluster_index):
        distances_path = (
            cluster_path.parent / f"cluster_{cluster_index:04d}.distances.gz"
        )
        if not distances_path.exists():
            features = self.load_features(cluster_path, cluster_index)
            print(f"Считаем новые расстояния")
            distances = self.calculate_distances(features)
            self.save_distances(distances, cluster_index, cluster_path.parent)
        else:
            with gzip.open(distances_path, "rb") as f:
                print(f"Загружаем расстояния")
                distances = pickle.load(f)
        return distances

    def select_diverse_files(self, distances):
        selected_indices = [0]
        while len(selected_indices) < self.config.max_count and len(
            selected_indices
        ) < len(distances):
            remaining_indices = set(range(len(distances))) - set(selected_indices)
            max_min_distance = -1
            next_index = None
            for i in remaining_indices:
                min_distance = min(distances[i][j] for j in selected_indices)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    next_index = i
            if next_index is not None:
                selected_indices.append(next_index)
        return selected_indices

    def remove_duplicates(self, features, distances, cluster_path, cluster_index):
        selected_indices = self.select_diverse_files(distances)
        to_delete = set(range(len(features))) - set(selected_indices)

        if self.config.move:
            deleted_folder = (
                cluster_path.parent / f"cluster_{cluster_index:04d}.deleted"
            )
            deleted_folder.mkdir(parents=True, exist_ok=True)

            deleted = 0
            for idx in to_delete:
                file_path = features[idx][2]
                file_name = file_path.name
                target_path = deleted_folder / file_name
                if file_path.exists():
                    shutil.move(file_path, target_path)
                    # print(f"Файл {file_name} перемещен в {deleted_folder}")
                    deleted += 1

            print(f"Удалено {deleted} дубликатов")
        else:
            print(
                f"Найдено {len(to_delete)} дубликатов, но удаление не выполнено, так как параметр move установлен в False."
            )

    def save_statistics(self, features, distances, cluster_path, cluster_index):
        file_names = [f[2].name for f in features]
        df = pd.DataFrame(distances, index=file_names, columns=file_names)
        statistics_path = cluster_path.parent / f"statictics_{cluster_index:04d}.csv"
        df.to_csv(statistics_path)
        print(f"Статистика сохранена в {statistics_path}")

    def process_cluster(self, cluster_path, cluster_index):
        features = self.load_features(cluster_path, cluster_index)
        print(f"Всего файлов в признаках: {len(features)}")
        distances = self.load_distances(cluster_path, cluster_index)
        self.remove_duplicates(features, distances, cluster_path, cluster_index)
        if self.config.statistics:
            self.save_statistics(features, distances, cluster_path, cluster_index)

    def process(self):
        source_folder = Path(self.config.source_folder).expanduser()
        for cluster_index, cluster_folder in enumerate(
            [f for f in source_folder.iterdir() if f.is_dir()], start=1
        ):
            if cluster_folder.name == "min":
                print(f"Пропуск кластера: {cluster_folder}")
                continue

            print(f"Обработка кластера: {cluster_folder}")
            self.process_cluster(cluster_folder, cluster_index)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    calculator = DistanceCalculator(config)
    calculator.process()
    return calculator


def main():
    config = Config()
    config.source_folder = "/cluster"
    config.distances = "cos"
    config.max_count = 300
    config.move = True

    parser = argparse.ArgumentParser(
        description="Утилита для расчета расстояний между признаками и удаления дубликатов"
    )

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к папке с кластерами",
    )
    parser.add_argument(
        "--distances",
        type=str,
        choices=["cos", "euclidean", "manhattan"],
        default=config.distances,
        help="Тип расстояния: cos, euclidean, manhattan",
    )
    parser.add_argument(
        "--max_count",
        type=int,
        default=config.max_count,
        help="Максимальное количество оставшихся файлов в каждом кластере",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        default=config.move,
        help="Перемещать файлы в папки вида cluster_0001.deleted",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        default=config.statistics,
        help="Сохранять статистику",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.width,
        help="Ширина изображения для рассчета признаков",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.height,
        help="Высота изображения для рассчета признаков",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
