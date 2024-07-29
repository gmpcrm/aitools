import re
import os
import sys
import cv2
import pickle
import gzip
import time
import numpy as np
from tqdm import tqdm
import gc
import argparse


class Config:
    def __init__(
        self,
        source_folder="~/data/source",
        target_file="~/data/features/features.gz",
        mode="images",
        fps=1,
        width=299,
        height=299,
    ):
        self.source_folder = source_folder
        self.target_file = target_file
        self.mode = mode
        self.fps = fps
        self.width = width
        self.height = height


class ExtractFeatures:
    def __init__(self, config):
        from tensorflow.keras.applications import InceptionV3
        from tensorflow.keras.models import Model

        self.config = config
        self.base_model = InceptionV3(weights="imagenet", include_top=False)
        self.feature_extraction_model = Model(
            inputs=self.base_model.input, outputs=self.base_model.layers[-2].output
        )
        if not os.path.exists(os.path.dirname(self.config.target_file)):
            os.makedirs(os.path.dirname(self.config.target_file))

    def extract_features_from_frames(self, frames):
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        from tensorflow.keras.preprocessing import image

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
                features_list.append((feature, frame_num, path))
            batch_frames.clear()

    def extract_features_from_video(self, input_video_path, batch_size=128):
        if not os.path.isfile(input_video_path):
            print("Файл не найден: ", input_video_path)
            return None

        print("Обрабатывается файл: ", input_video_path)
        video = cv2.VideoCapture(input_video_path)
        if not video.isOpened():
            print("Не удалось открыть видео: ", input_video_path)
            return None

        fps = video.get(cv2.CAP_PROP_FPS)
        step = int(fps / self.config.fps) if self.config.fps else 1

        features_list = []
        batch_frames = []

        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame_number in tqdm(range(total_frames), desc="Обработка кадров"):
            ret, frame = video.read()
            if not ret:
                break

            if frame_number % step == 0:
                batch_frames.append((frame, frame_number, input_video_path))
                if len(batch_frames) == batch_size:
                    self.process_batches(batch_frames, features_list)

        self.process_batches(batch_frames, features_list)

        video.release()
        return features_list

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

    def save_features(self, folder):
        all_features = []

        if self.config.mode == "video":
            video_files = []
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.lower().endswith(".mp4") or file.lower().endswith(".mov"):
                        video_files.append(os.path.join(root, file))

            for video_file in video_files:
                start_time = time.time()
                features_list = self.extract_features_from_video(video_file)
                if features_list is not None:
                    all_features.extend(features_list)

                end_time = time.time()
                time_taken = end_time - start_time
                print(
                    f"Время, затраченное на извлечение признаков: {time_taken} секунд"
                )
        else:
            features_list = self.extract_features_from_images(folder)

            if features_list is not None:
                all_features.extend(features_list)

        with gzip.open(self.config.target_file, "wb") as f:
            pickle.dump(all_features, f)

    def process_all_folders(self):
        self.save_features(self.config.source_folder)


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    extractor = ExtractFeatures(config)
    extractor.process_all_folders()
    return extractor


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для извлечения признаков из видео и изображений"
    )

    parser.add_argument(
        "--source_folder",
        default=config.source_folder,
        type=str,
        help="Путь к папке с видео или изображениями",
    )
    parser.add_argument(
        "--target_file",
        type=str,
        default=config.target_file,
        help="Путь к файлу для сохранения эмбеддингов",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["video", "images"],
        default=config.mode,
        help="Режим обработки: video или images",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=config.fps,
        help="Частота кадров для обработки (только для видео)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=config.width,
        help="Ширина изображения для изменения размера",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=config.height,
        help="Высота изображения для изменения размера",
    )

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
