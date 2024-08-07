import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from transliterate import translit


class Config:
    def __init__(
        self,
        source_folder="~/data/video/",
        target_folder="~/data/frames/",
        fps=-1.0,
        subfolders=True,
        format="png",
    ):
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.subfolders = subfolders
        self.fps = fps
        self.format = format


class VideoFrameExtractor:
    def __init__(self, config):
        self.config = config
        self.source_folder = Path(config.source_folder).expanduser()
        self.target_folder = Path(config.target_folder).expanduser()

        self.target_folder = self._transliterate_path(self.target_folder)
        self.target_folder.mkdir(parents=True, exist_ok=True)
        self.frames_file = self.target_folder / "frames.txt"
        self.processed_videos = self._load_processed_videos()

    def _load_processed_videos(self):
        processed_videos = set()
        if self.frames_file.exists():
            with open(self.frames_file, "r", encoding="utf-8") as f:
                for line in f:
                    video_name = line.split('"')[1]
                    processed_videos.add(video_name)
        return processed_videos

    def _transliterate_path(self, path):
        parts = path.parts
        parts = [translit(part, "ru", reversed=True) for part in parts]
        return Path(*parts)

    def get_files(self, source, ext):
        pattern = f"**/*.{ext}" if self.config.subfolders else f"*.{ext}"
        return list(source.glob(pattern))

    def extract_frames(self):
        video_files = self.get_files(self.source_folder, "mp4") + self.get_files(
            self.source_folder, "mov"
        )

        with open(self.frames_file, "a", encoding="utf-8") as frames_out:
            for video_path in video_files:
                if video_path.name in self.processed_videos:
                    print(f"Уже обработан, пропущено: {video_path.name}")
                    continue

                relative_path = video_path.parent.relative_to(self.source_folder)
                output_subfolder = self.target_folder / relative_path
                output_subfolder = self._transliterate_path(output_subfolder)
                output_subfolder.mkdir(parents=True, exist_ok=True)

                cap = cv2.VideoCapture(str(video_path))
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_interval = (
                    int(video_fps / self.config.fps) if self.config.fps > 0 else 1
                )
                frame_count = 0

                for frame_number in tqdm(
                    range(0, total_frames, frame_interval),
                    desc=f"Обработка {video_path.name}",
                    unit="кадр",
                ):
                    if frame_interval > 1:
                        if frame_interval < 10:
                            for _ in range(frame_interval - 1):
                                ret, frame = cap.read()
                                if not ret:
                                    break
                        else:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    output_filename = f"{translit(video_path.stem, 'ru', reversed=True)}.{frame_count:06d}.{self.config.format}"
                    output_path = output_subfolder / output_filename

                    cv2.imwrite(str(output_path), frame)

                    frames_out.write(f'"{video_path.name}", "{output_filename}"\n')

                cap.release()
                print(f"Обработано: {video_path}")


def run(**kwargs):
    return run_config(Config(**kwargs))


def run_config(config):
    extractor = VideoFrameExtractor(config)
    extractor.extract_frames()
    return extractor


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для извлечения кадров из видео"
    )

    parser.add_argument(
        "--source_folder",
        type=str,
        default=config.source_folder,
        help="Исходная папка с видеофайлами",
    )
    parser.add_argument(
        "--target_folder",
        type=str,
        default=config.target_folder,
        help="Папка с результатами для извлеченных кадров",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.fps,
        help="Частота извлечения кадров (-1 для всех кадров)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default=config.format,
        choices=["png", "jpg"],
        help="Формат сохранения извлеченных кадров (png или jpg)",
    )

    parser.add_argument(
        "--subfolders",
        action="store_true",
        help="Сканировать входную папку с подпапками",
    )
    parser.set_defaults(subfolders=config.subfolders)

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    run_config(config)


if __name__ == "__main__":
    main()
