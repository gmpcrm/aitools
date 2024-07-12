import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from transliterate import translit


def transliterate_path(path):
    parts = path.parts
    transliterated_parts = [translit(part, "ru", reversed=True) for part in parts]
    return Path(*transliterated_parts)


class Config:
    def __init__(
        self,
        input_folder="~/data/video",
        output_folder="~/data/frames",
        fps=-1.0,
        subfolders=True,
    ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.fps = fps
        self.subfolders = subfolders


class VideoFrameExtractor:
    def __init__(self, config):
        self.config = config
        self.input_folder = Path(config.input_folder).expanduser()
        self.output_folder = Path(config.output_folder).expanduser()

        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.frames_file = self.output_folder / "frames.txt"
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
        transliterated_parts = [translit(part, "ru", reversed=True) for part in parts]
        return Path(*transliterated_parts)

    def extract_frames(self):
        video_files = (
            list(self.input_folder.glob("**/*.mp4"))
            if self.config.subfolders
            else list(self.input_folder.glob("*.mp4"))
        )

        with open(self.frames_file, "a", encoding="utf-8") as frames_out:
            for video_path in video_files:
                if video_path.name in self.processed_videos:
                    print(f"Уже обработан, пропущено: {video_path.name}")
                    continue

                relative_path = video_path.parent.relative_to(self.input_folder)
                output_subfolder = self.output_folder / relative_path
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
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_count += 1
                    output_filename = f"{video_path.stem}.{frame_count:03d}.png"
                    output_path = output_subfolder / output_filename

                    cv2.imwrite(str(output_path), frame)

                    frames_out.write(f'"{video_path.name}", "{output_filename}"\n')

                cap.release()
                print(f"Обработано: {video_path}")


def run(
    input_folder="~/data/video",
    output_folder="~/data/frames",
    fps=-1.0,
    subfolders=True,
):
    config = Config(input_folder, output_folder, fps, subfolders)
    extractor = VideoFrameExtractor(config)
    extractor.extract_frames()


def main():
    config = Config()
    parser = argparse.ArgumentParser(
        description="Утилита для извлечения кадров из видео"
    )

    parser.add_argument(
        "--input_folder",
        type=str,
        default=config.input_folder,
        help="Исходная папка с видеофайлами",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=config.output_folder,
        help="Папка с результатами для извлеченных кадров",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=config.fps,
        help="Частота извлечения кадров (-1 для всех кадров)",
    )

    parser.add_argument(
        "--subfolders",
        action="store_true",
        help="Сканировать входную папку с подпапками",
    )
    parser.set_defaults(subfolders=config.subfolders)

    args = parser.parse_args()
    config.__dict__.update(vars(args))

    extractor = VideoFrameExtractor(config)
    extractor.extract_frames()


if __name__ == "__main__":
    main()
